"""
Unit tests for software ray tracing.

Tests GPU software RT against trimesh CPU ray tracing for correctness.
Generates random meshes and rays, runs both implementations, and compares results.

Usage:
    python test_software_rt.py
"""

import numpy as np
import trimesh
import slangpy as spy
from pathlib import Path
from bvh_builder import (
    BVHBuilder, build_blas_for_mesh, build_tlas_for_instances,
    compute_mesh_world_aabb, AABB, LEAF_FLAG
)
import struct
import sys
import time

PROJECT_DIR = Path(__file__).parent

# Vertex stride: float3 pos + float3 normal + float2 uv = 32 bytes
VERTEX_STRIDE = 32


def create_device():
    """Create a SlangPy device for testing."""
    return spy.Device(
        compiler_options={
            "include_paths": [PROJECT_DIR],
        },
    )


def make_vertex_buffer(positions: np.ndarray, normals: np.ndarray = None) -> np.ndarray:
    """
    Create vertex buffer in the format expected by Slang: (V, 8) float32.
    [pos.x, pos.y, pos.z, normal.x, normal.y, normal.z, uv.x, uv.y]
    """
    V = positions.shape[0]
    vertices = np.zeros((V, 8), dtype=np.float32)
    vertices[:, :3] = positions.astype(np.float32)
    if normals is not None:
        vertices[:, 3:6] = normals.astype(np.float32)
    return vertices


def generate_random_mesh(num_tris: int, seed: int = 42, scale: float = 2.0):
    """Generate a random triangle soup mesh."""
    rng = np.random.RandomState(seed)
    # Random triangles in [-scale, scale]^3
    positions = (rng.rand(num_tris * 3, 3).astype(np.float32) - 0.5) * 2.0 * scale
    indices = np.arange(num_tris * 3, dtype=np.uint32).reshape(-1, 3)
    
    # Compute normals
    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    face_normals = face_normals / norms
    
    # Assign face normal to each vertex
    normals = np.zeros_like(positions)
    for i in range(3):
        normals[indices[:, i]] = face_normals
    
    vertices = make_vertex_buffer(positions, normals)
    return vertices, indices


def generate_icosphere_mesh(subdivisions: int = 2):
    """Generate an icosphere mesh using trimesh."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    positions = np.array(mesh.vertices, dtype=np.float32)
    indices = np.array(mesh.faces, dtype=np.uint32)
    
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    vertices = make_vertex_buffer(positions, normals)
    return vertices, indices


def generate_random_rays(num_rays: int, seed: int = 123, origin_range: float = 5.0):
    """Generate random rays for testing."""
    rng = np.random.RandomState(seed)
    origins = (rng.rand(num_rays, 3).astype(np.float32) - 0.5) * 2.0 * origin_range
    
    # Random directions (normalized)
    directions = rng.randn(num_rays, 3).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    directions = directions / norms
    
    return origins, directions


def generate_targeted_rays(positions: np.ndarray, indices: np.ndarray, 
                           num_rays: int, seed: int = 456):
    """Generate rays that are likely to hit the mesh (aimed at triangle centroids)."""
    rng = np.random.RandomState(seed)
    
    num_tris = indices.shape[0]
    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    
    origins = np.zeros((num_rays, 3), dtype=np.float32)
    directions = np.zeros((num_rays, 3), dtype=np.float32)
    
    for i in range(num_rays):
        # Pick a random triangle centroid as target
        tri_idx = rng.randint(0, num_tris)
        target = centroids[tri_idx]
        
        # Random origin at some distance
        offset = rng.randn(3).astype(np.float32)
        offset_norm = np.linalg.norm(offset)
        if offset_norm < 1e-8:
            offset = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            offset_norm = 1.0
        offset = offset / offset_norm * (2.0 + rng.rand() * 3.0)
        
        origin = target + offset
        direction = target - origin
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-8:
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            direction = direction / dir_norm
        
        origins[i] = origin
        directions[i] = direction
    
    return origins, directions


def trimesh_ray_test(positions: np.ndarray, indices: np.ndarray, 
                     ray_origins: np.ndarray, ray_directions: np.ndarray,
                     transforms: list = None):
    """
    Run ray tracing using trimesh for reference.
    
    For multi-instance scenes, transforms each mesh to world space.
    Returns per-ray: (hit_flag, t, hit_position, tri_index, instance_id)
    """
    num_rays = ray_origins.shape[0]
    results = {
        'hit': np.zeros(num_rays, dtype=bool),
        't': np.full(num_rays, -1.0, dtype=np.float32),
        'tri_index': np.full(num_rays, -1, dtype=np.int32),
        'instance_id': np.full(num_rays, -1, dtype=np.int32),
        'hit_pos': np.zeros((num_rays, 3), dtype=np.float32),
    }
    
    if transforms is None:
        transforms = [np.eye(4, dtype=np.float32)]
    
    # For each instance, transform vertices and test rays
    for inst_id, transform in enumerate(transforms):
        # Transform positions to world space
        pos = positions[:, :3] if positions.shape[1] > 3 else positions
        ones = np.ones((pos.shape[0], 1), dtype=np.float32)
        pos_h = np.concatenate([pos, ones], axis=1)
        world_pos = (transform @ pos_h.T).T[:, :3]
        
        mesh = trimesh.Trimesh(vertices=world_pos, faces=indices, process=False)
        
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )
        
        if len(locations) == 0:
            continue
        
        # Compute t values
        for i in range(len(locations)):
            ray_id = index_ray[i]
            hit_pos = locations[i]
            t_val = np.linalg.norm(hit_pos - ray_origins[ray_id])
            
            # Keep closest hit
            if not results['hit'][ray_id] or t_val < results['t'][ray_id]:
                results['hit'][ray_id] = True
                results['t'][ray_id] = t_val
                results['tri_index'][ray_id] = index_tri[i]
                results['instance_id'][ray_id] = inst_id
                results['hit_pos'][ray_id] = hit_pos
    
    return results


def gpu_ray_test(device, vertices: np.ndarray, indices: np.ndarray,
                 ray_origins: np.ndarray, ray_directions: np.ndarray,
                 transforms: list = None, instance_descs: list = None,
                 mesh_descs: list = None):
    """
    Run ray tracing on GPU using software RT.
    Returns per-ray: (hit_flag, t, bary_u, bary_v, instance_id, prim_index)
    """
    num_rays = ray_origins.shape[0]
    num_tris = indices.shape[0]
    
    # Default: single mesh, single instance, identity transform
    if transforms is None:
        transforms = [np.eye(4, dtype=np.float32)]
    if instance_descs is None:
        instance_descs = [(0, 0, 0, 0)]  # (mesh_id, material_id, transform_id, flags)
    if mesh_descs is None:
        mesh_descs = [(vertices.shape[0], num_tris * 3, 0, 0)]  # (vcount, icount, voffset, ioffset)
    
    num_instances = len(instance_descs)
    
    # Build BVH
    blas_nodes, prim_order = build_blas_for_mesh(vertices, indices)
    
    # Build TLAS
    instance_aabbs = []
    for inst in instance_descs:
        mesh_id, mat_id, xform_id, flags = inst
        md = mesh_descs[mesh_id]
        aabb = compute_mesh_world_aabb(vertices, indices, transforms[xform_id])
        instance_aabbs.append(aabb)
    
    tlas_nodes, instance_order = build_tlas_for_instances(instance_aabbs)
    
    # Compute inverse transforms
    inverse_transforms = [np.linalg.inv(t).astype(np.float32) for t in transforms]
    
    # Create GPU buffers
    ray_origins_flat = ray_origins.flatten().astype(np.float32)
    ray_directions_flat = ray_directions.flatten().astype(np.float32)
    
    buf_ray_origins = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=ray_origins_flat, label="ray_origins")
    buf_ray_dirs = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=ray_directions_flat, label="ray_dirs")
    
    # BVH nodes: flatten (N, 8) uint32 to 1D uint32 for StructuredBuffer<uint>
    buf_tlas_nodes = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=tlas_nodes.flatten().astype(np.uint32), label="tlas_nodes")
    buf_tlas_inst_order = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=instance_order.astype(np.uint32), label="tlas_inst_order")
    buf_blas_nodes = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=blas_nodes.flatten().astype(np.uint32), label="blas_nodes")
    
    blas_offsets = np.array([0], dtype=np.uint32)
    blas_prim_offsets = np.array([0], dtype=np.uint32)
    buf_blas_offsets = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=blas_offsets, label="blas_offsets")
    buf_blas_prim_indices = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=prim_order.astype(np.uint32), label="blas_prim_indices")
    buf_blas_prim_offsets = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=blas_prim_offsets, label="blas_prim_offsets")
    
    buf_vertices = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=vertices.flatten().astype(np.float32), label="vertices")
    # Flatten indices to 1D uint32
    indices_flat = indices.flatten().astype(np.uint32)
    buf_indices = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=indices_flat, label="indices")
    
    # MeshDesc: (vertex_count, index_count, vertex_offset, index_offset) as uint32
    mesh_desc_data = np.array(mesh_descs, dtype=np.uint32).flatten()
    buf_mesh_descs = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=mesh_desc_data, label="mesh_descs")
    
    # InstanceDesc: (mesh_id, material_id, transform_id, instance_flags) as uint32
    inst_desc_data = np.array(instance_descs, dtype=np.uint32).flatten()
    buf_inst_descs = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=inst_desc_data, label="inst_descs")
    
    # Transforms as float4x4 (row-major, 16 floats each)
    xform_data = np.stack(transforms).astype(np.float32)
    buf_transforms = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=xform_data, label="transforms")
    
    inv_xform_data = np.stack(inverse_transforms).astype(np.float32)
    buf_inv_transforms = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=inv_xform_data, label="inv_transforms")
    
    # Output buffer: 7 floats per ray
    output_size = num_rays * 7 * 4  # float32
    buf_hit_results = device.create_buffer(
        size=output_size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        label="hit_results")
    
    # Debug output buffer: 32 floats (required by shader)
    buf_debug = device.create_buffer(
        size=32 * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        label="debug_output")
    
    # Load and dispatch shader using pipeline + ShaderCursor (like path_tracer.py)
    program = device.load_program("test_software_rt.slang", ["compute_main"])
    pipeline = device.create_compute_pipeline(program)
    
    command_encoder = device.create_command_encoder()
    with command_encoder.begin_compute_pass() as pass_encoder:
        shader_object = pass_encoder.bind_pipeline(pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor.g_ray_origins = buf_ray_origins
        cursor.g_ray_directions = buf_ray_dirs
        cursor.g_tlas_nodes = buf_tlas_nodes
        cursor.g_tlas_instance_order = buf_tlas_inst_order
        cursor.g_blas_nodes = buf_blas_nodes
        cursor.g_blas_offsets = buf_blas_offsets
        cursor.g_blas_prim_indices = buf_blas_prim_indices
        cursor.g_blas_prim_offsets = buf_blas_prim_offsets
        cursor.g_vertices = buf_vertices
        cursor.g_indices = buf_indices
        cursor.g_mesh_descs = buf_mesh_descs
        cursor.g_instance_descs = buf_inst_descs
        cursor.g_transforms = buf_transforms
        cursor.g_inverse_transforms = buf_inv_transforms
        cursor.g_hit_results = buf_hit_results
        cursor.g_debug_output = buf_debug
        cursor.g_num_rays = num_rays
        cursor.g_instance_count = num_instances
        pass_encoder.dispatch(thread_count=[num_rays, 1, 1])
    device.submit_command_buffer(command_encoder.finish())
    
    # Read back results
    result_data = buf_hit_results.to_numpy().view(np.float32)
    result_data = result_data.reshape(num_rays, 7)
    
    return {
        'hit': result_data[:, 0] > 0.5,
        't': result_data[:, 1],
        'bary_u': result_data[:, 2],
        'bary_v': result_data[:, 3],
        'instance_id': result_data[:, 4].astype(np.int32),
        'prim_index': result_data[:, 5].astype(np.int32),
    }


def compare_results(gpu_results, trimesh_results, num_rays, test_name, 
                    t_tolerance=1e-3, pos_tolerance=1e-3):
    """Compare GPU and trimesh ray tracing results."""
    gpu_hit = gpu_results['hit']
    tm_hit = trimesh_results['hit']
    
    # Count hit/miss agreement
    agree = np.sum(gpu_hit == tm_hit)
    disagree = num_rays - agree
    
    both_hit = gpu_hit & tm_hit
    num_both_hit = np.sum(both_hit)
    
    gpu_only = gpu_hit & ~tm_hit
    tm_only = ~gpu_hit & tm_hit
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"  Total rays: {num_rays}")
    print(f"  GPU hits: {np.sum(gpu_hit)}, trimesh hits: {np.sum(tm_hit)}")
    print(f"  Agreement: {agree}/{num_rays} ({100.0*agree/num_rays:.1f}%)")
    print(f"  Both hit: {num_both_hit}")
    print(f"  GPU only: {np.sum(gpu_only)}, trimesh only: {np.sum(tm_only)}")
    
    # Compare t values for both-hit rays
    t_match_count = 0
    t_max_diff = 0.0
    if num_both_hit > 0:
        gpu_t = gpu_results['t'][both_hit]
        tm_t = trimesh_results['t'][both_hit]
        
        t_diff = np.abs(gpu_t - tm_t)
        t_rel_diff = t_diff / np.maximum(np.abs(tm_t), 1e-6)
        t_match = (t_diff < t_tolerance) | (t_rel_diff < t_tolerance)
        t_match_count = np.sum(t_match)
        t_max_diff = np.max(t_diff)
        
        print(f"  T value match: {t_match_count}/{num_both_hit} (tol={t_tolerance})")
        print(f"  T max absolute diff: {t_max_diff:.6f}")
        
        if t_max_diff > t_tolerance * 10:
            worst_idx = np.argmax(t_diff)
            print(f"  Worst t mismatch: gpu_t={gpu_t[worst_idx]:.6f}, tm_t={tm_t[worst_idx]:.6f}")
    
    # Summary
    passed = (disagree <= max(1, num_rays * 0.02))  # Allow 2% disagreement
    if num_both_hit > 0:
        passed = passed and (t_match_count >= num_both_hit * 0.95)  # 95% t match
    
    status = "PASSED" if passed else "FAILED"
    print(f"  Result: {status}")
    return passed


# ============================================================================
# Test cases
# ============================================================================

def test_single_triangle(device):
    """Test with a single triangle."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    vertices = make_vertex_buffer(positions)
    
    # Rays: one hitting center, one missing
    origins = np.array([
        [0.25, 0.25, 1.0],   # Should hit
        [0.25, 0.25, -1.0],  # Should hit (from back)
        [2.0, 2.0, 1.0],     # Should miss
        [0.1, 0.1, 0.5],     # Should hit
    ], dtype=np.float32)
    directions = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions)
    tm_res = trimesh_ray_test(positions, indices, origins, directions)
    
    return compare_results(gpu_res, tm_res, len(origins), "Single Triangle")


def test_random_triangle_soup(device, num_tris=20, num_rays=50, seed=42):
    """Test with random triangle soup."""
    vertices, indices = generate_random_mesh(num_tris, seed=seed)
    positions = vertices[:, :3]
    
    # Mix of random + targeted rays
    r_origins, r_dirs = generate_random_rays(num_rays // 2, seed=seed + 1)
    t_origins, t_dirs = generate_targeted_rays(positions, indices, num_rays // 2, seed=seed + 2)
    
    origins = np.concatenate([r_origins, t_origins], axis=0)
    directions = np.concatenate([r_dirs, t_dirs], axis=0)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions)
    tm_res = trimesh_ray_test(positions, indices, origins, directions)
    
    return compare_results(gpu_res, tm_res, len(origins), 
                          f"Random Triangle Soup ({num_tris} tris, {len(origins)} rays)")


def test_icosphere(device, num_rays=50):
    """Test with icosphere mesh."""
    vertices, indices = generate_icosphere_mesh(subdivisions=1)
    positions = vertices[:, :3]
    
    r_origins, r_dirs = generate_random_rays(num_rays // 2, seed=789, origin_range=3.0)
    t_origins, t_dirs = generate_targeted_rays(positions, indices, num_rays // 2, seed=790)
    
    origins = np.concatenate([r_origins, t_origins], axis=0)
    directions = np.concatenate([r_dirs, t_dirs], axis=0)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions)
    tm_res = trimesh_ray_test(positions, indices, origins, directions)
    
    return compare_results(gpu_res, tm_res, len(origins),
                          f"Icosphere ({indices.shape[0]} tris, {len(origins)} rays)")


def test_multi_instance(device, num_rays=50):
    """Test with multiple instances of the same mesh at different transforms."""
    vertices, indices = generate_icosphere_mesh(subdivisions=1)
    positions = vertices[:, :3]
    
    # 3 instances with different transforms
    transforms = [
        np.eye(4, dtype=np.float32),  # identity
        np.array([  # translated
            [1, 0, 0, 3],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32),
        np.array([  # scaled + translated
            [0.5, 0, 0, -2],
            [0, 0.5, 0, 1],
            [0, 0, 0.5, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32),
    ]
    
    num_verts = vertices.shape[0]
    num_tris = indices.shape[0]
    
    instance_descs = [
        (0, 0, 0, 0),  # mesh_id=0, mat_id=0, xform_id=0, flags=0
        (0, 0, 1, 0),  # mesh_id=0, mat_id=0, xform_id=1
        (0, 0, 2, 0),  # mesh_id=0, mat_id=0, xform_id=2
    ]
    mesh_descs = [(num_verts, num_tris * 3, 0, 0)]
    
    # Generate rays spread across all instances
    origins, directions = generate_random_rays(num_rays, seed=999, origin_range=6.0)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions,
                           transforms=transforms, instance_descs=instance_descs,
                           mesh_descs=mesh_descs)
    tm_res = trimesh_ray_test(positions, indices, origins, directions,
                              transforms=transforms)
    
    return compare_results(gpu_res, tm_res, num_rays,
                          f"Multi-Instance (3 instances, {num_rays} rays)")


def test_axis_aligned_rays(device):
    """Test with axis-aligned rays (edge cases for AABB intersection)."""
    # Create a box-like mesh
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    positions = np.array(mesh.vertices, dtype=np.float32)
    indices = np.array(mesh.faces, dtype=np.uint32)
    vertices = make_vertex_buffer(positions)
    
    # Axis-aligned rays
    origins = np.array([
        [0, 0, 5],    # +Z looking down
        [0, 0, -5],   # -Z looking up
        [5, 0, 0],    # +X looking left
        [-5, 0, 0],   # -X looking right
        [0, 5, 0],    # +Y looking down
        [0, -5, 0],   # -Y looking up
        [0, 0, 5],    # Should miss (direction parallel)
        [3, 3, 5],    # Should miss (outside)
    ], dtype=np.float32)
    directions = np.array([
        [0, 0, -1],
        [0, 0, 1],
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [1, 0, 0],  # parallel to box face
        [0, 0, -1], # outside
    ], dtype=np.float32)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions)
    tm_res = trimesh_ray_test(positions, indices, origins, directions)
    
    return compare_results(gpu_res, tm_res, len(origins), "Axis-Aligned Rays (Box)")


def test_large_mesh(device, num_tris=100, num_rays=100):
    """Stress test with a larger mesh."""
    vertices, indices = generate_random_mesh(num_tris, seed=12345)
    positions = vertices[:, :3]
    
    r_origins, r_dirs = generate_random_rays(num_rays // 2, seed=111)
    t_origins, t_dirs = generate_targeted_rays(positions, indices, num_rays // 2, seed=222)
    
    origins = np.concatenate([r_origins, t_origins], axis=0)
    directions = np.concatenate([r_dirs, t_dirs], axis=0)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions)
    tm_res = trimesh_ray_test(positions, indices, origins, directions)
    
    return compare_results(gpu_res, tm_res, len(origins),
                          f"Large Mesh ({num_tris} tris, {len(origins)} rays)")


def test_degenerate_rays(device):
    """Test with near-degenerate cases: grazing angles, near-origin hits."""
    vertices, indices = generate_icosphere_mesh(subdivisions=1)
    positions = vertices[:, :3]
    
    # Rays at very grazing angles
    origins = np.array([
        [0, 1.01, 0],      # Just outside sphere, looking tangent
        [0, 0, 0],          # Origin inside sphere
        [0, 0, 100],        # Very far away
        [1e-6, 1e-6, 2],    # Nearly on-axis
    ], dtype=np.float32)
    directions = np.array([
        [1, 0, 0],          # Tangent
        [0, 0, 1],          # From inside
        [0, 0, -1],         # Far away aimed at center
        [0, 0, -1],         # Nearly on-axis
    ], dtype=np.float32)
    
    gpu_res = gpu_ray_test(device, vertices, indices, origins, directions)
    tm_res = trimesh_ray_test(positions, indices, origins, directions)
    
    return compare_results(gpu_res, tm_res, len(origins), 
                          "Degenerate Rays", t_tolerance=1e-2)


# ============================================================================
# BVH Builder unit tests (CPU only, no GPU needed)
# ============================================================================

def test_bvh_builder_basic():
    """Test that BVH builder produces valid tree structure."""
    print(f"\n{'='*60}")
    print(f"Test: BVH Builder Basic Validation")
    print(f"{'='*60}")
    
    positions = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 0, 0], [2, 0, 0], [1, 1, 0],
        [2, 0, 0], [3, 0, 0], [2, 1, 0],
    ], dtype=np.float32)
    indices = np.array([[0,1,2],[3,4,5],[6,7,8]], dtype=np.uint32)
    vertices = make_vertex_buffer(positions)
    
    bvh_nodes, prim_order = build_blas_for_mesh(vertices, indices)
    
    print(f"  Triangles: {indices.shape[0]}")
    print(f"  BVH nodes: {bvh_nodes.shape[0]}")
    print(f"  Prim order: {prim_order}")
    
    # Check all prims are accounted for
    all_prims = set()
    LEAF_MASK = np.uint32(0x7FFFFFFF)
    for i in range(bvh_nodes.shape[0]):
        right_val = bvh_nodes[i, 7]
        if right_val & np.uint32(LEAF_FLAG):
            count = int(right_val & LEAF_MASK)
            offset = int(bvh_nodes[i, 3])
            for j in range(count):
                all_prims.add(int(prim_order[offset + j]))
    
    expected = set(range(indices.shape[0]))
    passed = all_prims == expected
    print(f"  All primitives covered: {passed}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# Main
# ============================================================================

def main():
    print("Software Ray Tracing Unit Tests")
    print("=" * 60)
    
    # CPU-only tests first
    results = []
    results.append(("BVH Builder Basic", test_bvh_builder_basic()))
    
    # GPU tests
    print("\nInitializing GPU device...")
    t0 = time.time()
    device = create_device()
    print(f"Device created in {time.time()-t0:.2f}s")
    
    test_cases = [
        ("Single Triangle", lambda: test_single_triangle(device)),
        ("Random Triangle Soup", lambda: test_random_triangle_soup(device)),
        ("Icosphere", lambda: test_icosphere(device)),
        ("Multi-Instance", lambda: test_multi_instance(device)),
        ("Axis-Aligned Rays", lambda: test_axis_aligned_rays(device)),
        ("Large Mesh", lambda: test_large_mesh(device)),
        ("Degenerate Rays", lambda: test_degenerate_rays(device)),
    ]
    
    for name, test_fn in test_cases:
        try:
            t0 = time.time()
            passed = test_fn()
            elapsed = time.time() - t0
            results.append((name, passed))
            print(f"  Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"\n  Test '{name}' EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    passed = sum(1 for _, p in results if p)
    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
