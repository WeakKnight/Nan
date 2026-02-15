"""
Minimal test for software ray tracing - single triangle only.
Tests the basic pipeline before running full test suite.
"""

import numpy as np
import slangpy as spy
from pathlib import Path
from bvh_builder import build_blas_for_mesh, build_tlas_for_instances, compute_mesh_world_aabb, AABB

PROJECT_DIR = Path(__file__).parent


def main():
    print("Minimal Software RT Test")
    print("=" * 40)
    
    # Single triangle in XY plane at z=0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    
    # Vertex buffer: (V, 8) float32
    V = positions.shape[0]
    vertices = np.zeros((V, 8), dtype=np.float32)
    vertices[:, :3] = positions
    
    # Build BVH
    blas_nodes, prim_order = build_blas_for_mesh(vertices, indices)
    print(f"BLAS: {blas_nodes.shape[0]} nodes, prim_order={prim_order}")
    
    # Print BVH node details
    for i in range(blas_nodes.shape[0]):
        raw = blas_nodes[i]
        aabb_min = np.array([raw[0], raw[1], raw[2]], dtype=np.uint32).view(np.float32)
        left = raw[3]
        aabb_max = np.array([raw[4], raw[5], raw[6]], dtype=np.uint32).view(np.float32)
        right = raw[7]
        is_leaf = (right & np.uint32(0x80000000)) != 0
        print(f"  Node {i}: aabb_min={aabb_min}, left={left}, aabb_max={aabb_max}, right=0x{right:08x}, leaf={is_leaf}")
        if is_leaf:
            count = int(right & np.uint32(0x7FFFFFFF))
            print(f"    -> prim_offset={left}, prim_count={count}")
    
    # Single instance, identity transform
    transform = np.eye(4, dtype=np.float32)
    instance_aabbs = [compute_mesh_world_aabb(vertices, indices, transform)]
    tlas_nodes, instance_order = build_tlas_for_instances(instance_aabbs)
    print(f"TLAS: {tlas_nodes.shape[0]} nodes, instance_order={instance_order}")
    
    # Rays: one hitting, one missing
    ray_origins = np.array([
        [0.25, 0.25, 1.0],   # Should hit
        [2.0, 2.0, 1.0],     # Should miss
    ], dtype=np.float32)
    ray_directions = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)
    num_rays = 2
    
    # Create device
    device = spy.Device(compiler_options={"include_paths": [PROJECT_DIR]})
    print(f"Device created: {device}")
    
    # Create buffers
    buf_ray_origins = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=ray_origins.flatten())
    buf_ray_dirs = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=ray_directions.flatten())
    
    buf_tlas_nodes = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=tlas_nodes.flatten().astype(np.uint32))
    buf_tlas_inst_order = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=instance_order.astype(np.uint32))
    buf_blas_nodes = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=blas_nodes.flatten().astype(np.uint32))
    buf_blas_offsets = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([0], dtype=np.uint32))
    buf_blas_prim_indices = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=prim_order.astype(np.uint32))
    buf_blas_prim_offsets = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([0], dtype=np.uint32))
    buf_vertices = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=vertices.flatten().astype(np.float32))
    buf_indices = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=indices.flatten().astype(np.uint32))
    
    # MeshDesc: (vertex_count, index_count, vertex_offset, index_offset)
    mesh_desc_data = np.array([V, 3, 0, 0], dtype=np.uint32)
    buf_mesh_descs = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=mesh_desc_data)
    
    # InstanceDesc: (mesh_id, material_id, transform_id, flags)
    inst_desc_data = np.array([0, 0, 0, 0], dtype=np.uint32)
    buf_inst_descs = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=inst_desc_data)
    
    # Transforms
    buf_transforms = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=transform)
    inv_transform = np.linalg.inv(transform).astype(np.float32)
    buf_inv_transforms = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=inv_transform)
    
    # Output buffer
    output_size = num_rays * 7 * 4
    buf_hit_results = device.create_buffer(
        size=output_size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
    
    # Debug output buffer: 32 floats
    buf_debug = device.create_buffer(
        size=32 * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
    
    # Load shader and dispatch
    print("Loading shader...")
    program = device.load_program("test_software_rt.slang", ["compute_main"])
    pipeline = device.create_compute_pipeline(program)
    
    print("Dispatching...")
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
        cursor.g_instance_count = 1
        pass_encoder.dispatch(thread_count=[num_rays, 1, 1])
    device.submit_command_buffer(command_encoder.finish())
    
    print("Reading results...")
    result_data = buf_hit_results.to_numpy().view(np.float32).reshape(num_rays, 7)
    
    # Read debug output
    debug_data = buf_debug.to_numpy().view(np.float32)
    print(f"\n--- Debug Output (Ray 0) ---")
    print(f"  TLAS root: aabb_min=({debug_data[0]:.3f}, {debug_data[1]:.3f}, {debug_data[2]:.3f})")
    print(f"             aabb_max=({debug_data[3]:.3f}, {debug_data[4]:.3f}, {debug_data[5]:.3f})")
    print(f"             leaf={debug_data[6]:.0f} prim_count={debug_data[7]:.0f}")
    print(f"  TLAS AABB hit: {debug_data[8]:.0f}")
    print(f"  BLAS AABB hit: {debug_data[9]:.0f}")
    print(f"  Indices: [{debug_data[10]:.0f}, {debug_data[11]:.0f}, {debug_data[12]:.0f}]")
    print(f"  v0=({debug_data[13]:.3f}, {debug_data[14]:.3f}, {debug_data[15]:.3f})")
    print(f"  v1=({debug_data[16]:.3f}, {debug_data[17]:.3f}, {debug_data[18]:.3f})")
    print(f"  v2=({debug_data[19]:.3f}, {debug_data[20]:.3f}, {debug_data[21]:.3f})")
    print(f"  Direct tri hit: {debug_data[22]:.0f}, t={debug_data[23]:.4f}")
    print(f"  prim_indices[0]={debug_data[23]:.0f}")
    print(f"  blas_prim_offsets[0]={debug_data[24]:.0f}")
    print(f"  blas_offset={debug_data[25]:.0f}")
    print(f"  blas_root.prim_offset={debug_data[26]:.0f}")
    print(f"  instance_count={debug_data[27]:.0f}")
    print(f"  inv_dir=({debug_data[28]:.3f}, {debug_data[29]:.3f}, {debug_data[30]:.3f})")
    print(f"---")
    
    for i in range(num_rays):
        hit = result_data[i, 0] > 0.5
        t = result_data[i, 1]
        bary_u = result_data[i, 2]
        bary_v = result_data[i, 3]
        inst_id = int(result_data[i, 4])
        prim_idx = int(result_data[i, 5])
        print(f"  Ray {i}: hit={hit}, t={t:.4f}, bary=({bary_u:.4f}, {bary_v:.4f}), inst={inst_id}, prim={prim_idx}")
    
    # Expected: Ray 0 hits at t=1.0, Ray 1 misses
    ray0_hit = result_data[0, 0] > 0.5
    ray1_hit = result_data[1, 0] > 0.5
    ray0_t = result_data[0, 1]
    
    passed = ray0_hit and not ray1_hit and abs(ray0_t - 1.0) < 0.01
    print(f"\nResult: {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print(f"  Expected: ray0 hit=True t=1.0, ray1 hit=False")
        print(f"  Got:      ray0 hit={ray0_hit} t={ray0_t:.4f}, ray1 hit={ray1_hit}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
