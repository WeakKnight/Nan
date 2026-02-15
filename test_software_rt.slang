/**
 * Test compute shader for software ray tracing.
 * 
 * Executes batch ray queries against a scene BVH and outputs hit results
 * to a structured buffer for CPU-side validation against trimesh.
 */

import common;
import software_rt;

// Input: rays to test
StructuredBuffer<float> g_ray_origins;     // (N*3) flat float array
StructuredBuffer<float> g_ray_directions;  // (N*3) flat float array

// Scene BVH data
StructuredBuffer<uint> g_tlas_nodes;            // raw uint buffer, 8 uints per BVH node
StructuredBuffer<uint> g_tlas_instance_order;
StructuredBuffer<uint> g_blas_nodes;            // raw uint buffer, 8 uints per BVH node
StructuredBuffer<uint> g_blas_offsets;
StructuredBuffer<uint> g_blas_prim_indices;
StructuredBuffer<uint> g_blas_prim_offsets;

StructuredBuffer<float> g_vertices;          // raw float buffer, 8 floats per vertex
StructuredBuffer<uint> g_indices;
StructuredBuffer<uint> g_mesh_descs;          // raw uint buffer, 4 uints per MeshDesc
StructuredBuffer<uint> g_instance_descs;      // raw uint buffer, 4 uints per InstanceDesc
StructuredBuffer<float4x4> g_transforms;
StructuredBuffer<float4x4> g_inverse_transforms;

// Output: hit results
// Each ray produces 7 floats: [hit(0/1), t, bary_u, bary_v, instance_id, prim_index, padding]
RWStructuredBuffer<float> g_hit_results;

// Debug output: 32 floats for ray 0 diagnostics
RWStructuredBuffer<float> g_debug_output;

uniform uint g_num_rays;
uniform uint g_instance_count;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint ray_idx = tid.x;
    if (ray_idx >= g_num_rays)
        return;

    // Read ray
    float3 origin = float3(
        g_ray_origins[ray_idx * 3 + 0],
        g_ray_origins[ray_idx * 3 + 1],
        g_ray_origins[ray_idx * 3 + 2]
    );
    float3 direction = float3(
        g_ray_directions[ray_idx * 3 + 0],
        g_ray_directions[ray_idx * 3 + 1],
        g_ray_directions[ray_idx * 3 + 2]
    );

    // Build scene buffers
    SWRTSceneBuffers scene;
    scene.tlas_nodes = g_tlas_nodes;
    scene.tlas_instance_order = g_tlas_instance_order;
    scene.blas_nodes = g_blas_nodes;
    scene.blas_offsets = g_blas_offsets;
    scene.blas_prim_indices = g_blas_prim_indices;
    scene.blas_prim_offsets = g_blas_prim_offsets;
    scene.raw_vertices = g_vertices;
    scene.indices = g_indices;
    scene.raw_mesh_descs = g_mesh_descs;
    scene.raw_instance_descs = g_instance_descs;
    scene.transforms = g_transforms;
    scene.inverse_transforms = g_inverse_transforms;
    scene.instance_count = g_instance_count;

    // Debug: write diagnostic data for ray 0
    if (ray_idx == 0)
    {
        // Read TLAS root
        BVHNode tlas_root = load_bvh_node(g_tlas_nodes, 0);
        g_debug_output[0] = tlas_root.aabb_min.x;
        g_debug_output[1] = tlas_root.aabb_min.y;
        g_debug_output[2] = tlas_root.aabb_min.z;
        g_debug_output[3] = tlas_root.aabb_max.x;
        g_debug_output[4] = tlas_root.aabb_max.y;
        g_debug_output[5] = tlas_root.aabb_max.z;
        g_debug_output[6] = tlas_root.is_leaf() ? 1.0 : 0.0;
        g_debug_output[7] = float(tlas_root.get_prim_count());
        
        // Test AABB intersection directly
        float3 inv_dir = float3(
            abs(direction.x) > 1e-8 ? 1.0 / direction.x : (direction.x >= 0.0 ? 1e8 : -1e8),
            abs(direction.y) > 1e-8 ? 1.0 / direction.y : (direction.y >= 0.0 ? 1e8 : -1e8),
            abs(direction.z) > 1e-8 ? 1.0 / direction.z : (direction.z >= 0.0 ? 1e8 : -1e8)
        );
        bool tlas_aabb_hit = ray_aabb_intersect(origin, inv_dir, 0.0, 1000.0, tlas_root.aabb_min, tlas_root.aabb_max);
        g_debug_output[8] = tlas_aabb_hit ? 1.0 : 0.0;
        
        // Test BLAS AABB
        uint blas_off = g_blas_offsets[0];
        BVHNode blas_root = load_bvh_node(g_blas_nodes, blas_off);
        bool blas_aabb_hit = ray_aabb_intersect(origin, inv_dir, 0.0, 1000.0, blas_root.aabb_min, blas_root.aabb_max);
        g_debug_output[9] = blas_aabb_hit ? 1.0 : 0.0;
        
        // Direct triangle test - read vertices manually to check layout
        uint idx0 = g_indices[0];
        uint idx1 = g_indices[1];
        uint idx2 = g_indices[2];
        g_debug_output[10] = float(idx0);
        g_debug_output[11] = float(idx1);
        g_debug_output[12] = float(idx2);
        
        float3 vt0_pos = swrt_load_vertex_position(g_vertices, idx0);
        float3 vt1_pos = swrt_load_vertex_position(g_vertices, idx1);
        float3 vt2_pos = swrt_load_vertex_position(g_vertices, idx2);
        g_debug_output[13] = vt0_pos.x;
        g_debug_output[14] = vt0_pos.y;
        g_debug_output[15] = vt0_pos.z;
        g_debug_output[16] = vt1_pos.x;
        g_debug_output[17] = vt1_pos.y;
        g_debug_output[18] = vt1_pos.z;
        g_debug_output[19] = vt2_pos.x;
        g_debug_output[20] = vt2_pos.y;
        g_debug_output[21] = vt2_pos.z;
        
        TriHitResult tri_hit = ray_triangle_intersect(origin, direction, vt0_pos, vt1_pos, vt2_pos, 0.0, 1000.0);
        g_debug_output[22] = tri_hit.hit ? 1.0 : 0.0;
        g_debug_output[23] = tri_hit.t;
        
        // Test prim_indices
        g_debug_output[23] = float(g_blas_prim_indices[0]);
        g_debug_output[24] = float(g_blas_prim_offsets[0]);
        g_debug_output[25] = float(blas_off);
        g_debug_output[26] = float(blas_root.get_prim_offset());
        g_debug_output[27] = float(g_instance_count);
        
        // inv_dir values
        g_debug_output[28] = inv_dir.x;
        g_debug_output[29] = inv_dir.y;
        g_debug_output[30] = inv_dir.z;
        g_debug_output[31] = 0.0;
    }

    // Trace ray
    SWRT_RayQuery rq;
    swrt_trace_ray_inline(rq, origin, direction, 0.0, 1000.0);
    bool hit = swrt_proceed(rq, scene);

    // Write results
    uint out_offset = ray_idx * 7;
    if (hit)
    {
        g_hit_results[out_offset + 0] = 1.0;
        g_hit_results[out_offset + 1] = swrt_committed_ray_t(rq);
        g_hit_results[out_offset + 2] = swrt_committed_triangle_barycentrics(rq).x;
        g_hit_results[out_offset + 3] = swrt_committed_triangle_barycentrics(rq).y;
        g_hit_results[out_offset + 4] = float(swrt_committed_instance_index(rq));
        g_hit_results[out_offset + 5] = float(swrt_committed_primitive_index(rq));
        g_hit_results[out_offset + 6] = 0.0;
    }
    else
    {
        g_hit_results[out_offset + 0] = 0.0;
        g_hit_results[out_offset + 1] = -1.0;
        g_hit_results[out_offset + 2] = 0.0;
        g_hit_results[out_offset + 3] = 0.0;
        g_hit_results[out_offset + 4] = 0.0;
        g_hit_results[out_offset + 5] = 0.0;
        g_hit_results[out_offset + 6] = 0.0;
    }
}
