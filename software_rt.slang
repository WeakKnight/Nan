/**
 * Software Ray Tracing - GPU BVH2 traversal with DXR-like API.
 *
 * Provides a software implementation of inline ray tracing for platforms
 * without hardware RT support (e.g., Metal via Slang).
 *
 * API mirrors DXR inline ray tracing:
 *   SWRT_RayQuery           ~ RayQuery<>
 *   swrt_trace_ray_inline() ~ TraceRayInline()
 *   swrt_proceed()          ~ Proceed()
 *   swrt_committed_status() ~ CommittedStatus()
 *   etc.
 */

import common;

// ============================================================================
// Constants
// ============================================================================

static const uint SWRT_LEAF_FLAG = 0x80000000u;
static const uint SWRT_MAX_STACK_DEPTH = 32;
static const uint SWRT_MAX_TRAVERSAL_STEPS = 4096;  // safety limit to prevent GPU hang

// CommittedStatus values
static const uint SWRT_COMMITTED_NOTHING = 0;
static const uint SWRT_COMMITTED_TRIANGLE_HIT = 1;

// ============================================================================
// BVH Node - read from raw uint buffer to avoid struct padding issues
// Each node = 8 consecutive uint32 values:
//   [aabb_min.x_bits, aabb_min.y_bits, aabb_min.z_bits, left_or_prim_offset,
//    aabb_max.x_bits, aabb_max.y_bits, aabb_max.z_bits, right_or_prim_count]
// ============================================================================

struct BVHNode
{
    float3 aabb_min;
    uint left_or_prim_offset;
    float3 aabb_max;
    uint right_or_prim_count;

    bool is_leaf()
    {
        return (right_or_prim_count & SWRT_LEAF_FLAG) != 0;
    }

    uint get_prim_count()
    {
        return right_or_prim_count & (~SWRT_LEAF_FLAG);
    }

    uint get_prim_offset()
    {
        return left_or_prim_offset;
    }

    uint get_left_child()
    {
        return left_or_prim_offset;
    }

    uint get_right_child()
    {
        return right_or_prim_count;
    }
};

// Read a BVH node from a raw uint buffer (8 uints per node)
BVHNode load_bvh_node(StructuredBuffer<uint> raw_nodes, uint node_idx)
{
    uint base = node_idx * 8;
    BVHNode node;
    node.aabb_min = float3(
        asfloat(raw_nodes[base + 0]),
        asfloat(raw_nodes[base + 1]),
        asfloat(raw_nodes[base + 2])
    );
    node.left_or_prim_offset = raw_nodes[base + 3];
    node.aabb_max = float3(
        asfloat(raw_nodes[base + 4]),
        asfloat(raw_nodes[base + 5]),
        asfloat(raw_nodes[base + 6])
    );
    node.right_or_prim_count = raw_nodes[base + 7];
    return node;
}

// ============================================================================
// SWRT_RayQuery - equivalent to DXR RayQuery
// ============================================================================

struct SWRT_RayQuery
{
    // Ray parameters
    float3 origin;
    float3 direction;
    float t_min;
    float t_max;

    // Committed hit state
    bool has_committed_hit;
    float committed_t;
    float2 committed_bary;
    uint committed_instance_id;
    uint committed_primitive_index;

    // Internal: traversal done flag
    bool traversal_done;
};

// ============================================================================
// Ray-AABB intersection (slab method)
// ============================================================================

bool ray_aabb_intersect(float3 ray_origin, float3 ray_inv_dir, float t_min, float t_max,
                        float3 aabb_min, float3 aabb_max)
{
    float3 t0 = (aabb_min - ray_origin) * ray_inv_dir;
    float3 t1 = (aabb_max - ray_origin) * ray_inv_dir;

    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);

    float enter = max(max(tmin.x, tmin.y), max(tmin.z, t_min));
    float exit  = min(min(tmax.x, tmax.y), min(tmax.z, t_max));

    return enter <= exit;
}

// ============================================================================
// Ray-Triangle intersection (Moller-Trumbore)
// Returns: (hit, t, u, v) where barycentrics are (1-u-v, u, v)
// ============================================================================

struct TriHitResult
{
    bool hit;
    float t;
    float u;
    float v;
};

TriHitResult ray_triangle_intersect(
    float3 ray_origin, float3 ray_dir,
    float3 v0, float3 v1, float3 v2,
    float t_min, float t_max)
{
    TriHitResult result;
    result.hit = false;
    result.t = t_max;
    result.u = 0.0;
    result.v = 0.0;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(ray_dir, e2);
    float det = dot(e1, pvec);

    // No backface culling: check both sides
    if (abs(det) < 1e-8)
        return result;

    float inv_det = 1.0 / det;
    float3 tvec = ray_origin - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return result;

    float3 qvec = cross(tvec, e1);
    float v = dot(ray_dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return result;

    float t = dot(e2, qvec) * inv_det;
    if (t >= t_min && t < t_max)
    {
        result.hit = true;
        result.t = t;
        result.u = u;
        result.v = v;
    }

    return result;
}

// ============================================================================
// Vertex position reading from raw float buffer
// Each vertex = 8 floats: [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, u, v]
// This avoids struct padding issues with StructuredBuffer<Vertex> on Metal.
// ============================================================================

static const uint SWRT_VERTEX_STRIDE = 8;  // 8 floats per vertex

float3 swrt_load_vertex_position(StructuredBuffer<float> raw_vertices, uint vertex_index)
{
    uint base = vertex_index * SWRT_VERTEX_STRIDE;
    return float3(raw_vertices[base + 0], raw_vertices[base + 1], raw_vertices[base + 2]);
}

// Read MeshDesc from raw uint buffer (4 uints per MeshDesc)
struct SWRTMeshDesc
{
    uint vertex_count;
    uint index_count;
    uint vertex_offset;
    uint index_offset;
};

SWRTMeshDesc swrt_load_mesh_desc(StructuredBuffer<uint> raw, uint index)
{
    uint base = index * 4;
    SWRTMeshDesc md;
    md.vertex_count = raw[base + 0];
    md.index_count = raw[base + 1];
    md.vertex_offset = raw[base + 2];
    md.index_offset = raw[base + 3];
    return md;
}

// Read InstanceDesc from raw uint buffer (4 uints per InstanceDesc)
struct SWRTInstanceDesc
{
    uint mesh_id;
    uint material_id;
    uint transform_id;
    uint instance_flags;
};

SWRTInstanceDesc swrt_load_instance_desc(StructuredBuffer<uint> raw, uint index)
{
    uint base = index * 4;
    SWRTInstanceDesc id;
    id.mesh_id = raw[base + 0];
    id.material_id = raw[base + 1];
    id.transform_id = raw[base + 2];
    id.instance_flags = raw[base + 3];
    return id;
}

// ============================================================================
// BLAS traversal - traverse a single mesh BVH
// ============================================================================

struct BLASHitResult
{
    bool hit;
    float t;
    float2 bary;       // (u, v) matching DXR convention
    uint prim_index;    // original triangle index in the mesh
};

BLASHitResult traverse_blas(
    StructuredBuffer<uint> blas_nodes_raw,
    StructuredBuffer<uint> prim_indices_buffer,
    StructuredBuffer<float> raw_vertices,
    StructuredBuffer<uint> indices,
    uint vertex_offset,
    uint index_offset,
    float3 ray_origin,
    float3 ray_dir,
    float t_min,
    float t_max)
{
    BLASHitResult result;
    result.hit = false;
    result.t = t_max;
    result.bary = float2(0.0);
    result.prim_index = 0;

    float3 inv_dir = float3(
        abs(ray_dir.x) > 1e-8 ? 1.0 / ray_dir.x : (ray_dir.x >= 0.0 ? 1e8 : -1e8),
        abs(ray_dir.y) > 1e-8 ? 1.0 / ray_dir.y : (ray_dir.y >= 0.0 ? 1e8 : -1e8),
        abs(ray_dir.z) > 1e-8 ? 1.0 / ray_dir.z : (ray_dir.z >= 0.0 ? 1e8 : -1e8)
    );

    uint stack[SWRT_MAX_STACK_DEPTH];
    int stack_ptr = 0;
    stack[0] = 0;
    stack_ptr = 1;

    float current_t_max = t_max;
    uint steps = 0;

    while (stack_ptr > 0 && steps < SWRT_MAX_TRAVERSAL_STEPS)
    {
        steps++;
        stack_ptr--;
        uint node_idx = stack[stack_ptr];

        BVHNode node = load_bvh_node(blas_nodes_raw, node_idx);

        if (!ray_aabb_intersect(ray_origin, inv_dir, t_min, current_t_max, node.aabb_min, node.aabb_max))
            continue;

        if (node.is_leaf())
        {
            uint prim_offset = node.get_prim_offset();
            uint prim_count = node.get_prim_count();

            for (uint i = 0; i < prim_count && i < 8; i++)
            {
                uint reordered_idx = prim_offset + i;
                uint original_tri_idx = prim_indices_buffer[reordered_idx];

                uint base = index_offset + original_tri_idx * 3;
                uint i0 = indices[base + 0];
                uint i1 = indices[base + 1];
                uint i2 = indices[base + 2];

                float3 v0 = swrt_load_vertex_position(raw_vertices, vertex_offset + i0);
                float3 v1 = swrt_load_vertex_position(raw_vertices, vertex_offset + i1);
                float3 v2 = swrt_load_vertex_position(raw_vertices, vertex_offset + i2);

                TriHitResult tri_hit = ray_triangle_intersect(
                    ray_origin, ray_dir, v0, v1, v2, t_min, current_t_max
                );

                if (tri_hit.hit)
                {
                    current_t_max = tri_hit.t;
                    result.hit = true;
                    result.t = tri_hit.t;
                    result.bary = float2(tri_hit.u, tri_hit.v);
                    result.prim_index = original_tri_idx;
                }
            }
        }
        else
        {
            uint left = node.get_left_child();
            uint right = node.get_right_child();

            if (stack_ptr + 2 <= SWRT_MAX_STACK_DEPTH)
            {
                stack[stack_ptr] = right;
                stack_ptr++;
                stack[stack_ptr] = left;
                stack_ptr++;
            }
        }
    }

    return result;
}

// ============================================================================
// TLAS + BLAS full scene traversal
// ============================================================================

struct SWRTSceneBuffers
{
    StructuredBuffer<uint> tlas_nodes;          // raw uint buffer, 8 uints per node
    StructuredBuffer<uint> tlas_instance_order;

    // Per-BLAS data: all BLAS nodes concatenated, with offset table
    StructuredBuffer<uint> blas_nodes;          // raw uint buffer, 8 uints per node
    StructuredBuffer<uint> blas_offsets;        // blas_offsets[mesh_id] = start node index in blas_nodes
    StructuredBuffer<uint> blas_prim_indices;   // all reordered prim indices concatenated
    StructuredBuffer<uint> blas_prim_offsets;   // blas_prim_offsets[mesh_id] = start in blas_prim_indices

    StructuredBuffer<float> raw_vertices;     // raw float buffer, 8 floats per vertex
    StructuredBuffer<uint> indices;
    StructuredBuffer<uint> raw_mesh_descs;     // raw uint buffer, 4 uints per MeshDesc
    StructuredBuffer<uint> raw_instance_descs; // raw uint buffer, 4 uints per InstanceDesc
    StructuredBuffer<float4x4> transforms;
    StructuredBuffer<float4x4> inverse_transforms;

    uint instance_count;
};

// ============================================================================
// DXR-like API functions
// ============================================================================

void swrt_trace_ray_inline(
    inout SWRT_RayQuery rq,
    float3 origin,
    float3 direction,
    float t_min,
    float t_max)
{
    rq.origin = origin;
    rq.direction = direction;
    rq.t_min = t_min;
    rq.t_max = t_max;
    rq.has_committed_hit = false;
    rq.committed_t = t_max;
    rq.committed_bary = float2(0.0);
    rq.committed_instance_id = 0;
    rq.committed_primitive_index = 0;
    rq.traversal_done = false;
}

/**
 * Execute full scene traversal (TLAS + BLAS).
 * Returns true if any triangle was hit.
 *
 * This is the "proceed to end" variant - it finds the closest hit
 * across all instances in a single call. Equivalent to calling
 * Proceed() in a loop until it returns false.
 */
bool swrt_proceed(
    inout SWRT_RayQuery rq,
    SWRTSceneBuffers scene)
{
    if (rq.traversal_done)
        return false;

    rq.traversal_done = true;

    float3 inv_dir = float3(
        abs(rq.direction.x) > 1e-8 ? 1.0 / rq.direction.x : (rq.direction.x >= 0.0 ? 1e8 : -1e8),
        abs(rq.direction.y) > 1e-8 ? 1.0 / rq.direction.y : (rq.direction.y >= 0.0 ? 1e8 : -1e8),
        abs(rq.direction.z) > 1e-8 ? 1.0 / rq.direction.z : (rq.direction.z >= 0.0 ? 1e8 : -1e8)
    );

    // TLAS traversal stack
    uint tlas_stack[SWRT_MAX_STACK_DEPTH];
    int tlas_sp = 0;
    tlas_stack[0] = 0;
    tlas_sp = 1;

    float best_t = rq.t_max;
    uint steps = 0;

    while (tlas_sp > 0 && steps < SWRT_MAX_TRAVERSAL_STEPS)
    {
        steps++;
        tlas_sp--;
        uint tlas_node_idx = tlas_stack[tlas_sp];

        BVHNode tlas_node = load_bvh_node(scene.tlas_nodes, tlas_node_idx);

        if (!ray_aabb_intersect(rq.origin, inv_dir, rq.t_min, best_t,
                                tlas_node.aabb_min, tlas_node.aabb_max))
            continue;

        if (tlas_node.is_leaf())
        {
            // Leaf: test instance(s)
            uint inst_offset = tlas_node.get_prim_offset();
            uint inst_count = tlas_node.get_prim_count();

            for (uint ii = 0; ii < inst_count; ii++)
            {
                uint reordered_idx = inst_offset + ii;
                uint instance_id = scene.tlas_instance_order[reordered_idx];

                SWRTInstanceDesc inst = swrt_load_instance_desc(scene.raw_instance_descs, instance_id);
                SWRTMeshDesc mesh = swrt_load_mesh_desc(scene.raw_mesh_descs, inst.mesh_id);

                // Transform ray to object space
                float4x4 inv_xform = scene.inverse_transforms[inst.transform_id];
                float3 obj_origin = mul(inv_xform, float4(rq.origin, 1.0)).xyz;
                float3 obj_dir = mul(inv_xform, float4(rq.direction, 0.0)).xyz;

                // Note: obj_dir may not be unit length after non-uniform scale
                // We need to scale t accordingly
                float dir_len = length(obj_dir);
                if (dir_len < 1e-10)
                    continue;

                float3 obj_dir_normalized = obj_dir / dir_len;
                float obj_t_min = rq.t_min * dir_len;
                float obj_t_max = best_t * dir_len;

                // Get BLAS data
                uint blas_offset = scene.blas_offsets[inst.mesh_id];
                uint prim_offset = scene.blas_prim_offsets[inst.mesh_id];

                // Traverse BLAS in object space
                BLASHitResult blas_hit = traverse_blas_with_offset(
                    scene.blas_nodes,
                    scene.blas_prim_indices,
                    scene.raw_vertices,
                    scene.indices,
                    mesh.vertex_offset,
                    mesh.index_offset,
                    blas_offset,
                    prim_offset,
                    obj_origin,
                    obj_dir_normalized,
                    obj_t_min,
                    obj_t_max
                );

                if (blas_hit.hit)
                {
                    // Convert t back to world space
                    float world_t = blas_hit.t / dir_len;
                    if (world_t < best_t)
                    {
                        best_t = world_t;
                        rq.has_committed_hit = true;
                        rq.committed_t = world_t;
                        rq.committed_bary = blas_hit.bary;
                        rq.committed_instance_id = instance_id;
                        rq.committed_primitive_index = blas_hit.prim_index;
                    }
                }
            }
        }
        else
        {
            uint left = tlas_node.get_left_child();
            uint right = tlas_node.get_right_child();

            if (tlas_sp + 2 <= SWRT_MAX_STACK_DEPTH)
            {
                tlas_stack[tlas_sp] = right;
                tlas_sp++;
                tlas_stack[tlas_sp] = left;
                tlas_sp++;
            }
        }
    }

    return rq.has_committed_hit;
}

// ============================================================================
// BLAS traversal with offset (for concatenated BLAS nodes)
// ============================================================================

BLASHitResult traverse_blas_with_offset(
    StructuredBuffer<uint> blas_nodes_raw,
    StructuredBuffer<uint> prim_indices_buffer,
    StructuredBuffer<float> raw_vertices,
    StructuredBuffer<uint> indices,
    uint vertex_offset,
    uint index_offset,
    uint blas_node_offset,      // offset into concatenated blas_nodes (in node units)
    uint prim_indices_offset,   // offset into concatenated prim_indices
    float3 ray_origin,
    float3 ray_dir,
    float t_min,
    float t_max)
{
    BLASHitResult result;
    result.hit = false;
    result.t = t_max;
    result.bary = float2(0.0);
    result.prim_index = 0;

    float3 inv_dir = float3(
        abs(ray_dir.x) > 1e-8 ? 1.0 / ray_dir.x : (ray_dir.x >= 0.0 ? 1e8 : -1e8),
        abs(ray_dir.y) > 1e-8 ? 1.0 / ray_dir.y : (ray_dir.y >= 0.0 ? 1e8 : -1e8),
        abs(ray_dir.z) > 1e-8 ? 1.0 / ray_dir.z : (ray_dir.z >= 0.0 ? 1e8 : -1e8)
    );

    uint stack[SWRT_MAX_STACK_DEPTH];
    int stack_ptr = 0;
    stack[0] = 0;  // root node (relative to blas_node_offset)
    stack_ptr = 1;

    float current_t_max = t_max;
    uint steps = 0;

    while (stack_ptr > 0 && steps < SWRT_MAX_TRAVERSAL_STEPS)
    {
        steps++;
        stack_ptr--;
        uint local_node_idx = stack[stack_ptr];
        uint global_node_idx = blas_node_offset + local_node_idx;

        BVHNode node = load_bvh_node(blas_nodes_raw, global_node_idx);

        if (!ray_aabb_intersect(ray_origin, inv_dir, t_min, current_t_max, node.aabb_min, node.aabb_max))
            continue;

        if (node.is_leaf())
        {
            uint prim_offset = node.get_prim_offset();
            uint prim_count = node.get_prim_count();

            for (uint i = 0; i < prim_count && i < 8; i++)
            {
                uint global_prim_idx = prim_indices_offset + prim_offset + i;
                uint original_tri_idx = prim_indices_buffer[global_prim_idx];

                uint base = index_offset + original_tri_idx * 3;
                uint i0 = indices[base + 0];
                uint i1 = indices[base + 1];
                uint i2 = indices[base + 2];

                float3 v0 = swrt_load_vertex_position(raw_vertices, vertex_offset + i0);
                float3 v1 = swrt_load_vertex_position(raw_vertices, vertex_offset + i1);
                float3 v2 = swrt_load_vertex_position(raw_vertices, vertex_offset + i2);

                TriHitResult tri_hit = ray_triangle_intersect(
                    ray_origin, ray_dir, v0, v1, v2, t_min, current_t_max
                );

                if (tri_hit.hit)
                {
                    current_t_max = tri_hit.t;
                    result.hit = true;
                    result.t = tri_hit.t;
                    result.bary = float2(tri_hit.u, tri_hit.v);
                    result.prim_index = original_tri_idx;
                }
            }
        }
        else
        {
            uint left = node.get_left_child();
            uint right = node.get_right_child();

            if (stack_ptr + 2 <= SWRT_MAX_STACK_DEPTH)
            {
                stack[stack_ptr] = right;
                stack_ptr++;
                stack[stack_ptr] = left;
                stack_ptr++;
            }
        }
    }

    return result;
}

// ============================================================================
// Query functions (DXR-compatible naming)
// ============================================================================

uint swrt_committed_status(SWRT_RayQuery rq)
{
    return rq.has_committed_hit ? SWRT_COMMITTED_TRIANGLE_HIT : SWRT_COMMITTED_NOTHING;
}

float swrt_committed_ray_t(SWRT_RayQuery rq)
{
    return rq.committed_t;
}

float2 swrt_committed_triangle_barycentrics(SWRT_RayQuery rq)
{
    return rq.committed_bary;
}

uint swrt_committed_instance_index(SWRT_RayQuery rq)
{
    return rq.committed_instance_id;
}

uint swrt_committed_primitive_index(SWRT_RayQuery rq)
{
    return rq.committed_primitive_index;
}

// ============================================================================
// Visibility test (shadow ray) - returns 1.0 if visible, 0.0 if occluded
// ============================================================================

float swrt_get_visibility(
    float3 origin,
    float3 direction,
    float t_min,
    float t_max,
    SWRTSceneBuffers scene)
{
    SWRT_RayQuery rq;
    swrt_trace_ray_inline(rq, origin, direction, t_min, t_max);
    bool hit = swrt_proceed(rq, scene);
    return hit ? 0.0 : 1.0;
}
