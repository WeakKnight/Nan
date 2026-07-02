import common;
import scene;

RWTexture2D<float> g_output;
uniform float4x4 g_shadow_to_world;

float trace_static_shadow_depth(float3 ray_start, float3 ray_end)
{
    float3 ray_delta = ray_end - ray_start;
    float t_max = length(ray_delta);
    if (t_max <= 1e-6)
        return 1.0;

    Ray ray = Ray(ray_start, ray_delta / t_max, 0.0, t_max);
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;
    rayQuery.TraceRayInline(g_scene.tlas, RAY_FLAG_NONE, 0xff, ray.to_ray_desc());

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint instance_id = rayQuery.CandidateInstanceIndex();
            uint primitive_index = rayQuery.CandidatePrimitiveIndex();
            float2 bary = rayQuery.CandidateTriangleBarycentrics();

            InstanceDesc instance_desc = g_scene.instance_descs[instance_id];
            MaterialDesc material = g_scene.load_material_desc(instance_desc.material_id);

            if (material.alpha_mode == ALPHA_MODE_OPAQUE)
            {
                rayQuery.CommitNonOpaqueTriangleHit();
                continue;
            }

            Vertex v = g_scene.get_vertex(instance_id, primitive_index, bary);
            float opacity = g_scene.get_opacity(material, v.uv);

            if (g_scene.eval_alpha_test(material, opacity, v.position, g_scene.frame_index))
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        return saturate(rayQuery.CommittedRayT() / t_max);

    return 1.0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_output.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    float2 uv = (float2(pixel) + 0.5) / float2(dim);
    float3 shadow_start = float3(uv, 0.0);
    float3 shadow_end = float3(uv, 1.0);
    float3 world_start = mul(g_shadow_to_world, float4(shadow_start, 1.0)).xyz;
    float3 world_end = mul(g_shadow_to_world, float4(shadow_end, 1.0)).xyz;

    g_output[pixel] = trace_static_shadow_depth(world_start, world_end);
}
