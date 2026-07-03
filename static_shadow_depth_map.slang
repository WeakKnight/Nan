import common;
import scene;

RWTexture2D<float> g_output;
RWTexture2D<float> g_output_second;
uniform float4x4 g_shadow_to_world;

float trace_static_shadow_hit_t(Ray ray)
{
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
        return rayQuery.CommittedRayT();

    return -1.0;
}

float2 trace_static_shadow_depth_layers(float3 ray_start, float3 ray_end)
{
    float3 ray_delta = ray_end - ray_start;
    float t_max = length(ray_delta);
    if (t_max <= 1e-6)
        return float2(1.0);

    float3 ray_dir = ray_delta / t_max;
    float first_t = trace_static_shadow_hit_t(Ray(ray_start, ray_dir, 0.0, t_max));
    if (first_t < 0.0)
        return float2(1.0);

    float second_t = trace_static_shadow_hit_t(Ray(ray_start, ray_dir, min(first_t + 1e-4, t_max), t_max));
    if (second_t < 0.0)
        second_t = t_max;

    return saturate(float2(first_t, max(second_t, first_t)) / t_max);
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

    float2 depth_layers = trace_static_shadow_depth_layers(world_start, world_end);
    g_output[pixel] = depth_layers.x;
    g_output_second[pixel] = depth_layers.y;
}
