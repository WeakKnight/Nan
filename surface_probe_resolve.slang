import common;
import scene;
import surface_probes;

RWTexture2D<float4> g_output;
ByteAddressBuffer g_probe_irradiance;
ByteAddressBuffer g_probe_self_hit_counters;
StructuredBuffer<SurfaceProbe> g_surface_probes;
StructuredBuffer<uint4> g_surface_probe_nodes;
StructuredBuffer<SurfaceProbeInstanceInfo> g_surface_probe_instances;
StructuredBuffer<uint4> g_triangle_vertex_probes;
StructuredBuffer<uint> g_vertex_lighting_rgbm;
StructuredBuffer<float4> g_vertex_lighting_target;
StructuredBuffer<uint4> g_vertex_lighting_triangle_map;
StructuredBuffer<uint2> g_vertex_lighting_instance_offsets;
uniform uint g_debug_view;
uniform float g_max_density_multiplier;
uniform uint g_min_gather_count;
uniform uint g_use_vertex_fallback;
uniform uint g_use_vertex_lighting;
uniform uint g_vertex_lighting_built;
uniform float g_vertex_lighting_rgbm_range;

float3 gather_count_debug_color(uint count)
{
    if (count == 0)
        return float3(1.0, 0.0, 1.0);
    if (count < 4)
        return lerp(
            float3(1.0, 0.0, 0.0),
            float3(1.0, 0.5, 0.0),
            float(count - 1) / 3.0);
    if (count < 8)
        return lerp(
            float3(1.0, 1.0, 0.0),
            float3(0.0, 1.0, 0.0),
            float(count - 4) / 4.0);
    if (count < 16)
        return lerp(
            float3(0.0, 1.0, 0.0),
            float3(0.0, 1.0, 1.0),
            float(count - 8) / 8.0);
    if (count < 24)
        return lerp(
            float3(0.0, 1.0, 1.0),
            float3(0.0, 0.0, 1.0),
            float(count - 16) / 8.0);
    return lerp(
        float3(0.0, 0.0, 1.0),
        float3(1.0, 1.0, 1.0),
        float(min(count, 32u) - 24) / 8.0);
}

float3 support_debug_color(float support_f)
{
    float value = saturate(support_f);
    if (value < 0.25)
        return lerp(
            float3(1.0, 0.0, 1.0),
            float3(1.0, 0.0, 0.0),
            value * 4.0);
    if (value < 0.5)
        return lerp(
            float3(1.0, 0.0, 0.0),
            float3(1.0, 1.0, 0.0),
            (value - 0.25) * 4.0);
    if (value < 0.75)
        return lerp(
            float3(1.0, 1.0, 0.0),
            float3(0.0, 1.0, 0.0),
            (value - 0.5) * 4.0);
    return lerp(
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.25, 1.0),
        (value - 0.75) * 4.0);
}

float3 density_debug_color(float support_f)
{
    float maximum = max(g_max_density_multiplier, 1.0);
    float density = min(
        maximum,
        1.0 / max(saturate(support_f), 1.0 / maximum));
    float normalized = maximum > 1.0
        ? log2(density) / log2(maximum)
        : 0.0;
    return support_debug_color(1.0 - saturate(normalized));
}

float3 probe_self_hit_debug_color(float rate)
{
    if (rate <= 0.0)
        return float3(0.0, 0.0, 0.25);
    float log_rate = log2(max(rate, 1e-6)) / log2(10.0);
    float value = saturate((log_rate + 6.0) / 5.0);
    if (value < 1.0 / 3.0)
    {
        return lerp(
            float3(0.0, 0.0, 0.25),
            float3(0.0, 1.0, 1.0),
            value * 3.0);
    }
    if (value < 2.0 / 3.0)
    {
        return lerp(
            float3(0.0, 1.0, 1.0),
            float3(1.0, 1.0, 0.0),
            (value - 1.0 / 3.0) * 3.0);
    }
    return lerp(
        float3(1.0, 1.0, 0.0),
        float3(1.0, 0.0, 0.0),
        (value - 2.0 / 3.0) * 3.0);
}

float3 surface_probe_sample_irradiance(
    SurfaceProbeInstanceInfo instance_info,
    float3 query_position,
    float3 query_normal,
    out uint gathered_count,
    out float support_f,
    out float reconstruction_weight_sum,
    out float reconstruction_ess,
    out float probe_self_hit_rate)
{
    uint candidate_count;
    uint candidate_indices[SURFACE_PROBE_MAX_CANDIDATES];
    float candidate_weights[SURFACE_PROBE_MAX_CANDIDATES];
    uint gathered_indices[SURFACE_PROBE_MAX_GATHER];
    float gathered_weights[SURFACE_PROBE_MAX_GATHER];

    surface_probe_gather(
        g_surface_probe_nodes,
        g_surface_probes,
        instance_info,
        query_position,
        query_normal,
        1.0,
        candidate_count,
        candidate_indices,
        candidate_weights);
    surface_probe_select_strongest(
        candidate_count,
        candidate_indices,
        candidate_weights,
        gathered_count,
        gathered_indices,
        gathered_weights);

    if (gathered_count < clamp(g_min_gather_count, 1u, 32u))
    {
        uint fallback_count;
        uint fallback_indices[SURFACE_PROBE_MAX_GATHER];
        float fallback_weights[SURFACE_PROBE_MAX_GATHER];
        surface_probe_gather(
            g_surface_probe_nodes,
            g_surface_probes,
            instance_info,
            query_position,
            query_normal,
            2.0,
            candidate_count,
            candidate_indices,
            candidate_weights);
        surface_probe_select_strongest(
            candidate_count,
            candidate_indices,
            candidate_weights,
            fallback_count,
            fallback_indices,
            fallback_weights);
        surface_probe_merge_fallback(
            fallback_count,
            fallback_indices,
            fallback_weights,
            gathered_count,
            gathered_indices,
            gathered_weights);
    }

    float self_hit_sum = 0.0;
    float self_hit_weight_sum = 0.0;
    for (uint i = 0; i < gathered_count; i++)
    {
        uint probe_index = gathered_indices[i];
        uint sample_count = surface_probe_load_sample_count(
            g_probe_irradiance, probe_index);
        if (sample_count == 0u)
            continue;
        float rate = float(surface_probe_load_self_hit_count(
            g_probe_self_hit_counters, probe_index)) / float(sample_count);
        self_hit_sum += gathered_weights[i] * rate;
        self_hit_weight_sum += gathered_weights[i];
    }
    probe_self_hit_rate = self_hit_weight_sum > 1e-6
        ? self_hit_sum / self_hit_weight_sum
        : 0.0;

    return surface_probe_reconstruct_irradiance(
        g_probe_irradiance,
        g_surface_probes,
        gathered_count,
        gathered_indices,
        gathered_weights,
        support_f,
        reconstruction_weight_sum,
        reconstruction_ess);
}

bool surface_probe_sample_vertex_fallback(
    HitInfo hit,
    ShadingData shading,
    SurfaceProbeInstanceInfo instance_info,
    out float3 irradiance)
{
    uint side = shading.double_sided && !shading.front_facing ? 1u : 0u;
    uint map_offset = asuint(instance_info.params.w);
    uint4 anchors = g_triangle_vertex_probes[
        map_offset + hit.primitive_index * 2u + side];
    if ((anchors.w & 1u) == 0u)
    {
        irradiance = float3(0.0);
        return false;
    }

    float3 barycentrics = hit.get_barycentrics();
    uint3 indices = anchors.xyz;
    float3 accumulated = float3(0.0);
    float available_weight = 0.0;
    for (uint corner = 0; corner < 3; corner++)
    {
        uint probe_index = indices[corner];
        if (surface_probe_load_sample_count(
                g_probe_irradiance, probe_index) == 0)
        {
            continue;
        }
        float weight = barycentrics[corner];
        accumulated += weight * surface_probe_load_irradiance(
            g_probe_irradiance, probe_index);
        available_weight += weight;
    }
    if (available_weight <= 1e-6)
    {
        irradiance = float3(0.0);
        return false;
    }
    irradiance = accumulated / available_weight;
    return true;
}

float3 vertex_lighting_decode_rgbm(uint packed)
{
    float4 encoded = float4(
        packed & 255u,
        (packed >> 8) & 255u,
        (packed >> 16) & 255u,
        (packed >> 24) & 255u) / 255.0;
    return encoded.xyz * encoded.w * g_vertex_lighting_rgbm_range;
}

bool surface_probe_sample_vertex_lighting(
    HitInfo hit,
    ShadingData shading,
    out float3 irradiance,
    out float confidence)
{
    if (g_vertex_lighting_built == 0)
    {
        irradiance = float3(0.0);
        confidence = 0.0;
        return false;
    }
    uint2 instance_info =
        g_vertex_lighting_instance_offsets[hit.instance_id];
    uint side = shading.double_sided && !shading.front_facing ? 1u : 0u;
    uint4 vertices = g_vertex_lighting_triangle_map[
        instance_info.x + hit.primitive_index * 2u + side];
    if (vertices.w == 0)
    {
        irradiance = float3(0.0);
        confidence = 0.0;
        return false;
    }
    float3 barycentrics = hit.get_barycentrics();
    irradiance =
        barycentrics.x * vertex_lighting_decode_rgbm(
            g_vertex_lighting_rgbm[vertices.x])
        + barycentrics.y * vertex_lighting_decode_rgbm(
            g_vertex_lighting_rgbm[vertices.y])
        + barycentrics.z * vertex_lighting_decode_rgbm(
            g_vertex_lighting_rgbm[vertices.z]);
    confidence =
        barycentrics.x * g_vertex_lighting_target[vertices.x].w
        + barycentrics.y * g_vertex_lighting_target[vertices.y].w
        + barycentrics.z * g_vertex_lighting_target[vertices.z].w;
    return true;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_thread_id.xy;
    uint2 dimensions;
    g_output.GetDimensions(dimensions.x, dimensions.y);
    if (any(pixel >= dimensions))
        return;

    Ray ray = g_scene.camera.compute_ray_pinhole(pixel, dimensions);
    HitInfo hit = g_scene.intersect(ray);
    if (!hit.valid)
    {
        bool beauty_like = g_debug_view == 0 || g_debug_view == 6;
        float3 miss_color = beauty_like
            ? g_scene.sample_sky_with_sun_disk(ray.dir)
            : float3(0.02);
        g_output[pixel] = float4(miss_color, 1.0);
        return;
    }

    ShadingData shading = g_scene.get_shading_data(hit, -ray.dir);
    bool beauty_like = g_debug_view == 0 || g_debug_view == 6;
    float3 realtime_sun = float3(0.0);
    if (beauty_like)
    {
        RNG preview_rng = RNG(pixel, g_scene.frame_index);
        realtime_sun = g_scene.sample_directional_light(
            shading, preview_rng.next_2d()).xyz;
    }
    SurfaceProbeInstanceInfo instance_info = surface_probe_get_instance(
        g_surface_probe_instances, hit.instance_id);
    if (instance_info.offsets.w == 0)
    {
        float3 empty_color = beauty_like
            ? shading.emissive + realtime_sun
            : gather_count_debug_color(0);
        g_output[pixel] = float4(empty_color, 1.0);
        return;
    }

    float3 query_normal = normalize(shading.face_normal);
    uint gathered_count;
    float support_f;
    float reconstruction_weight_sum;
    float reconstruction_ess;
    float probe_self_hit_rate;
    float3 irradiance = surface_probe_sample_irradiance(
        instance_info,
        shading.position,
        query_normal,
        gathered_count,
        support_f,
        reconstruction_weight_sum,
        reconstruction_ess,
        probe_self_hit_rate);
    float3 vertex_irradiance = float3(0.0);
    bool has_vertex_fallback = g_use_vertex_fallback != 0 &&
        surface_probe_sample_vertex_fallback(
            hit, shading, instance_info, vertex_irradiance);
    float target_gather = float(clamp(g_min_gather_count, 1u, 32u));
    float reconstruction_confidence = min(
        saturate(float(gathered_count) / target_gather),
        saturate(reconstruction_ess / target_gather));
    if (reconstruction_weight_sum <= 1e-6)
        reconstruction_confidence = 0.0;
    if (!has_vertex_fallback)
        reconstruction_confidence = 1.0;
    float vertex_fallback_weight = 1.0 - reconstruction_confidence;
    irradiance = lerp(
        vertex_irradiance, irradiance, reconstruction_confidence);
    if (g_debug_view == 1)
    {
        g_output[pixel] = float4(
            gather_count_debug_color(gathered_count), 1.0);
        return;
    }
    if (g_debug_view == 2)
    {
        g_output[pixel] = float4(support_debug_color(support_f), 1.0);
        return;
    }
    if (g_debug_view == 3)
    {
        g_output[pixel] = float4(density_debug_color(support_f), 1.0);
        return;
    }
    if (g_debug_view == 4)
    {
        g_output[pixel] = float4(
            support_debug_color(1.0 - vertex_fallback_weight), 1.0);
        return;
    }
    if (g_debug_view == 5)
    {
        g_output[pixel] = float4(
            probe_self_hit_debug_color(probe_self_hit_rate), 1.0);
        return;
    }
    float3 baked_vertex_irradiance;
    float vertex_lighting_confidence;
    bool has_vertex_lighting = surface_probe_sample_vertex_lighting(
        hit,
        shading,
        baked_vertex_irradiance,
        vertex_lighting_confidence);
    if (g_debug_view == 6)
    {
        float3 vertex_color = has_vertex_lighting
            ? shading.emissive
                + realtime_sun
                + shading.diffuse * INV_PI * baked_vertex_irradiance
            : float3(1.0, 0.0, 1.0);
        g_output[pixel] = float4(vertex_color, 1.0);
        return;
    }
    if (g_debug_view == 7)
    {
        g_output[pixel] = float4(
            has_vertex_lighting
                ? support_debug_color(vertex_lighting_confidence)
                : float3(1.0, 0.0, 1.0),
            1.0);
        return;
    }
    if (g_use_vertex_lighting != 0 && has_vertex_lighting)
        irradiance = baked_vertex_irradiance;
    float3 color = shading.emissive
        + realtime_sun
        + shading.diffuse * INV_PI * irradiance;
    g_output[pixel] = float4(color, 1.0);
}
