import common;
import scene;
import surface_probes;
import surface_probe_field;
import surface_probe_prt_field;
import surface_probe_sh;

RWByteAddressBuffer g_probe_field_values;
RWByteAddressBuffer g_probe_sample_counts;
RWByteAddressBuffer g_probe_self_hit_counters;
RWByteAddressBuffer g_probe_radial_moments;
RWByteAddressBuffer g_probe_static_source_rgb;
StructuredBuffer<SurfaceProbe> g_surface_probes;
StructuredBuffer<SurfaceProbeInstanceInfo> g_surface_probe_instances;
StructuredBuffer<uint4> g_triangle_vertex_probes;
uniform uint g_probe_count;
uniform uint g_iteration;
uniform uint g_samples_per_probe;
uniform uint g_max_bounces;
uniform uint g_reset;

static const uint MAX_PROBE_SAMPLE_COUNT = 65535u;

struct SurfaceProbePrtPath
{
    Ray ray;
    float3 throughput;
    float3 static_source;
    RNG rng;
    uint source_instance_id;
    uint source_primitive_index;
    float source_origin_offset;
    uint self_hit_suspect_count;
    float first_hit_distance;
    float radial_max_distance;
};

uint load_cached_sample_count(uint probe_index)
{
    return surface_probe_load_sample_count(
        g_probe_sample_counts, probe_index);
}

void store_accumulated_prt(
    uint probe_index,
    float3 transport[SURFACE_PROBE_PRT_COEFFICIENT_COUNT],
    float3 static_source)
{
    uint previous_count = g_reset != 0
        ? 0u
        : load_cached_sample_count(probe_index);
    uint next_count = min(previous_count + 1u, MAX_PROBE_SAMPLE_COUNT);
    float weight = 1.0 / float(previous_count + 1u);
    for (uint coefficient = 0u;
         coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
         coefficient++)
    {
        float3 previous = previous_count == 0u
            ? float3(0.0)
            : surface_probe_load_prt_coefficient(
                g_probe_field_values, probe_index, coefficient);
        surface_probe_store_prt_coefficient(
            g_probe_field_values,
            probe_index,
            coefficient,
            lerp(previous, transport[coefficient], weight));
    }
    float3 previous_source = previous_count == 0u
        ? float3(0.0)
        : surface_probe_load_static_source(
            g_probe_static_source_rgb, probe_index);
    surface_probe_store_static_source(
        g_probe_static_source_rgb,
        probe_index,
        lerp(previous_source, static_source, weight));
    surface_probe_store_sample_count(
        g_probe_sample_counts, probe_index, next_count);
}

void store_accumulated_self_hit_count(
    uint probe_index,
    uint local_self_hit_count)
{
    uint destination = probe_index * SURFACE_PROBE_SELF_HIT_STRIDE;
    uint previous_count = g_reset != 0
        ? 0u
        : surface_probe_load_self_hit_count(
            g_probe_self_hit_counters, probe_index);
    g_probe_self_hit_counters.Store(
        destination,
        previous_count + local_self_hit_count);
}

void clear_radial_moments(uint probe_index)
{
    for (uint texel = 0u;
         texel < SURFACE_PROBE_RADIAL_TEXEL_COUNT;
         texel++)
    {
        surface_probe_store_radial_moments(
            g_probe_radial_moments,
            probe_index,
            texel,
            float2(0.0));
    }
}

float radial_texel_cosine_probability(uint texel)
{
    uint2 coord = uint2(
        texel % SURFACE_PROBE_RADIAL_DEPTH_DIM,
        texel / SURFACE_PROBE_RADIAL_DEPTH_DIM);
    bool horizontal_border =
        coord.x == 0u || coord.x == SURFACE_PROBE_RADIAL_DEPTH_DIM - 1u;
    bool vertical_border =
        coord.y == 0u || coord.y == SURFACE_PROBE_RADIAL_DEPTH_DIM - 1u;
    if (horizontal_border && vertical_border)
        return 0.02221812;
    if (horizontal_border || vertical_border)
        return 0.06578385;
    return 0.09621418;
}

void accumulate_radial_moment(
    uint probe_index,
    uint texel,
    float normalized_distance,
    uint previous_iteration_count,
    uint local_texel_sample_count)
{
    float2 previous = surface_probe_load_radial_moments(
        g_probe_radial_moments, probe_index, texel);
    float sample = clamp(normalized_distance, 0.0, 2.0);
    float2 sample_moments = float2(sample, sample * sample);
    if (previous.y <= 0.0)
    {
        previous = sample_moments;
    }
    else
    {
        float previous_sample_population =
            float(previous_iteration_count)
            * float(max(g_samples_per_probe, 1u))
            * radial_texel_cosine_probability(texel);
        uint expected_previous = previous_iteration_count == 0u
            ? local_texel_sample_count - 1u
            : max(uint(previous_sample_population), 1u);
        float alpha = 1.0 / float(
            max(
                min(expected_previous + local_texel_sample_count, 4096u),
                1u));
        previous = lerp(previous, sample_moments, alpha);
    }
    surface_probe_store_radial_moments(
        g_probe_radial_moments,
        probe_index,
        texel,
        clamp(previous, 0.0, 4.0));
}

void trace_prt_transport(
    inout SurfaceProbePrtPath path,
    inout float3 transport[SURFACE_PROBE_PRT_COEFFICIENT_COUNT])
{
    uint bounce = 0u;
    while (bounce < g_max_bounces)
    {
        float hit_distance = 0.0;
        HitInfo hit = g_scene.intersect(path.ray, hit_distance);
        if (!hit.valid)
        {
            if (bounce == 0u)
                path.first_hit_distance = path.radial_max_distance;
            for (uint coefficient = 0u;
                 coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
                 coefficient++)
            {
                transport[coefficient] += path.throughput
                    * surface_probe_sh_l2_basis(
                        coefficient, path.ray.dir);
            }
            return;
        }
        if (bounce == 0u)
        {
            path.first_hit_distance = min(
                hit_distance, path.radial_max_distance);
        }

        bool same_primitive =
            hit.instance_id == path.source_instance_id
            && hit.primitive_index == path.source_primitive_index;
        float near_hit_threshold = max(
            4.0 * path.source_origin_offset, 1e-6);
        bool adjacent_near_hit =
            hit.instance_id == path.source_instance_id
            && hit_distance <= near_hit_threshold;
        if (same_primitive || adjacent_near_hit)
            path.self_hit_suspect_count += 1u;

        ShadingData shading = g_scene.get_shading_data(
            hit, -path.ray.dir);
        // These sources are not representable as distant-environment SH.
        // Keeping them in an RGB residual also preserves the existing rule:
        // direct Sun at the probe surface is realtime, later-bounce Sun is
        // part of the cached indirect response.
        path.static_source += path.throughput * shading.emissive;
        path.static_source += path.throughput
            * g_scene.sample_directional_light(
                shading, path.rng.next_2d()).xyz;

        float4 sample = cosine_sample_hemisphere(path.rng.next_2d());
        if (sample.w <= 1e-6)
            return;
        float3 f = shading.diffuse * INV_PI;
        path.throughput *= f * abs(sample.z) / sample.w;
        float3 next_origin = shading.compute_new_ray_origin();
        float3 outgoing_direction = Frame(
            normalize(shading.face_normal)).to_global(sample.xyz);
        path.ray = Ray(next_origin, outgoing_direction);
        path.source_instance_id = hit.instance_id;
        path.source_primitive_index = hit.primitive_index;
        path.source_origin_offset = length(
            next_origin - shading.position);
        bounce++;
    }
}

void estimate_surface_prt(
    HitInfo surface_hit,
    float3 view_direction,
    uint local_sample_count,
    uint probe_index,
    float kernel_radius,
    inout RNG rng,
    out float3 transport[SURFACE_PROBE_PRT_COEFFICIENT_COUNT],
    out float3 static_source,
    out uint self_hit_suspect_count)
{
    ShadingData surface = g_scene.get_shading_data(
        surface_hit, view_direction);
    for (uint coefficient = 0u;
         coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
         coefficient++)
    {
        transport[coefficient] = float3(0.0);
    }
    static_source = float3(0.0);
    self_hit_suspect_count = 0u;
    uint previous_iteration_count = g_reset != 0
        ? 0u
        : load_cached_sample_count(probe_index);
    uint local_texel_counts[SURFACE_PROBE_RADIAL_TEXEL_COUNT] =
    {
        0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u
    };
    if (g_reset != 0)
        clear_radial_moments(probe_index);

    for (uint sample_index = 0u;
         sample_index < local_sample_count;
         sample_index++)
    {
        float4 direction_sample = cosine_sample_hemisphere(rng.next_2d());
        if (direction_sample.w <= 1e-6)
            continue;

        float3 local_transport[SURFACE_PROBE_PRT_COEFFICIENT_COUNT];
        for (uint coefficient = 0u;
             coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
             coefficient++)
        {
            local_transport[coefficient] = float3(0.0);
        }
        SurfaceProbePrtPath path;
        float3 initial_origin = surface.compute_new_ray_origin();
        float3 initial_direction = Frame(
            normalize(surface.face_normal)).to_global(direction_sample.xyz);
        path.ray = Ray(initial_origin, initial_direction);
        path.throughput = float3(1.0);
        path.static_source = float3(0.0);
        path.rng = rng;
        path.source_instance_id = surface_hit.instance_id;
        path.source_primitive_index = surface_hit.primitive_index;
        path.source_origin_offset = length(
            initial_origin - surface.position);
        path.self_hit_suspect_count = 0u;
        path.first_hit_distance = 2.0 * kernel_radius;
        path.radial_max_distance = 2.0 * kernel_radius;
        trace_prt_transport(path, local_transport);
        rng = path.rng;
        self_hit_suspect_count += path.self_hit_suspect_count;

        float initial_weight =
            abs(direction_sample.z) / direction_sample.w;
        for (uint coefficient = 0u;
             coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
             coefficient++)
        {
            transport[coefficient] +=
                initial_weight * local_transport[coefficient];
        }
        static_source += initial_weight * path.static_source;

        uint radial_texel = surface_probe_radial_texel_index(
            surface_probe_radial_texel_coord(
                surface_probe_hemi_oct_encode(direction_sample.xyz)));
        local_texel_counts[radial_texel] += 1u;
        accumulate_radial_moment(
            probe_index,
            radial_texel,
            path.first_hit_distance / max(kernel_radius, 1e-6),
            previous_iteration_count,
            local_texel_counts[radial_texel]);
    }

    float inverse_sample_count = 1.0 / float(local_sample_count);
    for (uint coefficient = 0u;
         coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
         coefficient++)
    {
        transport[coefficient] *= inverse_sample_count;
    }
    static_source *= inverse_sample_count;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint probe_index = dispatch_thread_id.x;
    if (probe_index >= g_probe_count)
        return;

    SurfaceProbe probe = surface_probe_load(
        g_surface_probes, probe_index);
    uint instance_index = surface_probe_instance_index(probe);
    SurfaceProbeInstanceInfo instance_info = surface_probe_get_instance(
        g_surface_probe_instances, instance_index);

    HitInfo surface_hit;
    surface_hit.valid = true;
    surface_hit.bary = float2(
        asfloat(probe.meta.y), asfloat(probe.meta.z));
    surface_hit.instance_id = instance_index;
    surface_hit.primitive_index = probe.meta.x;

    RNG rng = RNG(
        uint2(probe_index, instance_index),
        g_iteration * max(g_samples_per_probe, 1u));
    float3 view_direction = normalize(probe.normal_side.xyz);
    float3 transport[SURFACE_PROBE_PRT_COEFFICIENT_COUNT];
    float3 static_source;
    uint self_hit_suspect_count;
    estimate_surface_prt(
        surface_hit,
        view_direction,
        max(g_samples_per_probe, 1u),
        probe_index,
        max(instance_info.params.x, 1e-6),
        rng,
        transport,
        static_source,
        self_hit_suspect_count);
    store_accumulated_prt(probe_index, transport, static_source);
    store_accumulated_self_hit_count(
        probe_index, self_hit_suspect_count);
}
