import common;
import scene;
import surface_probes;
import surface_probe_field;
import surface_probe_irradiance_field;

RWByteAddressBuffer g_probe_field_values;
RWByteAddressBuffer g_probe_sample_counts;
RWByteAddressBuffer g_probe_self_hit_counters;
RWByteAddressBuffer g_probe_radial_moments;
StructuredBuffer<SurfaceProbe> g_surface_probes;
StructuredBuffer<SurfaceProbeInstanceInfo> g_surface_probe_instances;
StructuredBuffer<uint4> g_triangle_vertex_probes;
uniform uint g_probe_count;
uniform uint g_iteration;
uniform uint g_samples_per_probe;
uniform uint g_max_bounces;
uniform uint g_reset;

static const uint MAX_PROBE_SAMPLE_COUNT = 65535;

struct SurfaceProbePath
{
    Ray ray;
    float3 throughput;
    float3 radiance;
    RNG rng;
    uint source_instance_id;
    uint source_primitive_index;
    float source_origin_offset;
    uint self_hit_suspect_count;
    float first_hit_distance;
    float radial_max_distance;
};

float3 load_cached_irradiance(uint probe_index)
{
    return surface_probe_load_irradiance(g_probe_field_values, probe_index);
}

uint load_cached_sample_count(uint probe_index)
{
    return surface_probe_load_sample_count(
        g_probe_sample_counts, probe_index);
}

void store_accumulated_irradiance(uint probe_index, float3 irradiance_sample)
{
    uint previous_count =
        g_reset != 0 ? 0 : load_cached_sample_count(probe_index);
    float3 previous_irradiance = previous_count == 0
        ? float3(0.0)
        : load_cached_irradiance(probe_index);
    uint next_count = min(previous_count + 1, MAX_PROBE_SAMPLE_COUNT);
    float weight = 1.0 / float(previous_count + 1);
    float3 accumulated =
        lerp(previous_irradiance, irradiance_sample, weight);

    uint destination = probe_index * SURFACE_PROBE_IRRADIANCE_VALUE_STRIDE;
    g_probe_field_values.Store(destination + 0, asuint(accumulated.x));
    g_probe_field_values.Store(destination + 4, asuint(accumulated.y));
    g_probe_field_values.Store(destination + 8, asuint(accumulated.z));
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
        // RG16F intentionally carries no count.  Hemi-oct texels have unequal
        // cosine-sampling probability, so use the calibrated population of
        // this texel instead of assuming a uniform 1/16 distribution.
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

void trace_radiance(inout SurfaceProbePath path)
{
    uint bounce = 0;
    while (bounce < g_max_bounces)
    {
        float hit_distance = 0.0;
        HitInfo hit = g_scene.intersect(path.ray, hit_distance);
        if (!hit.valid)
        {
            if (bounce == 0u)
                path.first_hit_distance = path.radial_max_distance;
            path.radiance += path.throughput * g_scene.sample_sky_lut(path.ray.dir);
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

        ShadingData shading = g_scene.get_shading_data(hit, -path.ray.dir);
        path.radiance += path.throughput * shading.emissive;
        path.radiance += path.throughput
            * g_scene.sample_directional_light(shading, path.rng.next_2d()).xyz;

        float4 sample = cosine_sample_hemisphere(path.rng.next_2d());
        if (sample.w <= 1e-6)
            return;
        float3 f = shading.diffuse * INV_PI;
        path.throughput *= f * abs(sample.z) / sample.w;
        float3 next_origin = shading.compute_new_ray_origin();
        float3 outgoing_direction = Frame(
            normalize(shading.face_normal)).to_global(sample.xyz);
        path.ray = Ray(
            next_origin,
            outgoing_direction);
        path.source_instance_id = hit.instance_id;
        path.source_primitive_index = hit.primitive_index;
        path.source_origin_offset = length(
            next_origin - shading.position);
        bounce++;
    }
}

float3 estimate_surface_irradiance(
    HitInfo surface_hit,
    float3 view_direction,
    uint local_sample_count,
    uint probe_index,
    float kernel_radius,
    inout RNG rng,
    out uint self_hit_suspect_count)
{
    ShadingData surface =
        g_scene.get_shading_data(surface_hit, view_direction);
    float3 irradiance_sample = float3(0.0);
    self_hit_suspect_count = 0u;
    uint previous_iteration_count =
        g_reset != 0 ? 0u : load_cached_sample_count(probe_index);
    uint local_texel_counts[SURFACE_PROBE_RADIAL_TEXEL_COUNT] =
    {
        0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u
    };
    if (g_reset != 0)
        clear_radial_moments(probe_index);
    for (uint sample_index = 0; sample_index < local_sample_count; sample_index++)
    {
        // The probe cache is indirect-only. Sun light on the probe's own
        // surface is evaluated in the preview resolve so it remains fully
        // responsive to sun direction, intensity, and realtime visibility.
        // Sun NEE at later path vertices remains here because that energy has
        // bounced at least once before reaching this probe.
        float3 sample_irradiance = float3(0.0);
        float4 direction_sample = cosine_sample_hemisphere(rng.next_2d());
        if (direction_sample.w > 1e-6)
        {
            SurfaceProbePath path;
            float3 initial_origin = surface.compute_new_ray_origin();
            float3 initial_direction = Frame(
                normalize(surface.face_normal)).to_global(
                    direction_sample.xyz);
            path.ray = Ray(
                initial_origin,
                initial_direction);
            path.throughput = float3(1.0);
            path.radiance = float3(0.0);
            path.rng = rng;
            path.source_instance_id = surface_hit.instance_id;
            path.source_primitive_index = surface_hit.primitive_index;
            path.source_origin_offset = length(
                initial_origin - surface.position);
            path.self_hit_suspect_count = 0u;
            path.first_hit_distance = 2.0 * kernel_radius;
            path.radial_max_distance = 2.0 * kernel_radius;
            trace_radiance(path);
            rng = path.rng;
            self_hit_suspect_count += path.self_hit_suspect_count;
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
            sample_irradiance +=
                path.radiance * abs(direction_sample.z) / direction_sample.w;
        }
        irradiance_sample += sample_irradiance;
    }
    return irradiance_sample / float(local_sample_count);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint probe_index = dispatch_thread_id.x;
    if (probe_index >= g_probe_count)
        return;

    SurfaceProbe probe = surface_probe_load(g_surface_probes, probe_index);
    uint instance_index = surface_probe_instance_index(probe);
    SurfaceProbeInstanceInfo instance_info = surface_probe_get_instance(
        g_surface_probe_instances, instance_index);

    HitInfo surface_hit;
    surface_hit.valid = true;
    surface_hit.bary = float2(asfloat(probe.meta.y), asfloat(probe.meta.z));
    surface_hit.instance_id = instance_index;
    surface_hit.primitive_index = probe.meta.x;

    RNG rng = RNG(
        uint2(probe_index, instance_index),
        g_iteration * max(g_samples_per_probe, 1u));
    float3 view_direction = normalize(probe.normal_side.xyz);
    uint self_hit_suspect_count;
    float3 irradiance = estimate_surface_irradiance(
        surface_hit,
        view_direction,
        max(g_samples_per_probe, 1u),
        probe_index,
        max(instance_info.params.x, 1e-6),
        rng,
        self_hit_suspect_count);
    store_accumulated_irradiance(probe_index, irradiance);
    store_accumulated_self_hit_count(
        probe_index, self_hit_suspect_count);
}
