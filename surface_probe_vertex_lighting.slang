import common;
import surface_probes;

ByteAddressBuffer g_probe_irradiance;
StructuredBuffer<SurfaceProbe> g_surface_probes;
StructuredBuffer<uint4> g_surface_probe_nodes;
StructuredBuffer<SurfaceProbeInstanceInfo> g_surface_probe_instances;

// x=instance, y=mesh-local vertex, z=side, w=asuint(condition multiplier).
StructuredBuffer<uint4> g_vertex_meta;
// xyz=world-space position, w=expected incident-area projection mass.
StructuredBuffer<float4> g_vertex_position_area;
// xyz=world-space side-aware normal.
StructuredBuffer<float4> g_vertex_normal;
StructuredBuffer<uint> g_neighbor_offsets;
// x=neighbor vertex-lighting index, y=asuint(edge weight).
StructuredBuffer<uint2> g_neighbors;
StructuredBuffer<uint> g_projection_offsets;
// x=probe, y=asuint(area * barycentric basis), z=asuint(area / 3 coverage).
StructuredBuffer<uint3> g_projection_samples;

RWStructuredBuffer<float4> g_target;
StructuredBuffer<float4> g_source;
RWStructuredBuffer<float4> g_destination;
RWStructuredBuffer<uint> g_packed_rgbm;

uniform uint g_vertex_count;
uniform uint g_min_gather_count;
uniform float g_regularization_strength;
uniform float g_rgbm_range;

float3 vertex_lighting_spatial_gather(
    uint vertex_index,
    out float confidence)
{
    uint4 meta = g_vertex_meta[vertex_index];
    SurfaceProbeInstanceInfo instance_info =
        g_surface_probe_instances[meta.x];
    float3 query_position = g_vertex_position_area[vertex_index].xyz;
    float3 query_normal = normalize(g_vertex_normal[vertex_index].xyz);
    uint candidate_count;
    uint candidate_indices[SURFACE_PROBE_MAX_CANDIDATES];
    float candidate_weights[SURFACE_PROBE_MAX_CANDIDATES];
    uint gathered_count;
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

    uint minimum_count = clamp(g_min_gather_count, 1u, 32u);
    if (gathered_count < minimum_count)
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

    float support_f;
    float weight_sum;
    float ess;
    float3 irradiance = surface_probe_reconstruct_irradiance(
        g_probe_irradiance,
        g_surface_probes,
        gathered_count,
        gathered_indices,
        gathered_weights,
        support_f,
        weight_sum,
        ess);
    confidence = weight_sum > 1e-6
        ? min(
            saturate(float(gathered_count) / float(minimum_count)),
            saturate(ess / float(minimum_count)))
        : 0.0;
    return irradiance;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gather_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint vertex_index = dispatch_thread_id.x;
    if (vertex_index >= g_vertex_count)
        return;

    uint begin = g_projection_offsets[vertex_index];
    uint end = g_projection_offsets[vertex_index + 1];
    float3 weighted_irradiance = float3(0.0);
    float weighted_luma_square = 0.0;
    float total_weight = 0.0;
    float valid_weight = 0.0;
    float valid_integration_weight = 0.0;
    for (uint contribution = begin; contribution < end; contribution++)
    {
        uint3 sample = g_projection_samples[contribution];
        float weight = max(asfloat(sample.y), 0.0);
        float coverage_weight = max(asfloat(sample.z), 0.0);
        total_weight += coverage_weight;
        if (surface_probe_load_sample_count(
                g_probe_irradiance, sample.x) == 0)
        {
            continue;
        }
        float3 sample_irradiance = surface_probe_load_irradiance(
            g_probe_irradiance, sample.x);
        weighted_irradiance += weight * sample_irradiance;
        float sample_luma = dot(
            sample_irradiance, float3(0.2126, 0.7152, 0.0722));
        weighted_luma_square += weight * sample_luma * sample_luma;
        valid_weight += coverage_weight;
        valid_integration_weight += weight;
    }

    float3 projected_irradiance = valid_integration_weight > 1e-20
        ? weighted_irradiance / valid_integration_weight
        : float3(0.0);
    if (valid_integration_weight > 1e-20)
    {
        float mean_luma = dot(
            projected_irradiance, float3(0.2126, 0.7152, 0.0722));
        float rms_luma = sqrt(max(
            weighted_luma_square / valid_integration_weight, 0.0));
        float condition_multiplier = max(
            asfloat(g_vertex_meta[vertex_index].w), 1.0);
        float underresolved_t = sqrt(saturate(
            (condition_multiplier - 1.0) / 31.0));
        float standard_deviation = sqrt(max(
            rms_luma * rms_luma - mean_luma * mean_luma, 0.0));
        float energy_scale = min(
            (mean_luma + standard_deviation)
                / max(mean_luma, 1e-8),
            2.0);
        projected_irradiance *= lerp(
            1.0, energy_scale, underresolved_t);
    }
    float expected_weight = max(
        g_vertex_position_area[vertex_index].w, 0.0);
    // Measure against the vertex's full lumped incident area. Measuring only
    // surviving projection samples would misclassify uncovered fine
    // triangles as complete.
    float projection_coverage = expected_weight > 1e-20
        ? saturate(valid_weight / expected_weight)
        : (valid_weight > 1e-20 ? 1.0 : 0.0);
    float temporal_projection_confidence = total_weight > 1e-20
        ? saturate(valid_weight / total_weight)
        : 0.0;
    float3 irradiance = projected_irradiance;
    float confidence =
        projection_coverage * temporal_projection_confidence;
    if (projection_coverage < 0.999)
    {
        float spatial_confidence;
        float3 spatial_irradiance = vertex_lighting_spatial_gather(
            vertex_index, spatial_confidence);
        if (spatial_confidence > 0.0)
        {
            irradiance = valid_integration_weight > 1e-20
                ? lerp(
                    spatial_irradiance,
                    projected_irradiance,
                    projection_coverage)
                : spatial_irradiance;
            confidence = saturate(
                projection_coverage * temporal_projection_confidence
                + (1.0 - projection_coverage) * spatial_confidence);
        }
    }
    float4 value = float4(max(irradiance, 0.0), confidence);
    g_target[vertex_index] = value;
    g_destination[vertex_index] = value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smooth_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint vertex_index = dispatch_thread_id.x;
    if (vertex_index >= g_vertex_count)
        return;

    float4 target = g_target[vertex_index];
    uint begin = g_neighbor_offsets[vertex_index];
    uint end = g_neighbor_offsets[vertex_index + 1];
    float3 neighbor_sum = float3(0.0);
    float neighbor_weight_sum = 0.0;
    for (uint edge = begin; edge < end; edge++)
    {
        uint2 neighbor = g_neighbors[edge];
        float weight = asfloat(neighbor.y);
        neighbor_sum += weight * g_source[neighbor.x].xyz;
        neighbor_weight_sum += weight;
    }

    if (neighbor_weight_sum <= 1e-8)
    {
        g_destination[vertex_index] = target;
        return;
    }

    float3 neighbor_average = neighbor_sum / neighbor_weight_sum;
    float condition_multiplier = max(asfloat(g_vertex_meta[vertex_index].w), 1.0);
    float condition_t = sqrt(saturate(
        (condition_multiplier - 1.0) / 31.0));
    // Strength 1 is deliberately data preserving: even the worst sliver
    // receives only a 0.25 neighbor term instead of overwhelming a confidence
    // weight whose maximum is 1.
    float lambda = max(g_regularization_strength, 0.0)
        * lerp(0.02, 0.25, condition_t);
    float data_weight = saturate(target.w);
    float denominator = data_weight + lambda;
    float3 result = denominator > 1e-8
        ? (data_weight * target.xyz + lambda * neighbor_average)
            / denominator
        : neighbor_average;
    g_destination[vertex_index] = float4(max(result, 0.0), target.w);
}

uint vertex_lighting_pack_rgbm(float3 value, float range)
{
    float3 color = max(value, 0.0);
    float maximum = max(max(color.x, color.y), color.z);
    if (maximum <= 0.0)
        return 0;
    float multiplier = ceil(
        saturate(maximum / max(range, 1e-6)) * 255.0) / 255.0;
    float3 encoded = saturate(color / (multiplier * max(range, 1e-6)));
    uint4 bytes = uint4(
        uint3(round(encoded * 255.0)),
        uint(round(multiplier * 255.0)));
    return bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pack_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint vertex_index = dispatch_thread_id.x;
    if (vertex_index >= g_vertex_count)
        return;
    g_packed_rgbm[vertex_index] = vertex_lighting_pack_rgbm(
        g_source[vertex_index].xyz, g_rgbm_range);
}
