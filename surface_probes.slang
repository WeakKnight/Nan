struct SurfaceProbe
{
    float4 position_radius;
    float4 normal_side;
    uint4 meta;
};

struct SurfaceProbeInstanceInfo
{
    float4 root_center_extent;
    uint4 offsets;
    float4 params;
};

static const uint SURFACE_PROBE_PAYLOAD_STRIDE = 16;
static const uint SURFACE_PROBE_SELF_HIT_STRIDE = 4;
static const uint SURFACE_PROBE_FLAG_BACK_SIDE = 1u << 0;
static const uint SURFACE_PROBE_MAX_GATHER = 32;
static const uint SURFACE_PROBE_MAX_CANDIDATES = 96;

SurfaceProbeInstanceInfo surface_probe_get_instance(
    StructuredBuffer<SurfaceProbeInstanceInfo> instances,
    uint instance_index)
{
    return instances[instance_index];
}

SurfaceProbe surface_probe_load(
    StructuredBuffer<SurfaceProbe> probes,
    uint probe_index)
{
    return probes[probe_index];
}

float3 surface_probe_barycentrics(SurfaceProbe probe)
{
    return float3(
        1.0 - asfloat(probe.meta.y) - asfloat(probe.meta.z),
        asfloat(probe.meta.y),
        asfloat(probe.meta.z));
}

bool surface_probe_is_back_side(SurfaceProbe probe)
{
    return (probe.meta.w & SURFACE_PROBE_FLAG_BACK_SIDE) != 0;
}

float surface_probe_support_f(SurfaceProbe probe)
{
    return saturate(probe.normal_side.w);
}

float3 surface_probe_load_irradiance(
    ByteAddressBuffer buffer,
    uint probe_index)
{
    uint offset = probe_index * SURFACE_PROBE_PAYLOAD_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4),
        buffer.Load(offset + 8)));
}

float3 surface_probe_load_irradiance(
    RWByteAddressBuffer buffer,
    uint probe_index)
{
    uint offset = probe_index * SURFACE_PROBE_PAYLOAD_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4),
        buffer.Load(offset + 8)));
}

uint surface_probe_load_sample_count(
    ByteAddressBuffer buffer,
    uint probe_index)
{
    return buffer.Load(probe_index * SURFACE_PROBE_PAYLOAD_STRIDE + 12);
}

uint surface_probe_load_sample_count(
    RWByteAddressBuffer buffer,
    uint probe_index)
{
    return buffer.Load(probe_index * SURFACE_PROBE_PAYLOAD_STRIDE + 12);
}

uint surface_probe_load_self_hit_count(
    ByteAddressBuffer buffer,
    uint probe_index)
{
    return buffer.Load(probe_index * SURFACE_PROBE_SELF_HIT_STRIDE);
}

uint surface_probe_load_self_hit_count(
    RWByteAddressBuffer buffer,
    uint probe_index)
{
    return buffer.Load(probe_index * SURFACE_PROBE_SELF_HIT_STRIDE);
}

bool surface_probe_sphere_intersects_cube(
    float3 position,
    float radius,
    float3 center,
    float extent)
{
    float3 distance = max(abs(position - center) - extent, 0.0);
    return dot(distance, distance) <= radius * radius;
}

void surface_probe_gather(
    StructuredBuffer<uint4> nodes,
    StructuredBuffer<SurfaceProbe> probes,
    SurfaceProbeInstanceInfo instance_info,
    float3 query_position,
    float3 query_normal,
    float radius_scale,
    out uint gathered_count,
    out uint gathered_indices[SURFACE_PROBE_MAX_CANDIDATES],
    out float gathered_weights[SURFACE_PROBE_MAX_CANDIDATES])
{
    gathered_count = 0;
    float kernel_radius = max(
        instance_info.params.x * max(radius_scale, 1.0), 1e-6);
    float normal_cosine_threshold = instance_info.params.y;
    float plane_sigma = max(instance_info.params.z, 1e-6);
    float kernel_radius_squared = kernel_radius * kernel_radius;

    uint stack_nodes[64];
    float3 stack_centers[64];
    float stack_extents[64];
    uint stack_count = 1;
    stack_nodes[0] = instance_info.offsets.x;
    stack_centers[0] = instance_info.root_center_extent.xyz;
    stack_extents[0] = instance_info.root_center_extent.w;

    while (stack_count > 0)
    {
        stack_count -= 1;
        uint node_index = stack_nodes[stack_count];
        float3 center = stack_centers[stack_count];
        float extent = stack_extents[stack_count];
        if (!surface_probe_sphere_intersects_cube(
            query_position, kernel_radius, center, extent))
        {
            continue;
        }

        uint4 node = nodes[node_index];
        uint child_base = node.x;
        uint child_mask = node.y;
        uint probe_start = node.z;
        uint probe_count = node.w;
        if (child_mask == 0)
        {
            for (uint probe_offset = 0; probe_offset < probe_count; probe_offset++)
            {
                uint probe_index = probe_start + probe_offset;
                SurfaceProbe probe = surface_probe_load(probes, probe_index);
                float3 delta = probe.position_radius.xyz - query_position;
                float distance_squared = dot(delta, delta);
                if (distance_squared > kernel_radius_squared)
                    continue;

                float3 probe_normal = normalize(probe.normal_side.xyz);
                float normal_cosine = dot(query_normal, probe_normal);
                if (normal_cosine < normal_cosine_threshold)
                    continue;

                float plane_distance = abs(dot(delta, query_normal));
                float plane_weight = exp(
                    -0.5 * (plane_distance / plane_sigma)
                        * (plane_distance / plane_sigma));
                float normalized_distance_squared =
                    distance_squared / kernel_radius_squared;
                float spatial = max(
                    1.0 - normalized_distance_squared, 0.0);
                float weight = spatial * spatial * plane_weight
                    * max(normal_cosine, 0.0);
                if (weight <= 1e-6)
                    continue;

                if (gathered_count < SURFACE_PROBE_MAX_CANDIDATES)
                {
                    gathered_indices[gathered_count] = probe_index;
                    gathered_weights[gathered_count] = weight;
                    gathered_count += 1;
                }
                else
                {
                    uint weakest = 0;
                    for (uint i = 1; i < SURFACE_PROBE_MAX_CANDIDATES; i++)
                    {
                        if (gathered_weights[i] < gathered_weights[weakest])
                            weakest = i;
                    }
                    if (weight > gathered_weights[weakest])
                    {
                        gathered_indices[weakest] = probe_index;
                        gathered_weights[weakest] = weight;
                    }
                }
            }
            continue;
        }

        float child_extent = extent * 0.5;
        uint compact_index = 0;
        for (uint octant = 0; octant < 8; octant++)
        {
            if ((child_mask & (1u << octant)) == 0)
                continue;
            float3 offset = float3(
                (octant & 1) != 0 ? 1.0 : -1.0,
                (octant & 2) != 0 ? 1.0 : -1.0,
                (octant & 4) != 0 ? 1.0 : -1.0);
            float3 child_center = center + offset * child_extent;
            if (surface_probe_sphere_intersects_cube(
                query_position, kernel_radius, child_center, child_extent)
                && stack_count < 64)
            {
                stack_nodes[stack_count] = child_base + compact_index;
                stack_centers[stack_count] = child_center;
                stack_extents[stack_count] = child_extent;
                stack_count += 1;
            }
            compact_index += 1;
        }
    }
}

float3 surface_probe_reconstruct_irradiance(
    ByteAddressBuffer irradiance,
    StructuredBuffer<SurfaceProbe> probes,
    uint gathered_count,
    uint gathered_indices[SURFACE_PROBE_MAX_GATHER],
    float gathered_weights[SURFACE_PROBE_MAX_GATHER],
    out float support_f,
    out float reconstruction_weight_sum,
    out float reconstruction_ess)
{
    float3 accumulated = float3(0.0);
    float weight_sum = 0.0;
    float weight_square_sum = 0.0;
    float support_sum = 0.0;
    float support_weight_sum = 0.0;
    float best_weight = -1.0;
    float3 best_irradiance = float3(0.0);
    for (uint i = 0; i < gathered_count; i++)
    {
        SurfaceProbe probe = surface_probe_load(
            probes, gathered_indices[i]);
        support_sum += gathered_weights[i] * surface_probe_support_f(probe);
        support_weight_sum += gathered_weights[i];
        float3 probe_irradiance = surface_probe_load_irradiance(
            irradiance, gathered_indices[i]);
        uint sample_count = surface_probe_load_sample_count(
            irradiance, gathered_indices[i]);
        float confidence = sample_count > 0 ? 1.0 : 0.0;
        float weight = gathered_weights[i] * confidence;
        accumulated += weight * probe_irradiance;
        weight_sum += weight;
        weight_square_sum += weight * weight;
        if (weight > best_weight)
        {
            best_weight = weight;
            best_irradiance = probe_irradiance;
        }
    }

    support_f = support_weight_sum > 1e-6
        ? saturate(support_sum / support_weight_sum)
        : 0.0;
    reconstruction_weight_sum = weight_sum;
    reconstruction_ess = weight_square_sum > 1e-12
        ? weight_sum * weight_sum / weight_square_sum
        : 0.0;

    if (weight_sum > 1e-6)
        return accumulated / weight_sum;
    if (best_weight >= 0.0)
        return best_irradiance;
    return float3(0.0);
}
