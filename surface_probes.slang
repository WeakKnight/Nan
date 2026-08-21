import common;

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

static const uint SURFACE_PROBE_SELF_HIT_STRIDE = 4;
static const uint SURFACE_PROBE_RADIAL_DEPTH_DIM = 4;
static const uint SURFACE_PROBE_RADIAL_TEXEL_COUNT =
    SURFACE_PROBE_RADIAL_DEPTH_DIM * SURFACE_PROBE_RADIAL_DEPTH_DIM;
static const uint SURFACE_PROBE_RADIAL_MOMENT_STRIDE =
    SURFACE_PROBE_RADIAL_TEXEL_COUNT * 4;
static const uint SURFACE_PROBE_FLAG_BACK_SIDE = 1u << 0;
static const uint SURFACE_PROBE_INSTANCE_SHIFT = 8;
static const uint SURFACE_PROBE_MAX_GATHER = 32;
static const uint SURFACE_PROBE_MAX_CANDIDATES = 96;
// Forty-five degrees remains the full-confidence angle.  Unlike the old
// binary gate, compatible support now fades continuously until this
// conservative hard reject.  This avoids triangle-shaped support changes
// when interpolated shading normals straddle the old threshold.
static const float SURFACE_PROBE_HARD_NORMAL_COSINE = 0.17364818;

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

uint surface_probe_instance_index(SurfaceProbe probe)
{
    return probe.meta.w >> SURFACE_PROBE_INSTANCE_SHIFT;
}

float surface_probe_support_f(SurfaceProbe probe)
{
    return saturate(probe.normal_side.w);
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

uint surface_probe_pack_half2(float2 value)
{
    uint x = f32tof16(value.x) & 0xffffu;
    uint y = f32tof16(value.y) & 0xffffu;
    return x | (y << 16u);
}

float2 surface_probe_unpack_half2(uint packed)
{
    return float2(
        f16tof32(packed & 0xffffu),
        f16tof32((packed >> 16u) & 0xffffu));
}

float2 surface_probe_load_radial_moments(
    ByteAddressBuffer buffer,
    uint probe_index,
    uint texel)
{
    uint offset = probe_index * SURFACE_PROBE_RADIAL_MOMENT_STRIDE
        + min(texel, SURFACE_PROBE_RADIAL_TEXEL_COUNT - 1u) * 4u;
    return surface_probe_unpack_half2(buffer.Load(offset));
}

float2 surface_probe_load_radial_moments(
    RWByteAddressBuffer buffer,
    uint probe_index,
    uint texel)
{
    uint offset = probe_index * SURFACE_PROBE_RADIAL_MOMENT_STRIDE
        + min(texel, SURFACE_PROBE_RADIAL_TEXEL_COUNT - 1u) * 4u;
    return surface_probe_unpack_half2(buffer.Load(offset));
}

void surface_probe_store_radial_moments(
    RWByteAddressBuffer buffer,
    uint probe_index,
    uint texel,
    float2 moments)
{
    uint offset = probe_index * SURFACE_PROBE_RADIAL_MOMENT_STRIDE
        + min(texel, SURFACE_PROBE_RADIAL_TEXEL_COUNT - 1u) * 4u;
    buffer.Store(offset, surface_probe_pack_half2(moments));
}

// Maps the complete +Z hemisphere to a square.  The 45-degree rotation after
// the L1 projection turns the usual octahedral diamond into [0, 1]^2, so all
// 16 texels represent useful directions rather than wasting the square's
// corners.  Directions below the probe plane clamp to the horizon because the
// cache only traces the outward hemisphere.
float2 surface_probe_hemi_oct_encode(float3 direction)
{
    float3 hemisphere_direction = float3(
        direction.xy,
        max(direction.z, 0.0));
    float direction_length_squared = dot(
        hemisphere_direction,
        hemisphere_direction);
    if (direction_length_squared <= 1e-12)
        return float2(0.5);
    hemisphere_direction *= rsqrt(direction_length_squared);
    float inverse_l1 = 1.0 / max(
        abs(hemisphere_direction.x)
            + abs(hemisphere_direction.y)
            + hemisphere_direction.z,
        1e-6);
    float2 octahedral = hemisphere_direction.xy * inverse_l1;
    float2 square = float2(
        octahedral.x + octahedral.y,
        octahedral.x - octahedral.y);
    return saturate(square * 0.5 + 0.5);
}

uint2 surface_probe_radial_texel_coord(float2 uv)
{
    return min(
        uint2(saturate(uv) * float(SURFACE_PROBE_RADIAL_DEPTH_DIM)),
        uint2(SURFACE_PROBE_RADIAL_DEPTH_DIM - 1u));
}

uint surface_probe_radial_texel_index(uint2 coord)
{
    return coord.y * SURFACE_PROBE_RADIAL_DEPTH_DIM + coord.x;
}

float2 surface_probe_sample_radial_moments(
    ByteAddressBuffer buffer,
    uint probe_index,
    float2 uv,
    out float valid_weight)
{
    // Texture-style bilinear coordinates place texel centers at (n + 0.5)/4.
    float2 texel_position =
        saturate(uv) * float(SURFACE_PROBE_RADIAL_DEPTH_DIM) - 0.5;
    int2 base_coord = int2(floor(texel_position));
    float2 blend = frac(texel_position);
    int maximum_coord = int(SURFACE_PROBE_RADIAL_DEPTH_DIM) - 1;
    uint2 coord00 = uint2(clamp(base_coord, int2(0), int2(maximum_coord)));
    uint2 coord10 = uint2(clamp(
        base_coord + int2(1, 0), int2(0), int2(maximum_coord)));
    uint2 coord01 = uint2(clamp(
        base_coord + int2(0, 1), int2(0), int2(maximum_coord)));
    uint2 coord11 = uint2(clamp(
        base_coord + int2(1, 1), int2(0), int2(maximum_coord)));
    float4 weights = float4(
        (1.0 - blend.x) * (1.0 - blend.y),
        blend.x * (1.0 - blend.y),
        (1.0 - blend.x) * blend.y,
        blend.x * blend.y);
    float2 samples[4] =
    {
        surface_probe_load_radial_moments(
            buffer, probe_index, surface_probe_radial_texel_index(coord00)),
        surface_probe_load_radial_moments(
            buffer, probe_index, surface_probe_radial_texel_index(coord10)),
        surface_probe_load_radial_moments(
            buffer, probe_index, surface_probe_radial_texel_index(coord01)),
        surface_probe_load_radial_moments(
            buffer, probe_index, surface_probe_radial_texel_index(coord11))
    };

    float2 moments = float2(0.0);
    valid_weight = 0.0;
    for (uint sample_index = 0u; sample_index < 4u; sample_index++)
    {
        if (samples[sample_index].y > 0.0)
        {
            moments += samples[sample_index] * weights[sample_index];
            valid_weight += weights[sample_index];
        }
    }
    return valid_weight > 1e-6
        ? moments / valid_weight
        : float2(0.0);
}

float surface_probe_soft_normal_weight(
    float normal_cosine,
    float full_confidence_cosine)
{
    float soft_end = max(
        full_confidence_cosine,
        SURFACE_PROBE_HARD_NORMAL_COSINE + 1e-3);
    float gate = smoothstep(
        SURFACE_PROBE_HARD_NORMAL_COSINE,
        soft_end,
        normal_cosine);
    return gate * max(normal_cosine, 0.0);
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
                if (normal_cosine < SURFACE_PROBE_HARD_NORMAL_COSINE)
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
                    * surface_probe_soft_normal_weight(
                        normal_cosine, normal_cosine_threshold);
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

void surface_probe_select_strongest(
    uint candidate_count,
    uint candidate_indices[SURFACE_PROBE_MAX_CANDIDATES],
    float candidate_weights[SURFACE_PROBE_MAX_CANDIDATES],
    out uint selected_count,
    out uint selected_indices[SURFACE_PROBE_MAX_GATHER],
    out float selected_weights[SURFACE_PROBE_MAX_GATHER])
{
    selected_count = 0;
    uint3 consumed_mask = uint3(0);

    for (uint selection = 0;
         selection < candidate_count
             && selected_count < SURFACE_PROBE_MAX_GATHER;
         selection++)
    {
        uint strongest = 0;
        float strongest_weight = -1.0;
        for (uint i = 0; i < candidate_count; i++)
        {
            uint word = i >> 5;
            uint bit = 1u << (i & 31u);
            bool consumed = word == 0
                ? (consumed_mask.x & bit) != 0
                : (word == 1
                    ? (consumed_mask.y & bit) != 0
                    : (consumed_mask.z & bit) != 0);
            if (!consumed && candidate_weights[i] > strongest_weight)
            {
                strongest = i;
                strongest_weight = candidate_weights[i];
            }
        }
        uint strongest_word = strongest >> 5;
        uint strongest_bit = 1u << (strongest & 31u);
        if (strongest_word == 0)
            consumed_mask.x |= strongest_bit;
        else if (strongest_word == 1)
            consumed_mask.y |= strongest_bit;
        else
            consumed_mask.z |= strongest_bit;
        selected_indices[selected_count] = candidate_indices[strongest];
        selected_weights[selected_count] = strongest_weight;
        selected_count += 1;
    }
}

void surface_probe_merge_fallback(
    uint fallback_count,
    uint fallback_indices[SURFACE_PROBE_MAX_GATHER],
    float fallback_weights[SURFACE_PROBE_MAX_GATHER],
    float fallback_scale,
    inout uint gathered_count,
    inout uint gathered_indices[SURFACE_PROBE_MAX_GATHER],
    inout float gathered_weights[SURFACE_PROBE_MAX_GATHER])
{
    for (uint fallback = 0; fallback < fallback_count; fallback++)
    {
        bool duplicate = false;
        for (uint primary = 0; primary < gathered_count; primary++)
        {
            if (gathered_indices[primary] == fallback_indices[fallback])
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;
        float weight = fallback_weights[fallback] * fallback_scale;
        if (weight <= 1e-8)
            continue;
        if (gathered_count < SURFACE_PROBE_MAX_GATHER)
        {
            gathered_indices[gathered_count] = fallback_indices[fallback];
            gathered_weights[gathered_count] = weight;
            gathered_count += 1;
        }
        else
        {
            uint weakest = 0u;
            for (uint i = 1u; i < gathered_count; i++)
            {
                if (gathered_weights[i] < gathered_weights[weakest])
                    weakest = i;
            }
            if (weight > gathered_weights[weakest])
            {
                gathered_indices[weakest] = fallback_indices[fallback];
                gathered_weights[weakest] = weight;
            }
        }
    }
}

float surface_probe_weight_ess(
    uint gathered_count,
    float gathered_weights[SURFACE_PROBE_MAX_GATHER])
{
    float weight_sum = 0.0;
    float weight_square_sum = 0.0;
    for (uint i = 0; i < gathered_count; i++)
    {
        float weight = max(gathered_weights[i], 0.0);
        weight_sum += weight;
        weight_square_sum += weight * weight;
    }
    return weight_square_sum > 1e-12
        ? weight_sum * weight_sum / weight_square_sum
        : 0.0;
}

float surface_probe_chebyshev_visibility(
    float2 moments,
    float normalized_distance)
{
    if (moments.y <= 0.0)
        return 1.0;
    float mean = max(moments.x, 0.0);
    float bias = 0.035;
    if (normalized_distance <= mean + bias)
        return 1.0;
    float variance = max(moments.y - mean * mean, 2.5e-3);
    float difference = normalized_distance - mean;
    float probability = variance / (variance + difference * difference);
    // Reduce the characteristic Chebyshev light-bleeding tail, but retain a
    // non-zero floor because this signal only modulates the 2R safety tier.
    return max(saturate((probability - 0.2) / 0.8), 0.05);
}

float surface_probe_radial_visibility(
    ByteAddressBuffer radial_moments,
    StructuredBuffer<SurfaceProbe> probes,
    uint probe_index,
    float3 query_position,
    float kernel_radius,
    uint sample_count,
    uint enabled)
{
    if (enabled == 0u)
        return 1.0;

    SurfaceProbe probe = surface_probe_load(probes, probe_index);
    float3 delta = query_position - probe.position_radius.xyz;
    float distance = length(delta);
    if (distance <= 1e-6)
        return 1.0;
    float3 local = Frame(normalize(probe.normal_side.xyz)).to_local(
        delta / distance);
    float normalized_distance = min(
        distance / max(kernel_radius, 1e-6), 2.0);
    float valid_weight = 0.0;
    float2 moments = surface_probe_sample_radial_moments(
        radial_moments,
        probe_index,
        surface_probe_hemi_oct_encode(local),
        valid_weight);
    if (valid_weight <= 1e-6)
        return 1.0;
    float visibility = surface_probe_chebyshev_visibility(
        moments,
        normalized_distance);
    float temporal_confidence = saturate(
        float(sample_count) / 32.0);
    // This is deliberately a soft prior, not a binary visibility decision.
    return lerp(
        1.0,
        visibility,
        0.75 * temporal_confidence);
}
