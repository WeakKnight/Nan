struct MeshColorsFaceInfo
{
    uint address;
    uint resolution;
};

struct MeshColorsInstanceInfo
{
    uint face_offset;
    uint face_count;
    uint texel_offset;
    uint texel_count;
};

struct MeshColorsSideInfo
{
    uint back_texel_offset;
    uint flags;
};

struct MeshColorsEdgeAdjacency
{
    uint adjacent_face;
    uint adjacent_edge;
    uint flip;
};

static const uint MESH_COLORS_PAYLOAD_STRIDE = 16;
static const uint MESH_COLORS_INVALID_OFFSET = 0xffffffff;
static const uint MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED = 1u << 0;
static const uint MESH_COLORS_EDGE_COUNT = 3;
static const uint MESH_COLORS_ADJACENCY_FACE_SHIFT = 3;
static const uint MESH_COLORS_ADJACENCY_EDGE_MASK = 0x3;
static const uint MESH_COLORS_ADJACENCY_FLIP_BIT = 1u << 2;
static const float MESH_COLORS_RGB9E5_MAX = 65408.0;

uint mesh_colors_patch_texel_count(uint resolution)
{
    return (resolution + 1) * (resolution + 2) / 2;
}

uint mesh_colors_address(uint i, uint j, MeshColorsInstanceInfo instance_info, MeshColorsFaceInfo face_info)
{
    uint row_offset = i * (face_info.resolution + 1) - i * (i - 1) / 2;
    return instance_info.texel_offset + face_info.address + row_offset + j;
}

uint2 mesh_colors_address_to_ij(uint address, uint resolution)
{
    uint left = 0;
    uint right = resolution;
    while (left < right)
    {
        uint middle = (left + right + 1) / 2;
        uint middle_address =
            middle * (resolution + 1) - middle * (middle - 1) / 2;
        if (middle_address <= address)
            left = middle;
        else
            right = middle - 1;
    }
    uint i = left;
    uint cumulative = i * (resolution + 1) - i * (i - 1) / 2;
    return uint2(i, address - cumulative);
}

MeshColorsFaceInfo mesh_colors_get_face(
    StructuredBuffer<uint2> face_infos,
    uint face_address)
{
    uint2 packed = face_infos[face_address];
    MeshColorsFaceInfo result;
    result.address = packed.x;
    result.resolution = packed.y;
    return result;
}

MeshColorsInstanceInfo mesh_colors_get_instance(
    StructuredBuffer<uint4> instance_infos,
    uint instance_index)
{
    uint4 packed = instance_infos[instance_index];
    MeshColorsInstanceInfo result;
    result.face_offset = packed.x;
    result.face_count = packed.y;
    result.texel_offset = packed.z;
    result.texel_count = packed.w;
    return result;
}

MeshColorsSideInfo mesh_colors_get_side(
    StructuredBuffer<uint2> side_infos,
    uint instance_index)
{
    uint2 packed = side_infos[instance_index];
    MeshColorsSideInfo result;
    result.back_texel_offset = packed.x;
    result.flags = packed.y;
    return result;
}

MeshColorsEdgeAdjacency mesh_colors_get_adjacency(
    StructuredBuffer<uint4> adjacency_infos,
    MeshColorsInstanceInfo instance_info,
    uint primitive_index,
    uint edge)
{
    MeshColorsEdgeAdjacency result;
    result.adjacent_face = MESH_COLORS_INVALID_OFFSET;
    result.adjacent_edge = 0;
    result.flip = 0;
    if (edge >= MESH_COLORS_EDGE_COUNT)
        return result;

    uint4 face_adjacency =
        adjacency_infos[instance_info.face_offset + primitive_index];
    uint packed = face_adjacency[edge];
    if (packed == MESH_COLORS_INVALID_OFFSET)
        return result;

    result.adjacent_face = instance_info.face_offset
        + (packed >> MESH_COLORS_ADJACENCY_FACE_SHIFT);
    result.adjacent_edge = packed & MESH_COLORS_ADJACENCY_EDGE_MASK;
    result.flip =
        (packed & MESH_COLORS_ADJACENCY_FLIP_BIT) != 0 ? 1u : 0u;
    return result;
}

bool mesh_colors_adjacency_is_valid(MeshColorsEdgeAdjacency adjacency)
{
    return adjacency.adjacent_face != MESH_COLORS_INVALID_OFFSET;
}

float mesh_colors_remap_edge_parameter(
    float t,
    MeshColorsEdgeAdjacency adjacency)
{
    return adjacency.flip != 0 ? 1.0 - t : t;
}

float3 mesh_colors_edge_barycentrics(uint edge, float t)
{
    if (edge == 0)
        return float3(1.0 - t, t, 0.0);
    if (edge == 1)
        return float3(0.0, 1.0 - t, t);
    return float3(t, 0.0, 1.0 - t);
}

float2 mesh_colors_edge_ij(uint edge, float t, uint resolution)
{
    return mesh_colors_edge_barycentrics(edge, t).xy * float(resolution);
}

bool mesh_colors_map_one_edge_tap(
    int2 candidate_ij,
    uint source_resolution,
    MeshColorsEdgeAdjacency adjacency,
    uint crossed_edge,
    out float3 neighbor_barycentrics)
{
    neighbor_barycentrics = float3(0.0);
    if (!mesh_colors_adjacency_is_valid(adjacency)
        || crossed_edge >= MESH_COLORS_EDGE_COUNT)
    {
        return false;
    }

    float inverse_resolution = 1.0 / float(source_resolution);
    float3 source_barycentrics = float3(
        float(candidate_ij.x) * inverse_resolution,
        float(candidate_ij.y) * inverse_resolution,
        1.0 - float(candidate_ij.x + candidate_ij.y) * inverse_resolution);
    uint source_edge_start = crossed_edge;
    uint source_edge_end = (crossed_edge + 1) % 3;
    uint source_opposite = (crossed_edge + 2) % 3;
    uint neighbor_edge_start = adjacency.adjacent_edge;
    uint neighbor_edge_end = (adjacency.adjacent_edge + 1) % 3;
    uint neighbor_opposite = (adjacency.adjacent_edge + 2) % 3;
    float outside_weight = source_barycentrics[source_opposite];
    uint mapped_source_start = adjacency.flip != 0
        ? neighbor_edge_end
        : neighbor_edge_start;
    uint mapped_source_end = adjacency.flip != 0
        ? neighbor_edge_start
        : neighbor_edge_end;

    neighbor_barycentrics[mapped_source_start] =
        source_barycentrics[source_edge_start] + outside_weight;
    neighbor_barycentrics[mapped_source_end] =
        source_barycentrics[source_edge_end] + outside_weight;
    neighbor_barycentrics[neighbor_opposite] = -outside_weight;
    if (any(neighbor_barycentrics < -1e-5)
        || any(neighbor_barycentrics > 1.0 + 1e-5))
    {
        return false;
    }

    neighbor_barycentrics = max(neighbor_barycentrics, 0.0);
    neighbor_barycentrics /= max(
        dot(neighbor_barycentrics, float3(1.0)), 1e-6);
    return true;
}

float mesh_colors_normal_filter_weight(
    float3 center_normal,
    float3 sample_normal,
    float normal_sigma_radians)
{
    float cosine = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
    if (normal_sigma_radians <= 1e-6)
        return cosine >= 1.0 - 1e-6 ? 1.0 : 0.0;
    float angle = acos(cosine);
    float normalized_angle = angle / normal_sigma_radians;
    return exp(-0.5 * normalized_angle * normalized_angle);
}

uint mesh_colors_pack_rgb9e5(float3 value)
{
    value.x = isnan(value.x) ? 0.0 : value.x;
    value.y = isnan(value.y) ? 0.0 : value.y;
    value.z = isnan(value.z) ? 0.0 : value.z;
    value = min(max(value, 0.0), MESH_COLORS_RGB9E5_MAX);
    float maximum = max(value.x, max(value.y, value.z));
    if (maximum < exp2(-25.0))
        return 0;

    int exponent = max(-15, int(floor(log2(maximum))) + 1);
    uint shared_exponent = uint(exponent + 15);
    float scale = exp2(float(shared_exponent) - 24.0);
    uint3 mantissa = uint3(floor(value / scale + 0.5));
    if (max(mantissa.x, max(mantissa.y, mantissa.z)) == 512
        && shared_exponent < 31)
    {
        shared_exponent += 1;
        scale *= 2.0;
        mantissa = uint3(floor(value / scale + 0.5));
    }
    mantissa = min(mantissa, 511);
    return mantissa.x
        | (mantissa.y << 9)
        | (mantissa.z << 18)
        | (shared_exponent << 27);
}

float3 mesh_colors_unpack_rgb9e5(uint packed)
{
    float scale = exp2(float(packed >> 27) - 24.0);
    return float3(
        packed & 0x1ff,
        (packed >> 9) & 0x1ff,
        (packed >> 18) & 0x1ff) * scale;
}

bool mesh_colors_is_double_sided(MeshColorsSideInfo side_info)
{
    return (side_info.flags & MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED) != 0;
}

uint mesh_colors_side_texel(
    uint front_texel,
    MeshColorsInstanceInfo instance_info,
    MeshColorsSideInfo side_info,
    bool back_side)
{
    return back_side
        ? side_info.back_texel_offset
            + (front_texel - instance_info.texel_offset)
        : front_texel;
}

bool mesh_colors_find_face(
    StructuredBuffer<uint2> face_infos,
    MeshColorsInstanceInfo instance_info,
    uint instance_texel_index,
    out MeshColorsFaceInfo face_info,
    out uint primitive_index)
{
    face_info.address = 0;
    face_info.resolution = 1;
    primitive_index = 0;
    int left = int(instance_info.face_offset);
    int right = int(instance_info.face_offset + instance_info.face_count) - 1;
    while (left <= right)
    {
        int middle = (left + right) / 2;
        face_info = mesh_colors_get_face(face_infos, uint(middle));
        int local = int(instance_texel_index) - int(face_info.address);
        int count = int(mesh_colors_patch_texel_count(face_info.resolution));
        if (local < 0)
        {
            right = middle - 1;
        }
        else if (local >= count)
        {
            left = middle + 1;
        }
        else
        {
            primitive_index = uint(middle) - instance_info.face_offset;
            return true;
        }
    }
    return false;
}

float3 mesh_colors_load_irradiance(
    ByteAddressBuffer buffer,
    uint texel_index)
{
    uint offset = texel_index * MESH_COLORS_PAYLOAD_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4),
        buffer.Load(offset + 8)));
}

uint mesh_colors_load_sample_count(
    ByteAddressBuffer buffer,
    uint texel_index)
{
    return buffer.Load(texel_index * MESH_COLORS_PAYLOAD_STRIDE + 12);
}

uint mesh_colors_nearest_texel(
    float3 barycentrics,
    MeshColorsInstanceInfo instance_info,
    MeshColorsFaceInfo face_info)
{
    float3 scaled = saturate(barycentrics) * float(face_info.resolution);
    uint2 ij = uint2(scaled.xy);
    ij.x = min(ij.x, face_info.resolution);
    ij.y = min(ij.y, face_info.resolution - ij.x);
    float3 weight = frac(scaled);
    bool upper = abs(weight.x + weight.y + weight.z - 2.0) <= 1e-3;
    if (upper)
        weight = 1.0 - weight;

    uint i = ij.x;
    uint j = ij.y;
    if (dot(weight, float3(1.0)) <= 1e-6
        || i >= face_info.resolution
        || j >= face_info.resolution - i)
    {
        return mesh_colors_address(i, j, instance_info, face_info);
    }

    float largest_weight = max(weight.x, max(weight.y, weight.z));
    if (upper)
    {
        if (abs(largest_weight - weight.x) < 1e-3)
            return mesh_colors_address(i, j + 1, instance_info, face_info);
        if (abs(largest_weight - weight.y) < 1e-3)
            return mesh_colors_address(i + 1, j, instance_info, face_info);
        return mesh_colors_address(i + 1, j + 1, instance_info, face_info);
    }

    if (abs(largest_weight - weight.x) < 1e-3)
        return mesh_colors_address(i + 1, j, instance_info, face_info);
    if (abs(largest_weight - weight.y) < 1e-3)
        return mesh_colors_address(i, j + 1, instance_info, face_info);
    return mesh_colors_address(i, j, instance_info, face_info);
}

float3 mesh_colors_sample_irradiance(
    ByteAddressBuffer buffer,
    float3 barycentrics,
    MeshColorsInstanceInfo instance_info,
    MeshColorsFaceInfo face_info,
    MeshColorsSideInfo side_info,
    bool back_side)
{
    float3 scaled = saturate(barycentrics) * float(face_info.resolution);
    uint2 ij = uint2(scaled.xy);
    ij.x = min(ij.x, face_info.resolution);
    ij.y = min(ij.y, face_info.resolution - ij.x);

    float3 weight = frac(scaled);
    bool upper = abs(weight.x + weight.y + weight.z - 2.0) <= 1e-3;
    if (upper)
        weight = 1.0 - weight;

    uint i = ij.x;
    uint j = ij.y;
    if (dot(weight, float3(1.0)) <= 1e-6
        || i >= face_info.resolution
        || j >= face_info.resolution - i)
    {
        return mesh_colors_load_irradiance(
            buffer,
            mesh_colors_side_texel(
                mesh_colors_address(i, j, instance_info, face_info),
                instance_info,
                side_info,
                back_side));
    }

    uint texel_a;
    uint texel_b;
    uint texel_c;
    if (upper)
    {
        texel_a = mesh_colors_address(i, j + 1, instance_info, face_info);
        texel_b = mesh_colors_address(i + 1, j, instance_info, face_info);
        texel_c = mesh_colors_address(i + 1, j + 1, instance_info, face_info);
    }
    else
    {
        texel_a = mesh_colors_address(i + 1, j, instance_info, face_info);
        texel_b = mesh_colors_address(i, j + 1, instance_info, face_info);
        texel_c = mesh_colors_address(i, j, instance_info, face_info);
    }
    return (
        weight.x * mesh_colors_load_irradiance(
            buffer,
            mesh_colors_side_texel(
                texel_a, instance_info, side_info, back_side))
        + weight.y * mesh_colors_load_irradiance(
            buffer,
            mesh_colors_side_texel(
                texel_b, instance_info, side_info, back_side))
        + weight.z * mesh_colors_load_irradiance(
            buffer,
            mesh_colors_side_texel(
                texel_c, instance_info, side_info, back_side)));
}
