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

static const uint MESH_COLORS_PAYLOAD_STRIDE = 16;
static const uint MESH_COLORS_INVALID_OFFSET = 0xffffffff;
static const uint MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED = 1u << 0;

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
