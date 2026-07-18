import common;
import mesh_colors;

ByteAddressBuffer g_source;
RWByteAddressBuffer g_destination;
StructuredBuffer<uint2> g_mesh_colors_face_infos;
StructuredBuffer<uint4> g_mesh_colors_instance_infos;
StructuredBuffer<uint2> g_mesh_colors_side_infos;
StructuredBuffer<uint4> g_mesh_colors_adjacency_infos;
StructuredBuffer<float4> g_mesh_colors_face_normals;
uniform uint g_instance_index;
uniform float g_neighbor_weight;
uniform float g_normal_sigma_radians;

int2 mesh_colors_filter_offset(uint index)
{
    if (index == 0)
        return int2(1, 0);
    if (index == 1)
        return int2(-1, 0);
    if (index == 2)
        return int2(0, 1);
    if (index == 3)
        return int2(0, -1);
    if (index == 4)
        return int2(1, -1);
    return int2(-1, 1);
}

uint source_side_texel(
    uint front_texel,
    MeshColorsInstanceInfo instance_info,
    MeshColorsSideInfo side_info,
    bool back_side)
{
    return mesh_colors_side_texel(
        front_texel, instance_info, side_info, back_side);
}

bool map_filter_tap_to_neighbor(
    int2 candidate_ij,
    MeshColorsInstanceInfo instance_info,
    MeshColorsFaceInfo face_info,
    uint primitive_index,
    out MeshColorsFaceInfo neighbor_face_info,
    out uint neighbor_primitive_index,
    out float3 neighbor_barycentrics)
{
    neighbor_face_info.address = 0;
    neighbor_face_info.resolution = 1;
    neighbor_primitive_index = 0;
    neighbor_barycentrics = float3(0.0);
    uint crossed_edge;
    if (candidate_ij.x < 0)
        crossed_edge = 1;
    else if (candidate_ij.y < 0)
        crossed_edge = 2;
    else if (candidate_ij.x + candidate_ij.y > int(face_info.resolution))
        crossed_edge = 0;
    else
        return false;

    MeshColorsEdgeAdjacency adjacency = mesh_colors_get_adjacency(
        g_mesh_colors_adjacency_infos,
        instance_info,
        primitive_index,
        crossed_edge);
    if (!mesh_colors_adjacency_is_valid(adjacency))
        return false;
    if (!mesh_colors_map_one_edge_tap(
        candidate_ij,
        face_info.resolution,
        adjacency,
        crossed_edge,
        neighbor_barycentrics))
        return false;

    neighbor_primitive_index =
        adjacency.adjacent_face - instance_info.face_offset;
    neighbor_face_info = mesh_colors_get_face(
        g_mesh_colors_face_infos, adjacency.adjacent_face);
    return true;
}

bool load_filter_tap(
    int2 candidate_ij,
    MeshColorsInstanceInfo instance_info,
    MeshColorsSideInfo side_info,
    MeshColorsFaceInfo face_info,
    uint primitive_index,
    float3 center_normal,
    bool back_side,
    out float3 irradiance,
    out float tap_weight)
{
    int resolution = int(face_info.resolution);
    if (candidate_ij.x >= 0
        && candidate_ij.y >= 0
        && candidate_ij.x + candidate_ij.y <= resolution)
    {
        uint front_texel = mesh_colors_address(
            uint(candidate_ij.x),
            uint(candidate_ij.y),
            instance_info,
            face_info);
        irradiance = mesh_colors_load_irradiance(
            g_source,
            source_side_texel(
                front_texel, instance_info, side_info, back_side));
        tap_weight = g_neighbor_weight;
        return true;
    }

    MeshColorsFaceInfo neighbor_face_info;
    uint neighbor_primitive_index;
    float3 neighbor_barycentrics;
    if (!map_filter_tap_to_neighbor(
        candidate_ij,
        instance_info,
        face_info,
        primitive_index,
        neighbor_face_info,
        neighbor_primitive_index,
        neighbor_barycentrics))
    {
        irradiance = float3(0.0);
        tap_weight = 0.0;
        return false;
    }

    float3 neighbor_normal = g_mesh_colors_face_normals[
        instance_info.face_offset + neighbor_primitive_index].xyz;
    tap_weight = g_neighbor_weight
        * mesh_colors_normal_filter_weight(
            center_normal, neighbor_normal, g_normal_sigma_radians);
    irradiance = mesh_colors_sample_irradiance(
        g_source,
        neighbor_barycentrics,
        instance_info,
        neighbor_face_info,
        side_info,
        back_side);
    return tap_weight > 0.0;
}

void store_filtered_irradiance(
    uint texel_index,
    float3 irradiance,
    uint sample_count)
{
    uint destination = texel_index * MESH_COLORS_PAYLOAD_STRIDE;
    g_destination.Store(destination + 0, asuint(irradiance.x));
    g_destination.Store(destination + 4, asuint(irradiance.y));
    g_destination.Store(destination + 8, asuint(irradiance.z));
    g_destination.Store(destination + 12, sample_count);
}

void filter_side(
    uint front_texel,
    uint2 ij,
    MeshColorsInstanceInfo instance_info,
    MeshColorsSideInfo side_info,
    MeshColorsFaceInfo face_info,
    uint primitive_index,
    float3 center_normal,
    bool back_side)
{
    uint center_texel = source_side_texel(
        front_texel, instance_info, side_info, back_side);
    float3 accumulated = mesh_colors_load_irradiance(
        g_source, center_texel);
    float accumulated_weight = 1.0;
    int2 center_ij = int2(ij);

    for (uint tap_index = 0; tap_index < 6; tap_index++)
    {
        float3 tap_irradiance;
        float tap_weight;
        if (load_filter_tap(
            center_ij + mesh_colors_filter_offset(tap_index),
            instance_info,
            side_info,
            face_info,
            primitive_index,
            center_normal,
            back_side,
            tap_irradiance,
            tap_weight))
        {
            accumulated += tap_weight * tap_irradiance;
            accumulated_weight += tap_weight;
        }
    }

    store_filtered_irradiance(
        center_texel,
        accumulated / max(accumulated_weight, 1e-6),
        mesh_colors_load_sample_count(g_source, center_texel));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint instance_local_texel = dispatch_thread_id.x;
    MeshColorsInstanceInfo instance_info = mesh_colors_get_instance(
        g_mesh_colors_instance_infos, g_instance_index);
    if (instance_local_texel >= instance_info.texel_count)
        return;

    MeshColorsFaceInfo face_info;
    uint primitive_index;
    if (!mesh_colors_find_face(
        g_mesh_colors_face_infos,
        instance_info,
        instance_local_texel,
        face_info,
        primitive_index))
    {
        return;
    }

    MeshColorsSideInfo side_info = mesh_colors_get_side(
        g_mesh_colors_side_infos, g_instance_index);
    uint face_local_address = instance_local_texel - face_info.address;
    uint2 ij = mesh_colors_address_to_ij(
        face_local_address, face_info.resolution);
    uint front_texel = instance_info.texel_offset + instance_local_texel;
    float3 center_normal = g_mesh_colors_face_normals[
        instance_info.face_offset + primitive_index].xyz;

    filter_side(
        front_texel,
        ij,
        instance_info,
        side_info,
        face_info,
        primitive_index,
        center_normal,
        false);
    if (mesh_colors_is_double_sided(side_info))
    {
        filter_side(
            front_texel,
            ij,
            instance_info,
            side_info,
            face_info,
            primitive_index,
            center_normal,
            true);
    }
}
