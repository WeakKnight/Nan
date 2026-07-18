import common;
import scene;
import mesh_colors;

RWTexture2D<float4> g_output;
StructuredBuffer<uint> g_mesh_colors;
StructuredBuffer<uint2> g_mesh_colors_face_infos;
StructuredBuffer<uint4> g_mesh_colors_instance_infos;
StructuredBuffer<uint2> g_mesh_colors_side_infos;

float3 load_frozen_mesh_color(uint texel_index)
{
    return mesh_colors_unpack_rgb9e5(g_mesh_colors[texel_index]);
}

float3 sample_frozen_mesh_colors(
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
        uint texel = mesh_colors_side_texel(
            mesh_colors_address(i, j, instance_info, face_info),
            instance_info,
            side_info,
            back_side);
        return load_frozen_mesh_color(texel);
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
    texel_a = mesh_colors_side_texel(
        texel_a, instance_info, side_info, back_side);
    texel_b = mesh_colors_side_texel(
        texel_b, instance_info, side_info, back_side);
    texel_c = mesh_colors_side_texel(
        texel_c, instance_info, side_info, back_side);
    return (
        weight.x * load_frozen_mesh_color(texel_a)
        + weight.y * load_frozen_mesh_color(texel_b)
        + weight.z * load_frozen_mesh_color(texel_c));
}

float3 get_frozen_mesh_colors_world_flat_normal(HitInfo hit)
{
    InstanceDesc instance_desc = g_scene.instance_descs[hit.instance_id];
    MeshDesc mesh_desc = g_scene.mesh_descs[instance_desc.mesh_id];
    uint base = mesh_desc.index_offset + hit.primitive_index * 3;
    uint i0 = g_scene.indices[base + 0];
    uint i1 = g_scene.indices[base + 1];
    uint i2 = g_scene.indices[base + 2];
    float3 p0 = g_scene.vertices[mesh_desc.vertex_offset + i0].position;
    float3 p1 = g_scene.vertices[mesh_desc.vertex_offset + i1].position;
    float3 p2 = g_scene.vertices[mesh_desc.vertex_offset + i2].position;
    float3 object_normal = compute_flat_normal(
        p0, p1, p2, instance_desc.is_odd_negative());
    return normalize(mul(
        (float3x3)g_scene.inverse_transpose_transforms[instance_desc.transform_id],
        object_normal));
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
        g_output[pixel] = float4(
            g_scene.sample_sky_with_sun_disk(ray.dir), 1.0);
        return;
    }

    float3 flat_normal = get_frozen_mesh_colors_world_flat_normal(hit);
    bool back_side = dot(flat_normal, -ray.dir) <= 0.0;

    MeshColorsInstanceInfo instance_info = mesh_colors_get_instance(
        g_mesh_colors_instance_infos, hit.instance_id);
    MeshColorsSideInfo side_info = mesh_colors_get_side(
        g_mesh_colors_side_infos, hit.instance_id);
    if (back_side && !mesh_colors_is_double_sided(side_info))
    {
        g_output[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    MeshColorsFaceInfo face_info = mesh_colors_get_face(
        g_mesh_colors_face_infos,
        instance_info.face_offset + hit.primitive_index);
    float3 irradiance = sample_frozen_mesh_colors(
        hit.get_barycentrics(),
        instance_info,
        face_info,
        side_info,
        back_side);
    ShadingData shading = g_scene.get_shading_data(hit, -ray.dir);
    float3 color = shading.emissive + shading.diffuse * INV_PI * irradiance;
    g_output[pixel] = float4(color, 1.0);
}
