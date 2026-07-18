import common;
import scene;
import mesh_colors;

RWByteAddressBuffer g_mesh_colors;
StructuredBuffer<uint2> g_mesh_colors_face_infos;
StructuredBuffer<uint4> g_mesh_colors_instance_infos;
StructuredBuffer<uint2> g_mesh_colors_side_infos;
uniform uint g_instance_index;
uniform uint g_iteration;
uniform uint g_samples_per_texel;
uniform uint g_max_bounces;
uniform uint g_reset;

static const uint MAX_TEXEL_SAMPLE_COUNT = 65535;

struct TextureSpacePath
{
    Ray ray;
    float3 throughput;
    float3 radiance;
    RNG rng;
};

float3 load_cached_irradiance(uint texel_index)
{
    uint offset = texel_index * MESH_COLORS_PAYLOAD_STRIDE;
    return asfloat(uint3(
        g_mesh_colors.Load(offset),
        g_mesh_colors.Load(offset + 4),
        g_mesh_colors.Load(offset + 8)));
}

uint load_cached_sample_count(uint texel_index)
{
    return g_mesh_colors.Load(texel_index * MESH_COLORS_PAYLOAD_STRIDE + 12);
}

float3 get_world_flat_normal(HitInfo hit)
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

void trace_radiance(inout TextureSpacePath path)
{
    uint bounce = 0;
    while (bounce < g_max_bounces)
    {
        HitInfo hit = g_scene.intersect(path.ray);
        if (!hit.valid)
        {
            path.radiance += path.throughput * g_scene.sample_sky_lut(path.ray.dir);
            return;
        }

        ShadingData shading = g_scene.get_shading_data(hit, -path.ray.dir);
        path.radiance += path.throughput * shading.emissive;
        path.radiance += path.throughput
            * g_scene.sample_directional_light(shading, path.rng.next_2d()).xyz;

        float4 sample = cosine_sample_hemisphere(path.rng.next_2d());
        if (sample.w <= 1e-6)
            return;
        float3 f = shading.diffuse * INV_PI;
        path.throughput *= f * abs(sample.z) / sample.w;
        path.ray = Ray(
            shading.compute_new_ray_origin(),
            shading.shading_frame.to_global(sample.xyz));
        bounce++;
    }
}

float3 estimate_surface_irradiance(
    HitInfo surface_hit,
    float3 view_direction,
    uint local_sample_count,
    inout RNG rng)
{
    ShadingData surface =
        g_scene.get_shading_data(surface_hit, view_direction);
    float3 irradiance_sample = float3(0.0);
    for (uint sample_index = 0; sample_index < local_sample_count; sample_index++)
    {
        float3 sample_irradiance =
            g_scene.sample_directional_irradiance(surface, rng.next_2d());
        float4 direction_sample = cosine_sample_hemisphere(rng.next_2d());
        if (direction_sample.w > 1e-6)
        {
            TextureSpacePath path;
            path.ray = Ray(
                surface.compute_new_ray_origin(),
                surface.shading_frame.to_global(direction_sample.xyz));
            path.throughput = float3(1.0);
            path.radiance = float3(0.0);
            path.rng = rng;
            trace_radiance(path);
            rng = path.rng;
            sample_irradiance +=
                path.radiance * abs(direction_sample.z) / direction_sample.w;
        }
        irradiance_sample += sample_irradiance;
    }
    return irradiance_sample / float(local_sample_count);
}

void store_accumulated_irradiance(
    uint texel_index,
    float3 irradiance_sample)
{
    uint previous_count =
        g_reset != 0 ? 0 : load_cached_sample_count(texel_index);
    float3 previous_irradiance = previous_count == 0
        ? float3(0.0)
        : load_cached_irradiance(texel_index);
    uint next_count = min(previous_count + 1, MAX_TEXEL_SAMPLE_COUNT);
    float weight = 1.0 / float(previous_count + 1);
    float3 accumulated =
        lerp(previous_irradiance, irradiance_sample, weight);

    uint destination = texel_index * MESH_COLORS_PAYLOAD_STRIDE;
    g_mesh_colors.Store(destination + 0, asuint(accumulated.x));
    g_mesh_colors.Store(destination + 4, asuint(accumulated.y));
    g_mesh_colors.Store(destination + 8, asuint(accumulated.z));
    g_mesh_colors.Store(destination + 12, next_count);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint instance_local_texel = dispatch_thread_id.x;
    MeshColorsInstanceInfo instance_info = mesh_colors_get_instance(
        g_mesh_colors_instance_infos, g_instance_index);
    MeshColorsSideInfo side_info = mesh_colors_get_side(
        g_mesh_colors_side_infos, g_instance_index);
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

    uint global_texel = instance_info.texel_offset + instance_local_texel;
    uint face_local_address = instance_local_texel - face_info.address;
    uint2 ij = mesh_colors_address_to_ij(
        face_local_address, face_info.resolution);
    float2 base = float2(ij) / float(face_info.resolution);
    RNG rng = RNG(
        uint2(global_texel, g_instance_index),
        g_iteration * max(g_samples_per_texel, 1u));

    float2 bary_xy = base;
    for (uint attempt = 0; attempt < 32; attempt++)
    {
        float2 candidate = base
            + (rng.next_2d() - 0.5) / float(face_info.resolution);
        if (any(candidate < 0.0) || candidate.x + candidate.y > 1.0)
            continue;
        float3 candidate_barycentrics = float3(
            candidate.x,
            candidate.y,
            1.0 - candidate.x - candidate.y);
        if (mesh_colors_nearest_texel(
            candidate_barycentrics, instance_info, face_info) == global_texel)
        {
            bary_xy = candidate;
            break;
        }
    }

    float3 barycentrics = float3(
        bary_xy.x,
        bary_xy.y,
        max(0.0, 1.0 - bary_xy.x - bary_xy.y));
    HitInfo surface_hit;
    surface_hit.valid = true;
    surface_hit.bary = barycentrics.yz;
    surface_hit.instance_id = g_instance_index;
    surface_hit.primitive_index = primitive_index;

    uint local_sample_count = max(g_samples_per_texel, 1u);
    float3 flat_normal = get_world_flat_normal(surface_hit);
    float3 front_irradiance = estimate_surface_irradiance(
        surface_hit, flat_normal, local_sample_count, rng);
    store_accumulated_irradiance(global_texel, front_irradiance);

    if (mesh_colors_is_double_sided(side_info))
    {
        float3 back_irradiance = estimate_surface_irradiance(
            surface_hit, -flat_normal, local_sample_count, rng);
        uint back_texel = mesh_colors_side_texel(
            global_texel, instance_info, side_info, true);
        store_accumulated_irradiance(back_texel, back_irradiance);
    }
}
