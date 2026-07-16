import camera;
import common;
import tone_mapper;

static const uint INVALID_ID = 0xffffffffu;
static const uint CONE_RIM_SEGMENTS = 16u;

#ifndef USE_RAY_QUERY
#define USE_RAY_QUERY 0
#endif

#if USE_RAY_QUERY
RaytracingAccelerationStructure g_tlas;
#endif
StructuredBuffer<float3> g_positions;
StructuredBuffer<float3> g_normals;
StructuredBuffer<float2> g_uvs;
StructuredBuffer<uint> g_indices;
StructuredBuffer<uint> g_triangle_mesh_ids;
StructuredBuffer<uint> g_triangle_material_ids;
StructuredBuffer<float4> g_vertex_values;
StructuredBuffer<float4> g_vertex_cone0;
StructuredBuffer<float4> g_vertex_cone1;
Camera g_camera;

static const uint MATERIAL_HAS_BASE_COLOR = 1u << 0u;
static const uint MATERIAL_HAS_METALLIC_ROUGHNESS = 1u << 1u;
static const uint MATERIAL_HAS_NORMAL = 1u << 2u;
static const uint MATERIAL_HAS_OCCLUSION = 1u << 3u;
static const uint MATERIAL_HAS_EMISSIVE = 1u << 4u;
static const uint MATERIAL_DOUBLE_SIDED = 1u << 5u;

struct ViewerMaterial
{
    float4 base_color_factor;
    float3 emissive_factor;
    float roughness_factor;
    float metallic_factor;
    float normal_scale;
    float occlusion_strength;
    uint flags;
    Texture2D<float4>.Handle base_color_texture;
    Texture2D<float4>.Handle metallic_roughness_texture;
    Texture2D<float4>.Handle normal_texture;
    Texture2D<float4>.Handle occlusion_texture;
    Texture2D<float4>.Handle emissive_texture;
};

StructuredBuffer<ViewerMaterial> g_materials;
SamplerState g_material_sampler;

#if !USE_RAY_QUERY
uniform float4 g_raster_base_color_factor;
uniform float3 g_raster_emissive_factor;
uniform float g_raster_roughness_factor;
uniform float g_raster_metallic_factor;
uniform float g_raster_normal_scale;
uniform float g_raster_occlusion_strength;
uniform uint g_raster_material_flags;
Texture2D<float4> g_raster_base_color_texture;
Texture2D<float4> g_raster_metallic_roughness_texture;
Texture2D<float4> g_raster_normal_texture;
Texture2D<float4> g_raster_occlusion_texture;
Texture2D<float4> g_raster_emissive_texture;
#endif

RWTexture2D<float4> g_normal_roughness;
RWTexture2D<float4> g_albedo_metallic;
RWTexture2D<float4> g_emissive_occlusion;
RWTexture2D<float4> g_baked_value;
RWTexture2D<float4> g_visibility_cone;
RWTexture2D<float2> g_cone_params;
RWTexture2D<float> g_depth;
RWTexture2D<uint4> g_ids;

float3 interpolate3(float3 v0, float3 v1, float3 v2, float3 bary)
{
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}

float4 interpolate4(float4 v0, float4 v1, float4 v2, float3 bary)
{
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void clear_gbuffer_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_depth.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    g_normal_roughness[pixel] = float4(0.0, 0.0, 1.0, 1.0);
    g_albedo_metallic[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    g_emissive_occlusion[pixel] = float4(0.0, 0.0, 0.0, 1.0);
    g_baked_value[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    g_visibility_cone[pixel] = float4(0.0, 0.0, 1.0, 0.0);
    g_cone_params[pixel] = float2(0.0, 0.0);
    g_depth[pixel] = 0.0;
    g_ids[pixel] = uint4(INVALID_ID, INVALID_ID, INVALID_ID, 0u);
}

#if USE_RAY_QUERY
[shader("compute")]
[numthreads(16, 16, 1)]
void gbuffer_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_depth.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    Ray ray = g_camera.compute_ray_pinhole(pixel, dim, false);
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xff, ray.to_ray_desc());
    while (query.Proceed())
    {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        uint candidate_triangle = query.CandidatePrimitiveIndex();
        ViewerMaterial candidate_material = g_materials[g_triangle_material_ids[candidate_triangle]];
        uint c0 = g_indices[candidate_triangle * 3u + 0u];
        uint c1 = g_indices[candidate_triangle * 3u + 1u];
        uint c2 = g_indices[candidate_triangle * 3u + 2u];
        float3 face_normal = cross(g_positions[c1] - g_positions[c0], g_positions[c2] - g_positions[c0]);
        bool front_facing = dot(face_normal, ray.dir) < 0.0;
        if (front_facing || (candidate_material.flags & MATERIAL_DOUBLE_SIDED) != 0u)
            query.CommitNonOpaqueTriangleHit();
    }

    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return;

    uint triangle_id = query.CommittedPrimitiveIndex();
    float2 hit_bary = query.CommittedTriangleBarycentrics();
    float3 bary = float3(1.0 - hit_bary.x - hit_bary.y, hit_bary.x, hit_bary.y);
    uint i0 = g_indices[triangle_id * 3u + 0u];
    uint i1 = g_indices[triangle_id * 3u + 1u];
    uint i2 = g_indices[triangle_id * 3u + 2u];
    ViewerMaterial material = g_materials[g_triangle_material_ids[triangle_id]];
    float2 uv = bary.x * g_uvs[i0] + bary.y * g_uvs[i1] + bary.z * g_uvs[i2];

    float3 normal = normalize(interpolate3(g_normals[i0], g_normals[i1], g_normals[i2], bary));
    if (dot(normal, -ray.dir) < 0.0)
        normal = -normal;
    if ((material.flags & MATERIAL_HAS_NORMAL) != 0u)
    {
        float3 dp1 = g_positions[i1] - g_positions[i0];
        float3 dp2 = g_positions[i2] - g_positions[i0];
        float2 duv1 = g_uvs[i1] - g_uvs[i0];
        float2 duv2 = g_uvs[i2] - g_uvs[i0];
        float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
        if (abs(determinant) > 1e-7)
        {
            float3 tangent = normalize((dp1 * duv2.y - dp2 * duv1.y) / determinant);
            tangent = normalize(tangent - normal * dot(tangent, normal));
            float3 bitangent = normalize(cross(normal, tangent));
            float3 tangent_normal = material.normal_texture.SampleLevel(g_material_sampler, uv, 0.0).xyz * 2.0 - 1.0;
            tangent_normal.xy *= material.normal_scale;
            normal = normalize(tangent * tangent_normal.x + bitangent * tangent_normal.y + normal * tangent_normal.z);
        }
    }

    float4 base_color = material.base_color_factor;
    if ((material.flags & MATERIAL_HAS_BASE_COLOR) != 0u)
        base_color *= material.base_color_texture.SampleLevel(g_material_sampler, uv, 0.0);
    float roughness = material.roughness_factor;
    float metallic = material.metallic_factor;
    if ((material.flags & MATERIAL_HAS_METALLIC_ROUGHNESS) != 0u)
    {
        float4 metallic_roughness = material.metallic_roughness_texture.SampleLevel(g_material_sampler, uv, 0.0);
        roughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    float3 emissive = material.emissive_factor;
    if ((material.flags & MATERIAL_HAS_EMISSIVE) != 0u)
        emissive *= material.emissive_texture.SampleLevel(g_material_sampler, uv, 0.0).rgb;
    float occlusion = 1.0;
    if ((material.flags & MATERIAL_HAS_OCCLUSION) != 0u)
    {
        float sampled_occlusion = material.occlusion_texture.SampleLevel(g_material_sampler, uv, 0.0).r;
        occlusion = lerp(1.0, sampled_occlusion, saturate(material.occlusion_strength));
    }
    float4 baked = interpolate4(g_vertex_values[i0], g_vertex_values[i1], g_vertex_values[i2], bary);
    float aperture = dot(
        float3(g_vertex_cone0[i0].w, g_vertex_cone0[i1].w, g_vertex_cone0[i2].w),
        bary);
    float scale = dot(
        float3(g_vertex_cone1[i0].x, g_vertex_cone1[i1].x, g_vertex_cone1[i2].x),
        bary);
    float3 interpolated_cone_direction = interpolate3(
        g_vertex_cone0[i0].xyz,
        g_vertex_cone0[i1].xyz,
        g_vertex_cone0[i2].xyz,
        bary);
    float cone_direction_length_squared = dot(interpolated_cone_direction, interpolated_cone_direction);
    float3 cone_direction = cone_direction_length_squared > 1e-12
        ? interpolated_cone_direction * rsqrt(cone_direction_length_squared)
        : normal;

    float t = query.CommittedRayT();
    float3 world_position = ray.origin + ray.dir * t;
    float linear_depth = -mul(g_camera.view_matrix, float4(world_position, 1.0)).z;
    uint nearest_vertex = bary.x >= bary.y && bary.x >= bary.z ? i0 : (bary.y >= bary.z ? i1 : i2);

    g_normal_roughness[pixel] = float4(normal, saturate(roughness));
    g_albedo_metallic[pixel] = float4(max(base_color.rgb, 0.0), saturate(metallic));
    g_emissive_occlusion[pixel] = float4(max(emissive, 0.0), saturate(occlusion));
    g_baked_value[pixel] = baked;
    g_visibility_cone[pixel] = float4(cone_direction, saturate(aperture / (0.5 * PI)));
    g_cone_params[pixel] = float2(aperture, scale);
    g_depth[pixel] = max(linear_depth, 0.0);
    g_ids[pixel] = uint4(triangle_id, g_triangle_mesh_ids[triangle_id], nearest_vertex, 1u);
}
#endif

struct RasterGBufferVertex
{
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 baked : TEXCOORD3;
    float4 cone0 : TEXCOORD4;
    float cone_scale : TEXCOORD5;
    float3 bary : TEXCOORD6;
    nointerpolation uint triangle_id : TEXCOORD7;
};

struct RasterGBufferOutput
{
    float4 normal_roughness : SV_Target0;
    float4 albedo_metallic : SV_Target1;
    float4 emissive_occlusion : SV_Target2;
    float4 baked_value : SV_Target3;
    float4 visibility_cone : SV_Target4;
    float2 cone_params : SV_Target5;
    float linear_depth : SV_Target6;
    uint4 ids : SV_Target7;
};

[shader("vertex")]
RasterGBufferVertex gbuffer_raster_vertex(
    float3 world_position : POSITION,
    float3 normal : NORMAL,
    float2 uv : TEXCOORD0,
    float4 baked : TEXCOORD1,
    float4 cone0 : TEXCOORD2,
    float cone_scale : TEXCOORD3,
    float3 bary : TEXCOORD4,
    uint triangle_id : TEXCOORD5)
{
    RasterGBufferVertex output;
    output.position = mul(g_camera.view_proj_matrix_no_jitter, float4(world_position, 1.0));
    output.world_position = world_position;
    output.normal = normal;
    output.uv = uv;
    output.baked = baked;
    output.cone0 = cone0;
    output.cone_scale = cone_scale;
    output.bary = bary;
    output.triangle_id = triangle_id;
    return output;
}

[shader("fragment")]
RasterGBufferOutput gbuffer_raster_fragment(RasterGBufferVertex input)
{
    uint triangle_id = input.triangle_id;
    uint i0 = g_indices[triangle_id * 3u + 0u];
    uint i1 = g_indices[triangle_id * 3u + 1u];
    uint i2 = g_indices[triangle_id * 3u + 2u];

    float3 view_direction = normalize(g_camera.position - input.world_position);
    float3 face_normal_value = cross(g_positions[i1] - g_positions[i0], g_positions[i2] - g_positions[i0]);
    float face_normal_length_squared = dot(face_normal_value, face_normal_value);
    float3 face_normal = face_normal_length_squared > 1e-12
        ? face_normal_value * rsqrt(face_normal_length_squared)
        : float3(0.0, 1.0, 0.0);
    float normal_length_squared = dot(input.normal, input.normal);
    float3 normal = normal_length_squared > 1e-12
        ? input.normal * rsqrt(normal_length_squared)
        : face_normal;
    if (dot(normal, view_direction) < 0.0)
        normal = -normal;
    if ((g_raster_material_flags & MATERIAL_HAS_NORMAL) != 0u)
    {
        float3 dp1 = g_positions[i1] - g_positions[i0];
        float3 dp2 = g_positions[i2] - g_positions[i0];
        float2 duv1 = g_uvs[i1] - g_uvs[i0];
        float2 duv2 = g_uvs[i2] - g_uvs[i0];
        float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
        if (abs(determinant) > 1e-7)
        {
            float3 tangent = (dp1 * duv2.y - dp2 * duv1.y) / determinant;
            tangent -= normal * dot(tangent, normal);
            float tangent_length_squared = dot(tangent, tangent);
            if (tangent_length_squared > 1e-12)
            {
                tangent *= rsqrt(tangent_length_squared);
                float3 bitangent = cross(normal, tangent);
                float3 tangent_normal = g_raster_normal_texture.Sample(g_material_sampler, input.uv).xyz * 2.0 - 1.0;
                tangent_normal.xy *= g_raster_normal_scale;
                float3 mapped_normal =
                    tangent * tangent_normal.x + bitangent * tangent_normal.y + normal * tangent_normal.z;
                float mapped_length_squared = dot(mapped_normal, mapped_normal);
                if (mapped_length_squared > 1e-12)
                    normal = mapped_normal * rsqrt(mapped_length_squared);
            }
        }
    }

    float4 base_color = g_raster_base_color_factor;
    if ((g_raster_material_flags & MATERIAL_HAS_BASE_COLOR) != 0u)
        base_color *= g_raster_base_color_texture.Sample(g_material_sampler, input.uv);
    float roughness = g_raster_roughness_factor;
    float metallic = g_raster_metallic_factor;
    if ((g_raster_material_flags & MATERIAL_HAS_METALLIC_ROUGHNESS) != 0u)
    {
        float4 metallic_roughness = g_raster_metallic_roughness_texture.Sample(g_material_sampler, input.uv);
        roughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    float3 emissive = g_raster_emissive_factor;
    if ((g_raster_material_flags & MATERIAL_HAS_EMISSIVE) != 0u)
        emissive *= g_raster_emissive_texture.Sample(g_material_sampler, input.uv).rgb;
    float occlusion = 1.0;
    if ((g_raster_material_flags & MATERIAL_HAS_OCCLUSION) != 0u)
    {
        float sampled_occlusion = g_raster_occlusion_texture.Sample(g_material_sampler, input.uv).r;
        occlusion = lerp(1.0, sampled_occlusion, saturate(g_raster_occlusion_strength));
    }

    float cone_direction_length_squared = dot(input.cone0.xyz, input.cone0.xyz);
    float3 cone_direction = cone_direction_length_squared > 1e-12
        ? input.cone0.xyz * rsqrt(cone_direction_length_squared)
        : normal;
    float aperture = input.cone0.w;
    float3 bary = input.bary;
    uint nearest_vertex = bary.x >= bary.y && bary.x >= bary.z ? i0 : (bary.y >= bary.z ? i1 : i2);
    float linear_depth = -mul(g_camera.view_matrix, float4(input.world_position, 1.0)).z;

    RasterGBufferOutput output;
    output.normal_roughness = float4(normal, saturate(roughness));
    output.albedo_metallic = float4(max(base_color.rgb, 0.0), saturate(metallic));
    output.emissive_occlusion = float4(max(emissive, 0.0), saturate(occlusion));
    output.baked_value = input.baked;
    output.visibility_cone = float4(cone_direction, saturate(aperture / (0.5 * PI)));
    output.cone_params = float2(aperture, input.cone_scale);
    output.linear_depth = max(linear_depth, 0.0);
    output.ids = uint4(triangle_id, g_triangle_mesh_ids[triangle_id], nearest_vertex, 1u);
    return output;
}

Texture2D<uint4> g_pick_ids;
Texture2D<float> g_pick_depth;
StructuredBuffer<float3> g_pick_positions;
StructuredBuffer<uint> g_pick_vertex_mesh_ids;
RWStructuredBuffer<uint4> g_pick_result;
uniform uint2 g_pick_pixel;
uniform uint g_pick_vertex_count;
uniform float g_pick_radius;
uniform float g_pick_depth_tolerance;
Camera g_pick_camera;

float3 project_pick_vertex(float3 position, uint2 dim)
{
    float4 clip = mul(g_pick_camera.view_proj_matrix_no_jitter, float4(position, 1.0));
    if (clip.w <= 1e-6)
        return float3(-1e9, -1e9, 2.0);
    float3 ndc = clip.xyz / clip.w;
    return float3(
        (ndc.x * 0.5 + 0.5) * float(dim.x),
        (0.5 - ndc.y * 0.5) * float(dim.y),
        ndc.z);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pick_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_pick_ids.GetDimensions(dim.x, dim.y);
    uint2 pixel = min(g_pick_pixel, max(dim, uint2(1u, 1u)) - 1u);
    uint4 surface_hit = g_pick_ids[pixel];
    if (surface_hit.w == 0u)
    {
        g_pick_result[0] = uint4(INVALID_ID, INVALID_ID, asuint(0.0), 0u);
        return;
    }

    uint hit_mesh_id = surface_hit.y;
    float best_distance = max(g_pick_radius, 0.0) + 1.0;
    uint best_vertex = INVALID_ID;
    [loop]
    for (uint vertex_id = 0u; vertex_id < g_pick_vertex_count; ++vertex_id)
    {
        if (g_pick_vertex_mesh_ids[vertex_id] != hit_mesh_id)
            continue;
        float3 position = g_pick_positions[vertex_id];
        float3 projected = project_pick_vertex(position, dim);
        if (projected.z < -1.0 || projected.z > 1.0)
            continue;
        if (any(projected.xy < 0.0) || any(projected.xy >= float2(dim)))
            continue;

        float distance_pixels = length(projected.xy - (float2(pixel) + 0.5));
        if (distance_pixels > g_pick_radius || distance_pixels >= best_distance)
            continue;

        uint2 vertex_pixel = min(uint2(projected.xy), dim - 1u);
        float scene_depth = g_pick_depth[vertex_pixel];
        float vertex_depth = -mul(g_pick_camera.view_matrix, float4(position, 1.0)).z;
        if (scene_depth <= 0.0 || vertex_depth > scene_depth + g_pick_depth_tolerance)
            continue;

        best_distance = distance_pixels;
        best_vertex = vertex_id;
    }

    g_pick_result[0] = best_vertex == INVALID_ID
        ? uint4(INVALID_ID, hit_mesh_id, asuint(0.0), 0u)
        : uint4(best_vertex, hit_mesh_id, asuint(best_distance), 1u);
}

Texture2D<float4> g_comp_normal_roughness;
Texture2D<float4> g_comp_albedo_metallic;
Texture2D<float4> g_comp_emissive_occlusion;
Texture2D<float4> g_comp_baked_value;
Texture2D<float4> g_comp_visibility_cone;
Texture2D<float2> g_comp_cone_params;
Texture2D<float> g_comp_depth;
Texture2D<uint4> g_comp_ids;
Texture2DArray<float4> g_comp_sky;
Texture2DArray<float4> g_comp_specular;
Texture2D<float2> g_comp_dfg;
Texture2D<float> g_comp_specular_occlusion;
StructuredBuffer<float4> g_comp_sh;
SamplerState g_comp_environment_sampler;
Camera g_comp_camera;
RWTexture2D<float4> g_output;

uniform uint g_view_mode;
uniform float g_depth_scale;
uniform uint g_selected_valid;
uniform float3 g_selected_position;
uniform float g_selected_aperture;
uniform float3 g_selected_axis;
uniform float g_selected_scale;
uniform float g_selected_length;
uniform uint g_show_cone;
uniform uint g_show_diagram;
uniform float4x4 g_view_proj;
uniform float g_environment_exposure;
uniform float g_environment_rotation;
uniform float g_specular_mip_count;
uniform uint g_show_environment;
uniform uint g_apply_visibility;
uniform uint g_has_visibility;

float3 rotate_environment(float3 direction)
{
    float cosine = cos(g_environment_rotation);
    float sine = sin(g_environment_rotation);
    return float3(
        cosine * direction.x - sine * direction.z,
        direction.y,
        sine * direction.x + cosine * direction.z);
}

float sh_basis_comp(uint coefficient, float3 direction)
{
    float x = direction.x;
    float y = direction.y;
    float z = direction.z;
    if (coefficient == 0u) return 0.2820947918;
    if (coefficient == 1u) return 0.4886025119 * y;
    if (coefficient == 2u) return 0.4886025119 * z;
    if (coefficient == 3u) return 0.4886025119 * x;
    if (coefficient == 4u) return 1.0925484306 * x * y;
    if (coefficient == 5u) return 1.0925484306 * y * z;
    if (coefficient == 6u) return 0.3153915653 * (3.0 * y * y - 1.0);
    if (coefficient == 7u) return 1.0925484306 * x * z;
    return 0.5462742153 * (x * x - z * z);
}

float3 diffuse_irradiance(float3 normal)
{
    float3 direction = rotate_environment(normal);
    float3 irradiance = 0.0;
    [unroll]
    for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
    {
        float convolution = coefficient == 0u ? PI : (coefficient < 4u ? 2.0 * PI / 3.0 : PI / 4.0);
        irradiance += g_comp_sh[coefficient].rgb * (sh_basis_comp(coefficient, direction) * convolution);
    }
    return max(irradiance, 0.0);
}

float evaluate_pmr_ambient_occlusion(float3 cone_direction, float aperture, float scale, float3 normal)
{
    float cos_theta = saturate(dot(cone_direction, normal));
    float corrected_cos_theta = lerp(cos_theta, cos_theta * 0.5 + 0.5, aperture);
    float aperture_visibility = aperture * aperture * (3.0 - 2.0 * aperture);
    return corrected_cos_theta * aperture_visibility * scale;
}

float sample_pmr_specular_occlusion(float aperture, float cos_beta, float roughness)
{
    static const float lut_size = 16.0;
    float3 params = float3(
        saturate((roughness - 0.089) / (1.0 - 0.089)),
        saturate(cos_beta * 0.5 + 0.5),
        saturate(aperture));
    float3 lut_position = params * (lut_size - 1.0);
    float slice0 = floor(lut_position.z);
    float slice1 = min(slice0 + 1.0, lut_size - 1.0);
    float slice_weight = lut_position.z - slice0;
    float2 atlas_size = float2(lut_size * lut_size, lut_size);
    float2 uv0 = (float2(lut_position.x + slice0 * lut_size, lut_position.y) + 0.5) / atlas_size;
    float2 uv1 = (float2(lut_position.x + slice1 * lut_size, lut_position.y) + 0.5) / atlas_size;
    float value0 = g_comp_specular_occlusion.SampleLevel(g_comp_environment_sampler, uv0, 0.0);
    float value1 = g_comp_specular_occlusion.SampleLevel(g_comp_environment_sampler, uv1, 0.0);
    return lerp(value0, value1, slice_weight);
}

float evaluate_pmr_specular_occlusion(
    float3 cone_direction,
    float aperture,
    float scale,
    float3 normal,
    float3 view,
    float roughness)
{
    float3 reflection = reflect(-view, normal);
    float occlusion = sample_pmr_specular_occlusion(aperture, dot(cone_direction, reflection), roughness);
    return occlusion * scale;
}

float3 evaluate_pbr_ibl(
    float3 normal,
    float3 view,
    float3 base_color,
    float metallic,
    float roughness,
    float diffuse_visibility,
    float specular_visibility)
{
    float no_v = max(dot(normal, view), 1e-4);
    float3 diffuse_color = base_color * (1.0 - metallic);
    float3 f0 = lerp(float3(0.04), base_color, metallic);
    float3 reflected = rotate_environment(reflect(-view, normal));
    float lod = max(g_specular_mip_count - 1.0, 0.0) * roughness * (2.0 - roughness);
    float3 prefiltered = g_comp_specular.SampleLevel(
        g_comp_environment_sampler,
        direction_to_cube_array_uv(reflected),
        lod).rgb;
    float2 dfg = g_comp_dfg.SampleLevel(g_comp_environment_sampler, float2(no_v, roughness), 0.0);
    float3 diffuse = diffuse_irradiance(normal) * (diffuse_color / PI) * diffuse_visibility;
    float3 specular = prefiltered * (f0 * dfg.x + dfg.y) * specular_visibility;
    return diffuse + specular;
}

float3 hash_color(uint value)
{
    uint hash = value * 747796405u + 2891336453u;
    hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
    hash = (hash >> 22u) ^ hash;
    return float3(
        float(hash & 255u),
        float((hash >> 8u) & 255u),
        float((hash >> 16u) & 255u)) / 255.0;
}

float heat(float value, float center)
{
    return saturate(1.5 - abs(value * 4.0 - center));
}

float3 heat_map(float value)
{
    value = saturate(value);
    return float3(heat(value, 3.0), heat(value, 2.0), heat(value, 1.0));
}

float point_segment_distance(float2 point, float2 start, float2 end)
{
    float2 delta = end - start;
    float denominator = max(dot(delta, delta), 1e-6);
    float t = saturate(dot(point - start, delta) / denominator);
    return length(point - (start + delta * t));
}

float3 project_world(float3 position, uint2 dim)
{
    float4 clip = mul(g_view_proj, float4(position, 1.0));
    if (clip.w <= 1e-6)
        return float3(-1e9, -1e9, 2.0);
    float3 ndc = clip.xyz / clip.w;
    return float3(
        (ndc.x * 0.5 + 0.5) * float(dim.x),
        (0.5 - ndc.y * 0.5) * float(dim.y),
        ndc.z);
}

float3 overlay_selected_cone(float3 color, float2 pixel_center, uint2 dim)
{
    if (g_selected_valid == 0u || g_show_cone == 0u)
        return color;

    float3 axis = normalize(g_selected_axis);
    float3 helper = abs(axis.z) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 tangent = normalize(cross(helper, axis));
    float3 bitangent = normalize(cross(axis, tangent));
    float aperture = clamp(g_selected_aperture, 0.0, PI);
    float cosine = cos(aperture);
    float sine = sin(aperture);
    float3 origin_screen = project_world(g_selected_position, dim);
    float3 axis_screen = project_world(g_selected_position + axis * g_selected_length, dim);

    float axis_distance = point_segment_distance(pixel_center, origin_screen.xy, axis_screen.xy);
    if (axis_distance < 2.2)
        color = lerp(color, float3(1.0, 0.78, 0.08), 0.95);

    float boundary_distance = 1e9;
    float2 previous = float2(0.0);
    [unroll]
    for (uint segment = 0u; segment <= CONE_RIM_SEGMENTS; ++segment)
    {
        uint wrapped = segment % CONE_RIM_SEGMENTS;
        float angle = (2.0 * PI * float(wrapped)) / float(CONE_RIM_SEGMENTS);
        float3 radial = tangent * cos(angle) + bitangent * sin(angle);
        float3 direction = axis * cosine + radial * sine;
        float2 rim = project_world(g_selected_position + direction * g_selected_length, dim).xy;
        if (segment > 0u)
            boundary_distance = min(boundary_distance, point_segment_distance(pixel_center, previous, rim));
        if (segment < CONE_RIM_SEGMENTS && (segment % 4u) == 0u)
            boundary_distance = min(boundary_distance, point_segment_distance(pixel_center, origin_screen.xy, rim));
        previous = rim;
    }
    if (boundary_distance < 1.7)
        color = lerp(color, float3(0.05, 0.9, 1.0), 0.95);
    if (length(pixel_center - origin_screen.xy) < 4.0)
        color = float3(1.0);
    return color;
}

float3 overlay_cone_diagram(float3 color, float2 pixel_center, uint2 dim)
{
    if (g_selected_valid == 0u || g_show_diagram == 0u)
        return color;

    float2 panel_size = float2(230.0, 190.0);
    float2 panel_min = float2(16.0, float(dim.y) - panel_size.y - 16.0);
    float2 panel_max = panel_min + panel_size;
    if (any(pixel_center < panel_min) || any(pixel_center > panel_max))
        return color;

    color = float3(0.045, 0.065, 0.075);
    float2 center = panel_min + float2(panel_size.x * 0.5, panel_size.y * 0.56);
    float radius = 72.0;
    float2 offset = pixel_center - center;
    float distance_from_center = length(offset);
    float aperture = clamp(g_selected_aperture, 0.0, PI);
    float2 up = float2(0.0, -1.0);
    if (distance_from_center <= radius && distance_from_center > 1e-5)
    {
        float angle_from_axis = acos(clamp(dot(offset / distance_from_center, up), -1.0, 1.0));
        if (angle_from_axis <= aperture)
            color = lerp(color, float3(0.05, 0.65, 0.75), 0.3);
    }

    if (abs(distance_from_center - radius) < 1.1)
        color = float3(0.45, 0.52, 0.55);
    float2 left_boundary = float2(-sin(aperture), -cos(aperture));
    float2 right_boundary = float2(sin(aperture), -cos(aperture));
    float boundary_distance = min(
        point_segment_distance(pixel_center, center, center + left_boundary * radius),
        point_segment_distance(pixel_center, center, center + right_boundary * radius));
    if (boundary_distance < 2.0)
        color = float3(0.05, 0.9, 1.0);
    if (point_segment_distance(pixel_center, center, center + up * radius) < 2.0)
        color = float3(1.0, 0.78, 0.08);
    if (distance_from_center < 3.5)
        color = float3(1.0);
    return color;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void composite_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_output.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    uint4 ids = g_comp_ids[pixel];
    bool valid = ids.w != 0u;
    Ray camera_ray = g_comp_camera.compute_ray_pinhole(pixel, dim, false);
    float3 color = g_show_environment != 0u
        ? g_comp_sky.SampleLevel(
            g_comp_environment_sampler,
            direction_to_cube_array_uv(rotate_environment(camera_ray.dir)),
            0.0).rgb
        : float3(0.035, 0.045, 0.055);
    bool tone_map = g_view_mode == 0u;
    if (valid)
    {
        float4 normal_roughness = g_comp_normal_roughness[pixel];
        float4 albedo_metallic = g_comp_albedo_metallic[pixel];
        float4 emissive_occlusion = g_comp_emissive_occlusion[pixel];
        float4 baked = g_comp_baked_value[pixel];
        float4 visibility_cone = g_comp_visibility_cone[pixel];
        float2 cone_params = g_comp_cone_params[pixel];
        float depth = g_comp_depth[pixel];
        float3 normal = normalize(normal_roughness.xyz);
        float3 view = normalize(-camera_ray.dir);
        float roughness = clamp(normal_roughness.w, 0.089, 1.0);
        float3 cone_direction = normalize(visibility_cone.xyz);
        float cone_scale = saturate(cone_params.y);
        float cone_ambient_occlusion = g_has_visibility != 0u
            ? evaluate_pmr_ambient_occlusion(
                cone_direction,
                saturate(visibility_cone.w),
                cone_scale,
                normal)
            : 1.0;
        float cone_specular_occlusion = g_has_visibility != 0u
            ? evaluate_pmr_specular_occlusion(
                cone_direction,
                saturate(visibility_cone.w),
                cone_scale,
                normal,
                view,
                roughness)
            : 1.0;
        if (g_view_mode == 0u)
        {
            float diffuse_visibility = saturate(emissive_occlusion.a);
            float specular_visibility = 1.0;
            if (g_apply_visibility != 0u && g_has_visibility != 0u)
            {
                diffuse_visibility *= cone_ambient_occlusion;
                specular_visibility = cone_specular_occlusion;
            }
            color = evaluate_pbr_ibl(
                normal,
                view,
                max(albedo_metallic.rgb, 0.0),
                saturate(albedo_metallic.w),
                roughness,
                saturate(diffuse_visibility),
                saturate(specular_visibility)) + emissive_occlusion.rgb;
        }
        else if (g_view_mode == 1u)
            color = saturate(baked.rgb);
        else if (g_view_mode == 2u)
            color = saturate(albedo_metallic.rgb);
        else if (g_view_mode == 3u)
            color = normal_roughness.xyz * 0.5 + 0.5;
        else if (g_view_mode == 4u)
            color = heat_map(saturate(depth / max(g_depth_scale, 1e-5)));
        else if (g_view_mode == 5u)
            color = hash_color(ids.y + 1u);
        else if (g_view_mode == 6u)
            color = heat_map(saturate(cone_params.x / (0.5 * PI)));
        else if (g_view_mode == 7u)
            color = heat_map(max(cone_params.y, 0.0) / (1.0 + max(cone_params.y, 0.0)));
        else if (g_view_mode == 8u)
            color = normal_roughness.www;
        else if (g_view_mode == 9u)
            color = albedo_metallic.www;
        else if (g_view_mode == 10u)
            color = emissive_occlusion.rgb;
        else if (g_view_mode == 11u)
            color = emissive_occlusion.aaa;
        else if (g_view_mode == 12u)
            color = cone_ambient_occlusion.xxx;
        else if (g_view_mode == 13u)
            color = cone_specular_occlusion.xxx;
    }

    // Keep material debug values literal while still presenting the HDR sky correctly.
    if (tone_map || (!valid && g_show_environment != 0u))
        color = LinearToRec709(LpmToneMap(max(color, 0.0), 12.0, g_environment_exposure, 1.0));

    float2 pixel_center = float2(pixel) + 0.5;
    color = overlay_selected_cone(color, pixel_center, dim);
    color = overlay_cone_diagram(color, pixel_center, dim);
    g_output[pixel] = float4(color, 1.0);
}
