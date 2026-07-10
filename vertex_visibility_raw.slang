import common;
import scene;

StructuredBuffer<float4> g_sample_positions;
StructuredBuffer<float4> g_sample_normals;
StructuredBuffer<float4> g_sample_tangents;
StructuredBuffer<float4> g_local_directions;

RWStructuredBuffer<float4> g_output_cone0;
RWStructuredBuffer<float4> g_output_cone1;
RWStructuredBuffer<float4> g_output_sh0;
RWStructuredBuffer<float4> g_output_sh1;
RWStructuredBuffer<float4> g_output_sh2;
RWStructuredBuffer<float4> g_output_sh3;

uniform uint g_sample_count;
uniform uint g_ray_count;
uniform uint g_ray_offset;
uniform uint g_ray_batch_count;
uniform float g_self_bias;
uniform float g_max_distance;

float3 normalize_or(float3 value, float3 fallback)
{
    float len2 = dot(value, value);
    if (len2 > 1e-16)
        return value * rsqrt(len2);

    float fallback_len2 = dot(fallback, fallback);
    if (fallback_len2 > 1e-16)
        return fallback * rsqrt(fallback_len2);

    return float3(0.0, 0.0, 1.0);
}

float3 fallback_tangent(float3 normal)
{
    normal = normalize_or(normal, float3(0.0, 0.0, 1.0));
    float3 helper = abs(normal.z) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    return normalize_or(cross(helper, normal), float3(1.0, 0.0, 0.0));
}

float3 tangent_space_to_world(float3 local_dir, float3 normal, float4 tangent)
{
    float3 tangent_xyz = normalize_or(tangent.xyz, fallback_tangent(normal));
    float handedness = tangent.w < 0.0 ? -1.0 : 1.0;
    float3 bitangent = normalize_or(cross(normal, tangent_xyz) * handedness, fallback_tangent(normal));
    return normalize_or(local_dir.x * tangent_xyz + local_dir.y * bitangent + local_dir.z * normal, normal);
}

float radical_inverse_base2(uint value)
{
    value = (value << 16) | (value >> 16);
    value = ((value & 0x55555555u) << 1) | ((value & 0xAAAAAAAAu) >> 1);
    value = ((value & 0x33333333u) << 2) | ((value & 0xCCCCCCCCu) >> 2);
    value = ((value & 0x0F0F0F0Fu) << 4) | ((value & 0xF0F0F0F0u) >> 4);
    value = ((value & 0x00FF00FFu) << 8) | ((value & 0xFF00FF00u) >> 8);
    return float(value) * 2.3283064365386963e-10;
}

float3 sample_pmr_uniform_sphere(uint ray_index, uint ray_count)
{
    float2 u = float2(float(ray_index) / float(ray_count), radical_inverse_base2(ray_index));
    float phi = u.x * (2.0 * PI);
    float cos_theta = 1.0 - 2.0 * u.y;
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return float3(cos(phi) * sin_theta, cos_theta, sin(phi) * sin_theta);
}

void project_pmr_sh16(
    float3 dir,
    float value,
    out float4 sh0,
    out float4 sh1,
    out float4 sh2,
    out float4 sh3)
{
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    sh0 = value * float4(
        0.282094791774,
        -0.488602511903 * y,
        0.488602511903 * z,
        -0.488602511903 * x);
    sh1 = value * float4(
        1.09254843059 * x * y,
        -1.09254843059 * y * z,
        0.315391565253 * (-1.0 + 3.0 * z2),
        -1.09254843059 * x * z);
    sh2 = value * float4(
        0.546274215296 * (x2 - y2),
        -0.590043589927 * (3.0 * x2 * y - y2 * y),
        2.89061144264 * x * y * z,
        -0.457045799464 * y * (-1.0 + 5.0 * z2));
    sh3 = value * float4(
        0.37317633259 * z * (-3.0 + 5.0 * z2),
        -0.457045799464 * x * (-1.0 + 5.0 * z2),
        1.44530572132 * (x2 - y2) * z,
        -0.590043589927 * (x2 * x - 3.0 * x * y2));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint sample_index = tid.x;
    if (sample_index >= g_sample_count)
        return;

    uint ray_count = max(g_ray_count, 1u);
    float3 position = g_sample_positions[sample_index].xyz;
    float3 normal = normalize_or(g_sample_normals[sample_index].xyz, float3(0.0, 0.0, 1.0));
    float4 tangent = g_sample_tangents[sample_index];
    float ray_t_min = max(g_self_bias * 0.5, 1e-7);
    float ray_t_max = g_max_distance > 0.0 ? g_max_distance : 1.0e30;
    float3 origin = position + normal * max(g_self_bias, 0.0);

    float visible_count = 0.0;
    float3 visible_direction_sum = float3(0.0);
    [loop]
    for (uint ray_index = 0; ray_index < ray_count; ++ray_index)
    {
        float3 local_dir = normalize_or(g_local_directions[ray_index].xyz, float3(0.0, 0.0, 1.0));
        float3 world_dir = tangent_space_to_world(local_dir, normal, tangent);
        float visibility = g_scene.get_visibility(Ray(origin, world_dir, ray_t_min, ray_t_max));
        if (visibility > 0.5)
        {
            visible_count += 1.0;
            visible_direction_sum += world_dir;
        }
    }

    float visible_fraction = visible_count / float(ray_count);
    float3 cone_direction = visible_count > 0.0 ? normalize_or(visible_direction_sum, normal) : normal;
    float aperture = acos(clamp(1.0 - visible_fraction, 0.0, 1.0));
    float aperture_normalized = clamp(aperture / (0.5 * PI), 0.0, 1.0);
    float cone_scale = aperture_normalized > 1e-6 ? clamp(visible_fraction / aperture_normalized, 0.0, 1.0) : 0.0;

    g_output_cone0[sample_index] = float4(cone_direction, aperture);
    g_output_cone1[sample_index] = float4(cone_scale, visible_fraction, visible_count, float(ray_count));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_pmr_sh(uint3 tid: SV_DispatchThreadID)
{
    uint sample_index = tid.x;
    if (sample_index >= g_sample_count)
        return;

    uint ray_count = max(g_ray_count, 1u);
    float3 position = g_sample_positions[sample_index].xyz;
    float3 normal = normalize_or(g_sample_normals[sample_index].xyz, float3(0.0, 0.0, 1.0));
    float3 origin = position + normal * max(g_self_bias, 0.0);
    float ray_t_max = g_max_distance > 0.0 ? g_max_distance : 1.0e30;
    float sample_weight = (4.0 * PI) / float(ray_count);

    float4 batch0 = float4(0.0);
    float4 batch1 = float4(0.0);
    float4 batch2 = float4(0.0);
    float4 batch3 = float4(0.0);
    uint ray_end = min(g_ray_offset + max(g_ray_batch_count, 1u), ray_count);
    [loop]
    for (uint ray_index = g_ray_offset; ray_index < ray_end; ++ray_index)
    {
        float3 ray_dir = sample_pmr_uniform_sphere(ray_index, ray_count);
        float visibility = g_scene.get_visibility(Ray(origin, ray_dir, 0.0, ray_t_max));
        float4 sh0;
        float4 sh1;
        float4 sh2;
        float4 sh3;
        project_pmr_sh16(ray_dir, visibility * sample_weight, sh0, sh1, sh2, sh3);
        batch0 += sh0;
        batch1 += sh1;
        batch2 += sh2;
        batch3 += sh3;
    }

    float4 accum0 = g_ray_offset == 0 ? float4(0.0) : g_output_sh0[sample_index];
    float4 accum1 = g_ray_offset == 0 ? float4(0.0) : g_output_sh1[sample_index];
    float4 accum2 = g_ray_offset == 0 ? float4(0.0) : g_output_sh2[sample_index];
    float4 accum3 = g_ray_offset == 0 ? float4(0.0) : g_output_sh3[sample_index];
    g_output_sh0[sample_index] = accum0 + batch0;
    g_output_sh1[sample_index] = accum1 + batch1;
    g_output_sh2[sample_index] = accum2 + batch2;
    g_output_sh3[sample_index] = accum3 + batch3;
}
