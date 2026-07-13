import common;

static const uint PREFILTER_SAMPLE_COUNT = 256u;
static const uint DFG_SAMPLE_COUNT = 1024u;
static const uint SPECULAR_OCCLUSION_LUT_SIZE = 16u;
static const uint SPECULAR_OCCLUSION_SAMPLE_COUNT = 256u;
static const float MIN_LINEAR_ROUGHNESS = 0.089;

Texture2D<float4> g_env_equirect;
TextureCube<float4> g_env_sky;
SamplerState g_env_sampler;
RWTexture2D<float4> g_env_output;
RWTexture2D<float2> g_dfg_output;
RWTexture2D<float> g_specular_occlusion_output;
RWStructuredBuffer<float4> g_sh_output;
uniform uint g_env_face;
uniform float g_env_roughness;
uniform float g_env_source_size;
uniform float g_env_source_lod;

float3 cubemap_direction(uint face, float2 uv)
{
    float2 p = uv * 2.0 - 1.0;
    if (face == 0u) return normalize(float3( 1.0, -p.y, -p.x));
    if (face == 1u) return normalize(float3(-1.0, -p.y,  p.x));
    if (face == 2u) return normalize(float3( p.x,  1.0,  p.y));
    if (face == 3u) return normalize(float3( p.x, -1.0, -p.y));
    if (face == 4u) return normalize(float3( p.x, -p.y,  1.0));
    return normalize(float3(-p.x, -p.y, -1.0));
}

float2 wrapped_equirect_uv(float3 direction)
{
    float2 uv = direction_to_equirect_uv(normalize(direction));
    return float2(frac(uv.x), saturate(uv.y));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void equirect_to_cube_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_env_output.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(dim);
    float3 direction = cubemap_direction(g_env_face, uv);
    g_env_output[tid.xy] = float4(g_env_equirect.SampleLevel(g_env_sampler, wrapped_equirect_uv(direction), 0.0).rgb, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void downsample_cube_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_env_output.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(dim);
    float3 direction = cubemap_direction(g_env_face, uv);
    g_env_output[tid.xy] = float4(g_env_sky.SampleLevel(g_env_sampler, direction, g_env_source_lod).rgb, 1.0);
}

float radical_inverse(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint index, uint count)
{
    return float2(float(index) / float(count), radical_inverse(index));
}

float3 tangent_to_world(float3 local_direction, float3 normal)
{
    float3 up = abs(normal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return tangent * local_direction.x + bitangent * local_direction.y + normal * local_direction.z;
}

float3 importance_sample_ggx(float2 xi, float perceptual_roughness, float3 normal)
{
    float alpha = max(perceptual_roughness * perceptual_roughness, 1e-4);
    float alpha2 = alpha * alpha;
    float phi = TWO_PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / max(1.0 + (alpha2 - 1.0) * xi.y, 1e-6));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return normalize(tangent_to_world(float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta), normal));
}

float d_ggx(float no_h, float perceptual_roughness)
{
    float alpha = max(perceptual_roughness * perceptual_roughness, 1e-4);
    float alpha2 = alpha * alpha;
    float f = (no_h * alpha2 - no_h) * no_h + 1.0;
    return alpha2 / max(PI * f * f, 1e-6);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void prefilter_specular_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_env_output.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(dim);
    float3 normal = cubemap_direction(g_env_face, uv);
    if (g_env_roughness <= 1e-5)
    {
        g_env_output[tid.xy] = float4(g_env_sky.SampleLevel(g_env_sampler, normal, 0.0).rgb, 1.0);
        return;
    }

    float3 radiance = 0.0;
    float weight = 0.0;
    float texel_solid_angle = 4.0 * PI / max(6.0 * g_env_source_size * g_env_source_size, 1.0);
    [loop]
    for (uint sample_index = 0u; sample_index < PREFILTER_SAMPLE_COUNT; ++sample_index)
    {
        float3 half_vector = importance_sample_ggx(hammersley(sample_index, PREFILTER_SAMPLE_COUNT), g_env_roughness, normal);
        float vo_h = saturate(dot(normal, half_vector));
        float3 light = normalize(2.0 * vo_h * half_vector - normal);
        float no_l = saturate(dot(normal, light));
        if (no_l <= 0.0)
            continue;
        float no_h = saturate(dot(normal, half_vector));
        float pdf = max(d_ggx(no_h, g_env_roughness) * no_h / max(4.0 * vo_h, 1e-5), 1e-5);
        float sample_solid_angle = 1.0 / (float(PREFILTER_SAMPLE_COUNT) * pdf);
        float source_lod = max(0.5 * log2(sample_solid_angle / texel_solid_angle), 0.0);
        radiance += g_env_sky.SampleLevel(g_env_sampler, light, source_lod).rgb * no_l;
        weight += no_l;
    }
    g_env_output[tid.xy] = float4(radiance / max(weight, 1e-5), 1.0);
}

float geometry_dfg(float no_v, float no_l, float perceptual_roughness)
{
    float alpha = perceptual_roughness * perceptual_roughness;
    float alpha2 = alpha * alpha;
    float ggx_l = no_v * sqrt((-no_l * alpha2 + no_l) * no_l + alpha2);
    float ggx_v = no_l * sqrt((-no_v * alpha2 + no_v) * no_v + alpha2);
    return (2.0 * no_l) / max(ggx_v + ggx_l, 1e-6);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void integrate_dfg_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_dfg_output.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    float no_v = max((float(tid.x) + 0.5) / float(dim.x), 1e-4);
    float roughness = (float(tid.y) + 0.5) / float(dim.y);
    float3 view = float3(sqrt(max(1.0 - no_v * no_v, 0.0)), 0.0, no_v);
    float2 result = 0.0;
    [loop]
    for (uint sample_index = 0u; sample_index < DFG_SAMPLE_COUNT; ++sample_index)
    {
        float3 half_vector = importance_sample_ggx(hammersley(sample_index, DFG_SAMPLE_COUNT), roughness, float3(0.0, 0.0, 1.0));
        float3 light = normalize(2.0 * dot(view, half_vector) * half_vector - view);
        float no_l = saturate(light.z);
        float no_h = saturate(half_vector.z);
        float vo_h = saturate(dot(view, half_vector));
        if (no_l <= 0.0)
            continue;
        float visibility = geometry_dfg(no_v, no_l, roughness) * vo_h / max(no_h, 1e-5);
        float fresnel = pow(1.0 - vo_h, 5.0);
        result.x += (1.0 - fresnel) * visibility;
        result.y += fresnel * visibility;
    }
    g_dfg_output[tid.xy] = result / float(DFG_SAMPLE_COUNT);
}

float smith_masking(float no_v, float alpha)
{
    float no_v2 = max(no_v * no_v, 1e-12);
    return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * (1.0 - no_v2) / no_v2));
}

float3 sample_ggx_vndf(float3 view, float alpha, float2 sample_value)
{
    float3 stretched_view = normalize(float3(alpha * view.x, alpha * view.y, view.z));
    float lensq = stretched_view.x * stretched_view.x + stretched_view.y * stretched_view.y;
    float3 tangent = lensq > 0.0
        ? float3(-stretched_view.y, stretched_view.x, 0.0) * rsqrt(lensq)
        : float3(1.0, 0.0, 0.0);
    float3 bitangent = cross(stretched_view, tangent);
    float radius = sqrt(sample_value.x);
    float phi = TWO_PI * sample_value.y;
    float t1 = radius * cos(phi);
    float t2 = radius * sin(phi);
    float blend = 0.5 * (1.0 + stretched_view.z);
    t2 = (1.0 - blend) * sqrt(max(1.0 - t1 * t1, 0.0)) + blend * t2;
    float3 hemisphere_normal =
        t1 * tangent
        + t2 * bitangent
        + sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0)) * stretched_view;
    return normalize(float3(alpha * hemisphere_normal.x, alpha * hemisphere_normal.y, max(0.0, hemisphere_normal.z)));
}

float integrate_pmr_specular_occlusion(float aperture, float cos_beta, float roughness)
{
    float beta = acos(clamp(cos_beta, -1.0, 1.0));
    float theta_c0 = -0.5 * beta;
    float theta_o0 = 0.5 * beta;
    float theta_o1 = min(beta, 0.5 * PI);
    float theta_c1 = theta_o1 - beta;
    float theta_o = lerp(theta_o0, theta_o1, 0.78);
    float theta_c = lerp(theta_c0, theta_c1, 0.78);

    float3 cone_direction = float3(sin(theta_c), 0.0, cos(theta_c));
    float cos_theta_o = cos(theta_o);
    float sin_theta_o = sqrt(max(1.0 - cos_theta_o * cos_theta_o, 0.0));
    // Matches PMR buildCoordinateFrame(Nx, Wr), including its near-normal fallback.
    float3 frame_tangent = cos_theta_o >= 0.99 ? float3(-1.0, 0.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 frame_bitangent = cross(float3(0.0, 0.0, 1.0), frame_tangent);
    float3 view_world = float3(0.0, 0.0, cos_theta_o) - frame_tangent * sin_theta_o;
    float3 view_local = float3(
        dot(view_world, frame_tangent),
        dot(view_world, frame_bitangent),
        view_world.z);
    float alpha = roughness * roughness;
    float aperture_radians = aperture * (0.5 * PI);
    float accumulated_visibility = 0.0;
    float accumulated_weight = 0.0;

    [loop]
    for (uint sample_index = 0u; sample_index < SPECULAR_OCCLUSION_SAMPLE_COUNT; ++sample_index)
    {
        float3 half_vector = sample_ggx_vndf(
            view_local,
            alpha,
            hammersley(sample_index, SPECULAR_OCCLUSION_SAMPLE_COUNT));
        float3 incoming_local = -view_local + 2.0 * dot(view_local, half_vector) * half_vector;
        if (view_local.z * incoming_local.z <= 0.0)
            continue;
        float sample_weight = smith_masking(incoming_local.z, alpha);
        float3 incoming_world =
            frame_tangent * incoming_local.x
            + frame_bitangent * incoming_local.y
            + float3(0.0, 0.0, incoming_local.z);
        float visible = acos(clamp(dot(cone_direction, incoming_world), -1.0, 1.0)) < aperture_radians
            ? 1.0
            : 0.0;
        accumulated_visibility += visible * sample_weight;
        accumulated_weight += sample_weight;
    }
    return accumulated_weight > 0.0 ? accumulated_visibility / accumulated_weight : 0.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void integrate_specular_occlusion_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_specular_occlusion_output.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;

    uint aperture_index = tid.x / SPECULAR_OCCLUSION_LUT_SIZE;
    uint roughness_index = tid.x % SPECULAR_OCCLUSION_LUT_SIZE;
    float denominator = float(SPECULAR_OCCLUSION_LUT_SIZE - 1u);
    float roughness_param = float(roughness_index) / denominator;
    float cos_beta_param = float(tid.y) / denominator;
    float aperture = float(aperture_index) / denominator;
    float roughness = lerp(MIN_LINEAR_ROUGHNESS, 1.0, roughness_param);
    float cos_beta = lerp(-1.0, 1.0, cos_beta_param);
    g_specular_occlusion_output[tid.xy] = integrate_pmr_specular_occlusion(aperture, cos_beta, roughness);
}

float sh_basis(uint coefficient, float3 direction)
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

groupshared float4 g_sh_shared[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void project_sh_main(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID)
{
    uint2 dim;
    g_env_equirect.GetDimensions(dim.x, dim.y);
    uint coefficient = group_id.x;
    uint lane = group_thread_id.x;
    float3 sum = 0.0;
    uint pixel_count = dim.x * dim.y;
    for (uint index = lane; index < pixel_count; index += 64u)
    {
        uint2 pixel = uint2(index % dim.x, index / dim.x);
        float2 uv = (float2(pixel) + 0.5) / float2(dim);
        float elevation = (0.5 - uv.y) * PI;
        float azimuth = (uv.x + 0.25) * TWO_PI;
        float cos_elevation = cos(elevation);
        float3 direction = float3(cos_elevation * cos(azimuth), sin(elevation), cos_elevation * sin(azimuth));
        float solid_angle = (TWO_PI / float(dim.x)) * (PI / float(dim.y)) * max(cos_elevation, 0.0);
        sum += g_env_equirect.Load(int3(pixel, 0)).rgb * (sh_basis(coefficient, direction) * solid_angle);
    }
    g_sh_shared[lane] = float4(sum, 0.0);
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 32u; offset > 0u; offset >>= 1u)
    {
        if (lane < offset)
            g_sh_shared[lane] += g_sh_shared[lane + offset];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
        g_sh_output[coefficient] = g_sh_shared[0];
}
