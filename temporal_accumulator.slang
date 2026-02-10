// Temporal Accumulator (TAA) shader.
// References NRD Sample's Taa.cs.hlsl with adaptations for Nan's framework.
//
// Features:
//   - Shared memory preload of color + motion in a (16+2*BORDER)^2 tile
//   - 3x3 neighborhood mean/variance for AABB clamping
//   - Closest-depth motion vector dilation
//   - Bicubic (Catmull-Rom) history sampling
//   - Off-screen + color-distance disocclusion detection
//   - Reinhard tonemap in SMEM for stable variance estimation

import common;

// ─── Constants ──────────────────────────────────────────────────────────────
static const int   BORDER  = 2;                 // 5×5 neighborhood (more stable than 3×3)
static const int   GROUP_X = 16;
static const int   GROUP_Y = 16;
static const int   BUFFER_X = GROUP_X + BORDER * 2;
static const int   BUFFER_Y = GROUP_Y + BORDER * 2;

static const float3 LUMA_WEIGHT = float3(0.2126, 0.7152, 0.0722); // BT.709

// ─── Resources ──────────────────────────────────────────────────────────────
Texture2D<float4>   g_input_color;   // current frame path-traced color (.rgb, .a=1)
Texture2D<float4>   g_input_motion;  // .xy = motion in pixels, .z = viewZ delta, .w = viewZ
Texture2D<float4>   g_history;       // previous TAA result (.rgb = color, .a = mixRate)
RWTexture2D<float4> g_output;        // TAA output (.rgb = color, .a = mixRate)
SamplerState        g_linear_sampler;

uniform float2 g_rect_size;         // current resolution
uniform float2 g_inv_rect_size;     // 1.0 / g_rect_size
uniform float2 g_rect_size_prev;    // previous resolution (for history fetch)
uniform float2 g_inv_render_size;   // 1.0 / actual render target size (may differ if rect != render)
uniform float  g_min_alpha;         // minimum blend weight for current frame (e.g. 0.05)
uniform bool   g_reset;             // force reset (first frame / camera cut)
uniform bool   g_enable_clamp;      // enable AABB neighborhood clamping
uniform float  g_sigma_scale;       // AABB sigma multiplier (default 2.0)
uniform float  g_history_sharpness; // bicubic Catmull-Rom sharpness (default 0.66)
uniform float  g_max_convergence;   // adaptive-clamp ceiling  (default 10.0)

// ─── Shared Memory ──────────────────────────────────────────────────────────
groupshared float3 s_Color[BUFFER_Y][BUFFER_X];
groupshared float3 s_Mv[BUFFER_Y][BUFFER_X];     // .xy = motion px, .z = viewZ

// ─── Helpers ────────────────────────────────────────────────────────────────

// Simple Reinhard tonemap for variance estimation in perceptual space
float3 tonemap_reinhard(float3 c)
{
    float luma = dot(c, LUMA_WEIGHT);
    return c / (1.0 + luma);
}

float3 inverse_tonemap_reinhard(float3 c)
{
    float luma = dot(c, LUMA_WEIGHT);
    return c / max(1.0 - luma, 1e-6);
}

// AABB clamp (from NRD MathLib)
float3 clamp_aabb(float3 center, float3 extents, float3 color)
{
    float3 d = color - center;
    float3 ts = abs(extents / (d + 0.0001));
    float t = min(ts.x, min(ts.y, ts.z));
    return center + d * saturate(t);
}

float positive_rcp(float x)
{
    return 1.0 / max(x, 1e-6);
}

// Bicubic filter (5-tap, no corners) -- ported from NRD Shared.hlsli BicubicFilterNoCorners
float4 bicubic_filter_no_corners(Texture2D<float4> tex, SamplerState samp,
                                  float2 sample_pos, float2 inv_resource_size, float sharpness)
{
    float2 center_pos = floor(sample_pos - 0.5) + 0.5;
    float2 f = saturate(sample_pos - center_pos);
    float2 f2 = f * f;
    float2 f3 = f * f2;

    float2 w0 = -sharpness * f3 + 2.0 * sharpness * f2 - sharpness * f;
    float2 w1 = (2.0 - sharpness) * f3 - (3.0 - sharpness) * f2 + 1.0;
    float2 w2 = -(2.0 - sharpness) * f3 + (3.0 - 2.0 * sharpness) * f2 + sharpness * f;
    float2 w3 = sharpness * f3 - sharpness * f2;

    float2 wl2 = w1 + w2;
    float2 tc2 = inv_resource_size * (center_pos + w2 * float2(positive_rcp(wl2.x), positive_rcp(wl2.y)));
    float2 tc0 = inv_resource_size * (center_pos - 1.0);
    float2 tc3 = inv_resource_size * (center_pos + 2.0);

    float w;
    float sum;

    w = wl2.x * w0.y;
    float4 color = tex.SampleLevel(samp, float2(tc2.x, tc0.y), 0) * w;
    sum = w;

    w = w0.x * wl2.y;
    color += tex.SampleLevel(samp, float2(tc0.x, tc2.y), 0) * w;
    sum += w;

    w = wl2.x * wl2.y;
    color += tex.SampleLevel(samp, float2(tc2.x, tc2.y), 0) * w;
    sum += w;

    w = w3.x * wl2.y;
    color += tex.SampleLevel(samp, float2(tc3.x, tc2.y), 0) * w;
    sum += w;

    w = wl2.x * w3.y;
    color += tex.SampleLevel(samp, float2(tc2.x, tc3.y), 0) * w;
    sum += w;

    color *= positive_rcp(sum);
    return color;
}

// ─── SMEM Preload ───────────────────────────────────────────────────────────
void preload(uint2 shared_pos, int2 global_pos)
{
    global_pos = clamp(global_pos, int2(0, 0), int2(g_rect_size) - 1);

    float3 c = g_input_color[global_pos].rgb;
    s_Color[shared_pos.y][shared_pos.x] = tonemap_reinhard(c);

    float4 mv = g_input_motion[global_pos];
    s_Mv[shared_pos.y][shared_pos.x] = float3(mv.xy, mv.w); // .xy = motion px, .z = viewZ
}

// ─── Main ───────────────────────────────────────────────────────────────────
[shader("compute")]
[numthreads(GROUP_X, GROUP_Y, 1)]
void compute_main(int2 thread_pos: SV_GroupThreadID,
                  int2 pixel_pos: SV_DispatchThreadID,
                  uint thread_index: SV_GroupIndex)
{
    float2 pixel_uv = float2(pixel_pos + 0.5) * g_inv_rect_size;

    // ── Preload into shared memory ──────────────────────────────────────
    int2 group_base = pixel_pos - thread_pos - BORDER;
    uint stage_num = (BUFFER_X * BUFFER_Y + GROUP_X * GROUP_Y - 1) / (GROUP_X * GROUP_Y);
    [unroll]
    for (uint stage = 0; stage < stage_num; stage++)
    {
        uint virtual_index = thread_index + stage * GROUP_X * GROUP_Y;
        uint2 new_id = uint2(virtual_index % BUFFER_X, virtual_index / BUFFER_X);
        if (stage == 0 || virtual_index < uint(BUFFER_X * BUFFER_Y))
            preload(new_id, group_base + int2(new_id));
    }
    GroupMemoryBarrierWithGroupSync();

    // Out-of-bounds guard
    if (pixel_uv.x > 1.0 || pixel_uv.y > 1.0)
        return;

    // ── Neighborhood statistics (3x3) ───────────────────────────────────
    float sum = 0;
    float3 m1 = float3(0);
    float3 m2 = float3(0);
    float3 input_color = float3(0);

    float3 center_mv = s_Mv[thread_pos.y + BORDER][thread_pos.x + BORDER];
    float min_view_z = abs(center_mv.z);
    int2 offset_closest = int2(BORDER, BORDER);

    [unroll]
    for (int dy = 0; dy <= BORDER * 2; dy++)
    {
        [unroll]
        for (int dx = 0; dx <= BORDER * 2; dx++)
        {
            int2 offset = int2(dx, dy);
            int2 smem_pos = thread_pos + offset;

            float3 c = s_Color[smem_pos.y][smem_pos.x];
            float3 mv = s_Mv[smem_pos.y][smem_pos.x];
            float view_z = abs(mv.z);

            if (dx == BORDER && dy == BORDER)
                input_color = c;
            else if (view_z < min_view_z)
            {
                min_view_z = view_z;
                offset_closest = offset;
            }

            // Gaussian-ish weight based on distance
            float2 d_offset = float2(offset) / float(BORDER) - 1.0;
            float r2 = dot(d_offset, d_offset);
            float w = exp(-r2);

            m1 += c * w;
            m2 += c * c * w;
            sum += w;
        }
    }

    m1 /= sum;
    m2 /= sum;

    float3 sigma = sqrt(abs(m2 - m1 * m1)) * g_sigma_scale;

    // ── Previous pixel position (closest-depth motion) ──────────────────
    float3 mv = s_Mv[thread_pos.y + offset_closest.y][thread_pos.x + offset_closest.x];
    float2 motion_uv = mv.xy * g_inv_rect_size;
    float2 pixel_uv_prev = pixel_uv + motion_uv;

    // ── History fetch (bicubic) ─────────────────────────────────────────
    float2 pixel_pos_prev = saturate(pixel_uv_prev) * g_rect_size_prev;
    float4 history = bicubic_filter_no_corners(g_history, g_linear_sampler,
                                                pixel_pos_prev, g_inv_render_size,
                                                g_history_sharpness);
    history.rgb = max(history.rgb, float3(0));

    // Tonemap history for clamping comparison
    float3 history_tonemapped = tonemap_reinhard(history.rgb);

    // ── Mix rate from history alpha ─────────────────────────────────────
    float mix_rate = saturate(history.w);
    mix_rate /= 1.0 + mix_rate; // exponential convergence

    // ── Disocclusion #0: uninitialized history ──────────────────────────
    // history.w == 0 means the texture was never written (fresh allocation).
    // Treat as a full reset to avoid blending with zeros.
    if (history.w <= 0.0)
        mix_rate = 1.0;

    // ── Disocclusion #1: off-screen ─────────────────────────────────────
    bool is_in_screen = all(saturate(pixel_uv_prev) == pixel_uv_prev);
    mix_rate = !is_in_screen ? 1.0 : mix_rate;

    // ── Disocclusion #2: AABB color-space distance (in tonemapped space) ─
    if (g_enable_clamp)
    {
        // Adaptive clamping: relax as history converges.
        // For noisy 1spp path tracing, aggressive clamping clips high-energy
        // samples causing systematic energy loss (darkening).  As history
        // stabilises (mix_rate → 0) we widen the AABB so clamping only fires
        // for true disocclusion, not noise-to-noise differences.
        float convergence_factor = 1.0 / max(mix_rate, g_min_alpha);
        float3 adaptive_sigma = sigma * min(convergence_factor, g_max_convergence);

        float3 clamped_history_tm = clamp_aabb(m1, adaptive_sigma, history_tonemapped);
        float diff = length(clamped_history_tm - history_tonemapped);
        mix_rate += diff;
    }

    // ── Clamp mix rate ──────────────────────────────────────────────────
    mix_rate = saturate(mix_rate);

    // ── Force reset ─────────────────────────────────────────────────────
    if (g_reset)
        mix_rate = 1.0;

    // ── Final blend in LINEAR space ─────────────────────────────────────
    // Blend with the *original* history (not clamped) when converged,
    // to preserve energy.  Use clamped history only for the disocclusion
    // part of the mix_rate which already bumped it toward 1.0 above.
    float3 history_linear = history.rgb; // already linear
    float3 input_color_linear = g_input_color[pixel_pos].rgb;

    float effective_alpha = max(mix_rate, g_min_alpha);
    float3 result = lerp(history_linear, input_color_linear, effective_alpha);

    // ── Output (already in linear HDR) ──────────────────────────────────
    g_output[pixel_pos] = float4(result, mix_rate);
}
