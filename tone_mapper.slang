// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Based on AMD FidelityFX LPM (Luma Preserving Mapper)
// https://github.com/GPUOpen-Effects/FidelityFX-LPM

static const float3 LPM_LUMA_709 = float3(0.2126, 0.7152, 0.0722);
static const float3 LPM_CROSSTALK = float3(1.0, 0.5, 0.0625);

float3 LpmToneMap(float3 color, float hdrMax, float exposure, float contrast)
{
    float midIn = hdrMax * 0.18 * exp2(-exposure);
    float midOut = 0.18;
    
    float cs = contrast;
    float z0 = -pow(midIn, contrast);
    float z1 = pow(hdrMax, cs) * pow(midIn, contrast);
    float z2 = pow(hdrMax, contrast) * pow(midIn, cs) * midOut;
    float z3 = pow(hdrMax, cs) * midOut;
    float z4 = pow(midIn, cs) * midOut;
    float toneScaleBiasX = -((z0 + (midOut * (z1 - z2)) / (z3 - z4)) / z4);
    
    float w0 = pow(hdrMax, cs) * pow(midIn, contrast);
    float w1 = pow(hdrMax, contrast) * pow(midIn, cs) * midOut;
    float w2 = pow(hdrMax, cs) * midOut;
    float w3 = pow(midIn, cs) * midOut;
    float toneScaleBiasY = (w0 - w1) / (w2 - w3);
    
    float maxC = max(max(color.r, color.g), color.b);
    if (maxC <= 0.0) return float3(0.0, 0.0, 0.0);
    
    float rcpMax = 1.0 / maxC;
    float3 ratio = color * rcpMax;
    
    float luma = dot(color, LPM_LUMA_709);
    luma = pow(max(luma, 0.0), contrast);
    luma = luma / (luma * toneScaleBiasX + toneScaleBiasY);
    
    float lumaRatio = dot(ratio, LPM_LUMA_709);
    float ratioScale = saturate(luma / max(lumaRatio, 0.00001));
    
    float3 outColor = saturate(ratio * ratioScale);
    
    float3 cap = LPM_CROSSTALK * (1.0 - outColor);
    float lumaAdd = saturate(luma - dot(outColor, LPM_LUMA_709));
    
    float capLuma = dot(cap, LPM_LUMA_709);
    if (capLuma > 0.00001)
    {
        float t = lumaAdd / capLuma;
        outColor = saturate(outColor + t * cap);
    }
    
    lumaAdd = saturate(luma - dot(outColor, LPM_LUMA_709));
    float3 rcpLuma = 1.0 / LPM_LUMA_709;
    outColor = saturate(outColor + lumaAdd * rcpLuma);
    
    return outColor;
}

// BT.709 OETF (ITU-R BT.709)
float3 LinearToRec709(float3 lin)
{
    float3 result;
    result.x = (lin.x < 0.018) ? (4.5 * lin.x) : (1.099 * pow(lin.x, 0.45) - 0.099);
    result.y = (lin.y < 0.018) ? (4.5 * lin.y) : (1.099 * pow(lin.y, 0.45) - 0.099);
    result.z = (lin.z < 0.018) ? (4.5 * lin.z) : (1.099 * pow(lin.z, 0.45) - 0.099);
    return result;
}

struct ToneMapper {
    Texture2D<float4> input;
    RWTexture2D<float4> output;
    float exposure;

    void execute(uint2 pixel)
    {
        float3 lin = input[pixel].xyz;
        float3 tonemapped = LpmToneMap(lin, 12.0, exposure, 1.0);
        float3 o = LinearToRec709(tonemapped);
        output[pixel] = float4(o, 1.0);
    }
}

ParameterBlock<ToneMapper> g_tone_mapper;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    g_tone_mapper.execute(tid.xy);
}
