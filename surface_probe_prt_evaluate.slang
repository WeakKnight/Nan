import common;
import scene;
import surface_probe_field;
import surface_probe_irradiance_field;
import surface_probe_prt_field;
import surface_probe_sh;

RWStructuredBuffer<float4> g_lighting_sh;
ByteAddressBuffer g_prt_field_values;
ByteAddressBuffer g_prt_sample_counts;
ByteAddressBuffer g_prt_static_source_rgb;
RWByteAddressBuffer g_evaluated_field_values;
RWByteAddressBuffer g_evaluated_sample_counts;
uniform uint g_probe_count;
uniform uint g_lighting_projection_sample_count;

uint surface_probe_prt_reverse_bits(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u)
        | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u)
        | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u)
        | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u)
        | ((bits & 0xFF00FF00u) >> 8u);
    return bits;
}

float surface_probe_prt_radical_inverse(uint index)
{
    return float(surface_probe_prt_reverse_bits(index))
        * 2.3283064365386963e-10;
}

float3 surface_probe_prt_uniform_sphere(uint index, uint sample_count)
{
    float u = (float(index) + 0.5) / float(sample_count);
    float v = surface_probe_prt_radical_inverse(index);
    float y = 1.0 - 2.0 * u;
    float radius = sqrt(max(1.0 - y * y, 0.0));
    float phi = TWO_PI * v;
    return float3(radius * cos(phi), y, radius * sin(phi));
}

[shader("compute")]
[numthreads(9, 1, 1)]
void project_lighting_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint coefficient = dispatch_thread_id.x;
    if (coefficient >= SURFACE_PROBE_PRT_COEFFICIENT_COUNT)
        return;
    uint sample_count = max(g_lighting_projection_sample_count, 1u);
    float3 projected = float3(0.0);
    for (uint sample_index = 0u;
         sample_index < sample_count;
         sample_index++)
    {
        float3 direction = surface_probe_prt_uniform_sphere(
            sample_index, sample_count);
        projected += g_scene.sample_sky_lut(direction)
            * surface_probe_sh_l2_basis(coefficient, direction);
    }
    projected *= 4.0 * PI / float(sample_count);
    g_lighting_sh[coefficient] = float4(projected, 0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void evaluate_field_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint probe_index = dispatch_thread_id.x;
    if (probe_index >= g_probe_count)
        return;
    uint sample_count = surface_probe_load_sample_count(
        g_prt_sample_counts, probe_index);
    float3 irradiance = surface_probe_load_static_source(
        g_prt_static_source_rgb, probe_index);
    if (sample_count > 0u)
    {
        for (uint coefficient = 0u;
             coefficient < SURFACE_PROBE_PRT_COEFFICIENT_COUNT;
             coefficient++)
        {
            irradiance += surface_probe_load_prt_coefficient(
                g_prt_field_values, probe_index, coefficient)
                * g_lighting_sh[coefficient].xyz;
        }
    }
    irradiance = max(irradiance, float3(0.0));
    uint destination = probe_index
        * SURFACE_PROBE_IRRADIANCE_VALUE_STRIDE;
    g_evaluated_field_values.Store(
        destination, asuint(irradiance.x));
    g_evaluated_field_values.Store(
        destination + 4u, asuint(irradiance.y));
    g_evaluated_field_values.Store(
        destination + 8u, asuint(irradiance.z));
    surface_probe_store_sample_count(
        g_evaluated_sample_counts, probe_index, sample_count);
}
