import surface_probes;
import surface_probe_field;

static const uint SURFACE_PROBE_IRRADIANCE_VALUE_STRIDE = 12;

float3 surface_probe_load_irradiance(
    ByteAddressBuffer buffer,
    uint probe_index)
{
    uint offset = probe_index * SURFACE_PROBE_IRRADIANCE_VALUE_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4),
        buffer.Load(offset + 8)));
}

float3 surface_probe_load_irradiance(
    RWByteAddressBuffer buffer,
    uint probe_index)
{
    uint offset = probe_index * SURFACE_PROBE_IRRADIANCE_VALUE_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4),
        buffer.Load(offset + 8)));
}

float3 surface_probe_reconstruct_irradiance(
    ByteAddressBuffer field_values,
    ByteAddressBuffer sample_counts,
    StructuredBuffer<SurfaceProbe> probes,
    uint gathered_count,
    uint gathered_indices[SURFACE_PROBE_MAX_GATHER],
    float gathered_weights[SURFACE_PROBE_MAX_GATHER],
    out float support_f,
    out float reconstruction_weight_sum,
    out float reconstruction_ess)
{
    float3 accumulated = float3(0.0);
    float weight_sum = 0.0;
    float weight_square_sum = 0.0;
    float support_sum = 0.0;
    float support_weight_sum = 0.0;
    float best_weight = -1.0;
    float3 best_irradiance = float3(0.0);
    for (uint i = 0; i < gathered_count; i++)
    {
        SurfaceProbe probe = surface_probe_load(
            probes, gathered_indices[i]);
        support_sum += gathered_weights[i] * surface_probe_support_f(probe);
        support_weight_sum += gathered_weights[i];
        float3 probe_irradiance = surface_probe_load_irradiance(
            field_values, gathered_indices[i]);
        uint sample_count = surface_probe_load_sample_count(
            sample_counts, gathered_indices[i]);
        float confidence = sample_count > 0 ? 1.0 : 0.0;
        float weight = gathered_weights[i] * confidence;
        accumulated += weight * probe_irradiance;
        weight_sum += weight;
        weight_square_sum += weight * weight;
        if (weight > best_weight)
        {
            best_weight = weight;
            best_irradiance = probe_irradiance;
        }
    }

    support_f = support_weight_sum > 1e-6
        ? saturate(support_sum / support_weight_sum)
        : 0.0;
    reconstruction_weight_sum = weight_sum;
    reconstruction_ess = weight_square_sum > 1e-12
        ? weight_sum * weight_sum / weight_square_sum
        : 0.0;

    if (weight_sum > 1e-6)
        return accumulated / weight_sum;
    if (best_weight >= 0.0)
        return best_irradiance;
    return float3(0.0);
}
