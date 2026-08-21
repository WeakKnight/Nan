import surface_probe_field;
import surface_probe_sh;

static const uint SURFACE_PROBE_PRT_COEFFICIENT_COUNT =
    SURFACE_PROBE_SH_L2_COEFFICIENT_COUNT;
static const uint SURFACE_PROBE_PRT_COEFFICIENT_STRIDE = 12u;
static const uint SURFACE_PROBE_PRT_VALUE_STRIDE =
    SURFACE_PROBE_PRT_COEFFICIENT_COUNT
    * SURFACE_PROBE_PRT_COEFFICIENT_STRIDE;
static const uint SURFACE_PROBE_STATIC_SOURCE_STRIDE = 12u;

float3 surface_probe_load_prt_coefficient(
    ByteAddressBuffer buffer,
    uint probe_index,
    uint coefficient)
{
    uint offset = probe_index * SURFACE_PROBE_PRT_VALUE_STRIDE
        + coefficient * SURFACE_PROBE_PRT_COEFFICIENT_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4u),
        buffer.Load(offset + 8u)));
}

float3 surface_probe_load_prt_coefficient(
    RWByteAddressBuffer buffer,
    uint probe_index,
    uint coefficient)
{
    uint offset = probe_index * SURFACE_PROBE_PRT_VALUE_STRIDE
        + coefficient * SURFACE_PROBE_PRT_COEFFICIENT_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4u),
        buffer.Load(offset + 8u)));
}

void surface_probe_store_prt_coefficient(
    RWByteAddressBuffer buffer,
    uint probe_index,
    uint coefficient,
    float3 value)
{
    uint offset = probe_index * SURFACE_PROBE_PRT_VALUE_STRIDE
        + coefficient * SURFACE_PROBE_PRT_COEFFICIENT_STRIDE;
    buffer.Store(offset, asuint(value.x));
    buffer.Store(offset + 4u, asuint(value.y));
    buffer.Store(offset + 8u, asuint(value.z));
}

float3 surface_probe_load_static_source(
    ByteAddressBuffer buffer,
    uint probe_index)
{
    uint offset = probe_index * SURFACE_PROBE_STATIC_SOURCE_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4u),
        buffer.Load(offset + 8u)));
}

float3 surface_probe_load_static_source(
    RWByteAddressBuffer buffer,
    uint probe_index)
{
    uint offset = probe_index * SURFACE_PROBE_STATIC_SOURCE_STRIDE;
    return asfloat(uint3(
        buffer.Load(offset),
        buffer.Load(offset + 4u),
        buffer.Load(offset + 8u)));
}

void surface_probe_store_static_source(
    RWByteAddressBuffer buffer,
    uint probe_index,
    float3 value)
{
    uint offset = probe_index * SURFACE_PROBE_STATIC_SOURCE_STRIDE;
    buffer.Store(offset, asuint(value.x));
    buffer.Store(offset + 4u, asuint(value.y));
    buffer.Store(offset + 8u, asuint(value.z));
}

