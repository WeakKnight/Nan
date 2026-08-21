static const uint SURFACE_PROBE_SAMPLE_COUNT_STRIDE = 4;

uint surface_probe_load_sample_count(
    ByteAddressBuffer buffer,
    uint probe_index)
{
    return buffer.Load(probe_index * SURFACE_PROBE_SAMPLE_COUNT_STRIDE);
}

uint surface_probe_load_sample_count(
    RWByteAddressBuffer buffer,
    uint probe_index)
{
    return buffer.Load(probe_index * SURFACE_PROBE_SAMPLE_COUNT_STRIDE);
}

void surface_probe_store_sample_count(
    RWByteAddressBuffer buffer,
    uint probe_index,
    uint sample_count)
{
    buffer.Store(
        probe_index * SURFACE_PROBE_SAMPLE_COUNT_STRIDE,
        sample_count);
}
