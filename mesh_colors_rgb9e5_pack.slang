import mesh_colors;

ByteAddressBuffer g_source;
RWStructuredBuffer<uint> g_destination;
uniform uint g_texel_count;

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint texel_index = dispatch_thread_id.x;
    if (texel_index >= g_texel_count)
        return;
    g_destination[texel_index] = mesh_colors_pack_rgb9e5(
        mesh_colors_load_irradiance(g_source, texel_index));
}
