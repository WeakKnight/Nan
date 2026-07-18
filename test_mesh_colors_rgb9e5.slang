import mesh_colors;

StructuredBuffer<uint> g_input;
RWStructuredBuffer<float4> g_output;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint index = dispatch_thread_id.x;
    uint count;
    uint stride;
    g_input.GetDimensions(count, stride);
    if (index >= count)
        return;
    g_output[index] = float4(
        mesh_colors_unpack_rgb9e5(g_input[index]), 1.0);
}
