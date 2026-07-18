import mesh_colors;

StructuredBuffer<uint4> g_cases;
RWStructuredBuffer<float4> g_output;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint case_index = dispatch_thread_id.x;
    uint4 packed_case = g_cases[case_index];
    MeshColorsEdgeAdjacency adjacency;
    adjacency.adjacent_face = 0;
    adjacency.adjacent_edge =
        packed_case.w & MESH_COLORS_ADJACENCY_EDGE_MASK;
    adjacency.flip =
        (packed_case.w & MESH_COLORS_ADJACENCY_FLIP_BIT) != 0 ? 1u : 0u;
    uint crossed_edge = (packed_case.w >> 8) & 0x3;

    float3 barycentrics;
    bool valid = mesh_colors_map_one_edge_tap(
        int2(asint(packed_case.x), asint(packed_case.y)),
        packed_case.z,
        adjacency,
        crossed_edge,
        barycentrics);
    g_output[case_index] = float4(barycentrics, valid ? 1.0 : 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void normal_weight_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    float sigma = 3.14159265359 / 6.0;
    g_output[0] = float4(
        mesh_colors_normal_filter_weight(
            float3(0.0, 0.0, 1.0),
            float3(0.0, 0.0, 1.0),
            sigma),
        mesh_colors_normal_filter_weight(
            float3(0.0, 0.0, 1.0),
            float3(0.0, 1.0, 0.0),
            sigma),
        mesh_colors_normal_filter_weight(
            float3(0.0, 0.0, 1.0),
            normalize(float3(0.0, 1.0, 1.0)),
            sigma),
        0.0);
}
