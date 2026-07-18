import mesh_colors;

StructuredBuffer<uint4> g_adjacency_infos;
RWStructuredBuffer<uint4> g_output;
uniform uint g_face_offset;

[shader("compute")]
[numthreads(3, 1, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint edge = dispatch_thread_id.x;
    MeshColorsInstanceInfo instance_info;
    instance_info.face_offset = g_face_offset;
    instance_info.face_count = 2;
    instance_info.texel_offset = 0;
    instance_info.texel_count = 0;
    MeshColorsEdgeAdjacency adjacency = mesh_colors_get_adjacency(
        g_adjacency_infos,
        instance_info,
        0,
        edge);
    g_output[edge] = uint4(
        adjacency.adjacent_face,
        adjacency.adjacent_edge,
        adjacency.flip,
        mesh_colors_adjacency_is_valid(adjacency) ? 1u : 0u);
}
