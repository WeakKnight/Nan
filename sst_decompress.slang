RWTexture2D<float> g_output;
StructuredBuffer<uint> g_sst_compact_words;
StructuredBuffer<uint> g_sst_compact_roots;

uniform uint g_resolution;
uniform uint2 g_tile_grid;
uniform uint g_tile_size;
uniform uint g_max_traversal_steps;
uniform uint g_compact_word_count;
uniform uint g_branch_10bit_start_level;

float decode_unorm_bits(uint value, uint max_value)
{
    return float(value) / float(max_value);
}

float decode_snorm_bits(uint value, uint max_value, float value_range)
{
    return ((float(value) / float(max_value)) * 2.0 - 1.0) * value_range;
}

bool is_power_of_two_uint(uint value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

uint log2_power_of_two_uint(uint value)
{
    uint bits = 0;
    [loop]
    for (uint i = 0; i < 16; i++)
    {
        if (value <= 1)
            break;
        value >>= 1;
        bits++;
    }
    return bits;
}

uint decode_packed_branch_offset(uint word0, uint word1, uint level, uint quadrant)
{
    if (level >= g_branch_10bit_start_level)
    {
        if (quadrant == 0)
            return (word0 >> 2) & 0x3ff;
        if (quadrant == 1)
            return (word0 >> 12) & 0x3ff;
        if (quadrant == 2)
            return (word0 >> 22) & 0x3ff;
        return word1 & 0x3ff;
    }

    if (quadrant == 0)
        return (word0 >> 2) & 0x1fff;
    if (quadrant == 1)
        return (word0 >> 15) & 0x1fff;
    if (quadrant == 2)
        return ((word0 >> 28) & 0xf) | ((word1 & 0x1ff) << 4);
    return (word1 >> 9) & 0x1fff;
}

uint decode_tile_morton_quadrant(
    uint pixel_x,
    uint pixel_y,
    uint tile_origin_x,
    uint tile_origin_y,
    uint tile_extent_x,
    uint tile_extent_y,
    float local_x,
    float local_y,
    uint tree_bits,
    uint level)
{
    uint fallback_quadrant = (local_x >= 0.5 ? 1 : 0) + (local_y >= 0.5 ? 2 : 0);
    if (tree_bits == 0 || tile_extent_x != g_tile_size || tile_extent_y != g_tile_size)
        return fallback_quadrant;

    if (level >= tree_bits)
        return fallback_quadrant;

    uint bit_index = tree_bits - 1 - level;
    uint local_pixel_x = min(pixel_x - tile_origin_x, g_tile_size - 1);
    uint local_pixel_y = min(pixel_y - tile_origin_y, g_tile_size - 1);
    return ((local_pixel_x >> bit_index) & 1) | (((local_pixel_y >> bit_index) & 1) << 1);
}

float decode_compact_sst_depth(uint2 pixel)
{
    if (g_compact_word_count == 0 || g_tile_size == 0 || g_tile_grid.x == 0 || g_tile_grid.y == 0)
        return 1.0;

    uint resolution = max(g_resolution, 1);
    uint pixel_x = min(pixel.x, resolution - 1);
    uint pixel_y = min(pixel.y, resolution - 1);
    uint tile_x = min(pixel_x / g_tile_size, g_tile_grid.x - 1);
    uint tile_y = min(pixel_y / g_tile_size, g_tile_grid.y - 1);
    uint node_word_offset = g_sst_compact_roots[tile_y * g_tile_grid.x + tile_x];
    if (node_word_offset >= g_compact_word_count)
        return 1.0;

    uint tile_origin_x = tile_x * g_tile_size;
    uint tile_origin_y = tile_y * g_tile_size;
    uint tile_extent_x = max(min(g_tile_size, resolution - tile_origin_x), 1);
    uint tile_extent_y = max(min(g_tile_size, resolution - tile_origin_y), 1);
    float local_x = ((float(pixel_x) + 0.5) - float(tile_origin_x)) / float(tile_extent_x);
    float local_y = ((float(pixel_y) + 0.5) - float(tile_origin_y)) / float(tile_extent_y);
    uint tree_bits = is_power_of_two_uint(g_tile_size) ? log2_power_of_two_uint(g_tile_size) : 0;

    uint max_steps = min(max(g_max_traversal_steps, 1), 32);
    [loop]
    for (uint i = 0; i < max_steps; i++)
    {
        if (node_word_offset >= g_compact_word_count)
            return 1.0;

        uint word0 = g_sst_compact_words[node_word_offset];
        uint control = word0 & 0x3;
        if (control == 1)
            return decode_unorm_bits((word0 >> 2) & 0x3fffffff, 0x3fffffff);

        if (node_word_offset + 1 >= g_compact_word_count)
            return 1.0;

        uint word1 = g_sst_compact_words[node_word_offset + 1];
        if (control == 2)
        {
            uint qz = (word0 >> 2) & 0x3fffffff;
            uint qx = word1 & 0xffff;
            uint qy = (word1 >> 16) & 0xffff;
            float a = decode_snorm_bits(qx, 0xffff, 2.0);
            float b = decode_snorm_bits(qy, 0xffff, 2.0);
            float c = decode_unorm_bits(qz, 0x3fffffff);
            return saturate(a * (local_x - 0.5) + b * (local_y - 0.5) + c);
        }

        uint quadrant = decode_tile_morton_quadrant(
            pixel_x,
            pixel_y,
            tile_origin_x,
            tile_origin_y,
            tile_extent_x,
            tile_extent_y,
            local_x,
            local_y,
            tree_bits,
            i
        );
        uint relative_offset = decode_packed_branch_offset(word0, word1, i, quadrant);
        if (relative_offset == 0)
            return 1.0;

        node_word_offset += relative_offset;
        local_x = local_x * 2.0 - ((quadrant & 1) != 0 ? 1.0 : 0.0);
        local_y = local_y * 2.0 - ((quadrant & 2) != 0 ? 1.0 : 0.0);
    }

    return 1.0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_output.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    g_output[pixel] = decode_compact_sst_depth(pixel);
}
