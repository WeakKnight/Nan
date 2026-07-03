import common;
import scene;

RWTexture2D<float> g_shadow_mask;

uniform uint2 g_screen_size;
uniform uint2 g_screen_grid_dim;
uniform uint2 g_super_grid_dim;
uniform float g_adaptive_threshold;
uniform uint g_adaptive_bootstrap_passes;

static const uint WAVE_SIZE = 32;

static const uint2 kBlockOffsets[4] = {
    uint2(0, 0), uint2(4, 0), uint2(0, 4), uint2(4, 4)
};

static const uint2 kPass2Pixels[8] = {
    uint2(0, 2), uint2(2, 0), uint2(4, 2), uint2(6, 0),
    uint2(0, 6), uint2(2, 4), uint2(4, 6), uint2(6, 4)
};

static const uint2 kPass3Pixels[4] = {
    uint2(1, 1), uint2(1, 3), uint2(3, 1), uint2(3, 3)
};

static const uint2 kPass4Pixels[8] = {
    uint2(0, 1), uint2(1, 0), uint2(1, 2), uint2(2, 1),
    uint2(2, 3), uint2(3, 2), uint2(0, 3), uint2(3, 0)
};

static const int2 kPass1NeighborOffsets[4] = {
    int2(2, 2), int2(-2, 2), int2(2, -2), int2(-2, -2)
};

static const int2 kPass2NeighborOffsets[4] = {
    int2(0, 2), int2(2, 0), int2(0, -2), int2(-2, 0)
};

static const int2 kPass3NeighborOffsets[4] = {
    int2(1, 1), int2(-1, -1), int2(1, -1), int2(-1, 1)
};

static const int2 kPass4NeighborOffsets[4] = {
    int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)
};

uint2 linear_to_grid(uint linear, uint2 grid_dim)
{
    if (grid_dim.x == 0)
        return uint2(0, 0);
    return uint2(linear % grid_dim.x, linear / grid_dim.x);
}

bool pixel_in_bounds(uint2 pixel)
{
    return all(pixel < g_screen_size);
}

bool neighbor_in_bounds(int2 pixel)
{
    return pixel.x >= 0 && pixel.y >= 0 &&
        pixel.x < int(g_screen_size.x) && pixel.y < int(g_screen_size.y);
}

uint pack_base(uint2 base)
{
    return (base.x & 0xffffu) | ((base.y & 0xffffu) << 16);
}

uint2 unpack_base(uint packed_base)
{
    return uint2(packed_base & 0xffffu, packed_base >> 16);
}

uint nth_set_bit(uint mask, uint n)
{
    [unroll]
    for (uint i = 0; i < 32; i++)
    {
        if ((mask & (1u << i)) != 0)
        {
            if (n == 0)
                return i;
            n--;
        }
    }
    return 0;
}

float shade_shadow_pixel(uint2 pixel)
{
    if (!pixel_in_bounds(pixel))
        return 1.0;

    Ray ray = g_scene.camera.compute_ray_pinhole(pixel, g_screen_size);
    HitInfo hit = g_scene.intersect(ray);
    if (!hit.valid)
        return 1.0;

    ShadingData shading_data = g_scene.get_shading_data(hit, -ray.dir);
    return saturate(g_scene.sample_static_sun_visibility(shading_data, g_scene.directional_light.direction));
}

bool read_neighbor(uint2 pixel, int2 offset, out float value)
{
    int2 neighbor = int2(pixel) + offset;
    if (!neighbor_in_bounds(neighbor))
    {
        value = 1.0;
        return false;
    }

    value = g_shadow_mask[uint2(neighbor)];
    return true;
}

bool should_shade_from_neighbors(
    uint2 pixel,
    int2 offset0,
    int2 offset1,
    int2 offset2,
    int2 offset3,
    out float interpolated)
{
    float v0;
    float v1;
    float v2;
    float v3;
    bool valid0 = read_neighbor(pixel, offset0, v0);
    bool valid1 = read_neighbor(pixel, offset1, v1);
    bool valid2 = read_neighbor(pixel, offset2, v2);
    bool valid3 = read_neighbor(pixel, offset3, v3);
    bool valid = valid0 && valid1 && valid2 && valid3;

    interpolated = 0.25 * (v0 + v1 + v2 + v3);
    if (!valid)
        return true;

    float max_value = max(max(v0, v1), max(v2, v3));
    float min_value = min(min(v0, v1), min(v2, v3));
    return (max_value - min_value) > g_adaptive_threshold;
}

void shade_or_interpolate(
    uint2 pixel,
    uint pass_index,
    int2 offset0,
    int2 offset1,
    int2 offset2,
    int2 offset3)
{
    if (!pixel_in_bounds(pixel))
        return;

    float interpolated = 1.0;
    bool needs_shade = pass_index <= g_adaptive_bootstrap_passes ||
        should_shade_from_neighbors(pixel, offset0, offset1, offset2, offset3, interpolated);

    g_shadow_mask[pixel] = needs_shade ? shade_shadow_pixel(pixel) : interpolated;
}

bool collect_candidate(
    uint2 pixel,
    uint pass_index,
    int2 offset0,
    int2 offset1,
    int2 offset2,
    int2 offset3,
    inout uint shade_mask,
    inout uint shade_count,
    uint bit_index)
{
    if (!pixel_in_bounds(pixel))
        return false;

    float interpolated = 1.0;
    bool needs_shade = pass_index <= g_adaptive_bootstrap_passes ||
        should_shade_from_neighbors(pixel, offset0, offset1, offset2, offset3, interpolated);
    if (needs_shade)
    {
        shade_mask |= 1u << bit_index;
        shade_count++;
    }
    else
    {
        g_shadow_mask[pixel] = interpolated;
    }

    return needs_shade;
}

struct WaveWorkItem
{
    bool active;
    uint source_lane;
    uint local_item_index;
};

WaveWorkItem map_wave_work_item(
    uint item_index,
    uint total_items,
    uint first_work_item,
    uint num_work_items,
    uint lane_count)
{
    WaveWorkItem result;
    result.active = item_index < total_items;
    result.source_lane = 0;
    result.local_item_index = 0;

    [loop]
    for (uint lane = 0; lane < lane_count; lane++)
    {
        uint source_first = WaveReadLaneAt(first_work_item, lane);
        uint source_count = WaveReadLaneAt(num_work_items, lane);
        bool owns_item = result.active &&
            item_index >= source_first &&
            item_index < source_first + source_count;
        if (owns_item)
        {
            result.source_lane = lane;
            result.local_item_index = item_index - source_first;
        }
    }

    return result;
}

void distribute_pass1(uint lane_index, uint producer_base, uint producer_mask, uint num_work_items)
{
    uint lane_count = min(WaveGetLaneCount(), WAVE_SIZE);
    uint first_work_item = WavePrefixSum(num_work_items);
    uint total_items = WaveActiveSum(num_work_items);

    for (uint batch_first = 0; batch_first < total_items; batch_first += lane_count)
    {
        WaveWorkItem item = map_wave_work_item(batch_first + lane_index, total_items, first_work_item, num_work_items, lane_count);
        uint source_base = WaveReadLaneAt(producer_base, item.source_lane);
        uint source_mask = WaveReadLaneAt(producer_mask, item.source_lane);
        uint bit_index = nth_set_bit(source_mask, item.local_item_index);
        uint2 pixel = unpack_base(source_base) + kBlockOffsets[bit_index] + uint2(2, 2);
        if (item.active)
            g_shadow_mask[pixel] = shade_shadow_pixel(pixel);
    }
}

void distribute_pass2(uint lane_index, uint producer_base, uint producer_mask, uint num_work_items)
{
    uint lane_count = min(WaveGetLaneCount(), WAVE_SIZE);
    uint first_work_item = WavePrefixSum(num_work_items);
    uint total_items = WaveActiveSum(num_work_items);

    for (uint batch_first = 0; batch_first < total_items; batch_first += lane_count)
    {
        WaveWorkItem item = map_wave_work_item(batch_first + lane_index, total_items, first_work_item, num_work_items, lane_count);
        uint source_base = WaveReadLaneAt(producer_base, item.source_lane);
        uint source_mask = WaveReadLaneAt(producer_mask, item.source_lane);
        uint bit_index = nth_set_bit(source_mask, item.local_item_index);
        uint2 pixel = unpack_base(source_base) + kPass2Pixels[bit_index];
        if (item.active)
            g_shadow_mask[pixel] = shade_shadow_pixel(pixel);
    }
}

void distribute_pass3(uint lane_index, uint producer_base, uint producer_mask, uint num_work_items)
{
    uint lane_count = min(WaveGetLaneCount(), WAVE_SIZE);
    uint first_work_item = WavePrefixSum(num_work_items);
    uint total_items = WaveActiveSum(num_work_items);

    for (uint batch_first = 0; batch_first < total_items; batch_first += lane_count)
    {
        WaveWorkItem item = map_wave_work_item(batch_first + lane_index, total_items, first_work_item, num_work_items, lane_count);
        uint source_base = WaveReadLaneAt(producer_base, item.source_lane);
        uint source_mask = WaveReadLaneAt(producer_mask, item.source_lane);
        uint bit_index = nth_set_bit(source_mask, item.local_item_index);
        uint2 pixel = unpack_base(source_base) + kPass3Pixels[bit_index];
        if (item.active)
            g_shadow_mask[pixel] = shade_shadow_pixel(pixel);
    }
}

void distribute_pass4(uint lane_index, uint producer_base, uint producer_mask, uint num_work_items)
{
    uint lane_count = min(WaveGetLaneCount(), WAVE_SIZE);
    uint first_work_item = WavePrefixSum(num_work_items);
    uint total_items = WaveActiveSum(num_work_items);

    for (uint batch_first = 0; batch_first < total_items; batch_first += lane_count)
    {
        WaveWorkItem item = map_wave_work_item(batch_first + lane_index, total_items, first_work_item, num_work_items, lane_count);
        uint source_base = WaveReadLaneAt(producer_base, item.source_lane);
        uint source_mask = WaveReadLaneAt(producer_mask, item.source_lane);
        uint bit_index = nth_set_bit(source_mask, item.local_item_index);
        uint2 pixel = unpack_base(source_base) + kPass4Pixels[bit_index];
        if (item.active)
            g_shadow_mask[pixel] = shade_shadow_pixel(pixel);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void full_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (!pixel_in_bounds(pixel))
        return;

    g_shadow_mask[pixel] = shade_shadow_pixel(pixel);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pass0(uint3 tid: SV_DispatchThreadID)
{
    uint2 block_base = tid.xy * 4;
    if (!pixel_in_bounds(block_base))
        return;

    g_shadow_mask[block_base] = shade_shadow_pixel(block_base);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pass1(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= g_super_grid_dim))
        return;

    uint2 super_base = tid.xy * 8;
    [unroll]
    for (uint i = 0; i < 4; i++)
    {
        shade_or_interpolate(
            super_base + kBlockOffsets[i] + uint2(2, 2),
            1,
            kPass1NeighborOffsets[0],
            kPass1NeighborOffsets[1],
            kPass1NeighborOffsets[2],
            kPass1NeighborOffsets[3]);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pass2(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= g_super_grid_dim))
        return;

    uint2 super_base = tid.xy * 8;
    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        shade_or_interpolate(
            super_base + kPass2Pixels[i],
            2,
            kPass2NeighborOffsets[0],
            kPass2NeighborOffsets[1],
            kPass2NeighborOffsets[2],
            kPass2NeighborOffsets[3]);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pass3(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= g_screen_grid_dim))
        return;

    uint2 block_base = tid.xy * 4;
    [unroll]
    for (uint i = 0; i < 4; i++)
    {
        shade_or_interpolate(
            block_base + kPass3Pixels[i],
            3,
            kPass3NeighborOffsets[0],
            kPass3NeighborOffsets[1],
            kPass3NeighborOffsets[2],
            kPass3NeighborOffsets[3]);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pass4(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= g_screen_grid_dim))
        return;

    uint2 block_base = tid.xy * 4;
    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        shade_or_interpolate(
            block_base + kPass4Pixels[i],
            4,
            kPass4NeighborOffsets[0],
            kPass4NeighborOffsets[1],
            kPass4NeighborOffsets[2],
            kPass4NeighborOffsets[3]);
    }
}

[shader("compute")]
[numthreads(WAVE_SIZE, 1, 1)]
void wave_pass1(uint tid: SV_DispatchThreadID, uint gtid: SV_GroupThreadID)
{
    uint lane_index = WaveGetLaneIndex();
    uint2 super_id = linear_to_grid(tid, g_super_grid_dim);
    uint2 super_base = super_id * 8;
    uint shade_mask = 0;
    uint shade_count = 0;

    if (tid < g_super_grid_dim.x * g_super_grid_dim.y)
    {
        [unroll]
        for (uint i = 0; i < 4; i++)
        {
            collect_candidate(
                super_base + kBlockOffsets[i] + uint2(2, 2),
                1,
                kPass1NeighborOffsets[0],
                kPass1NeighborOffsets[1],
                kPass1NeighborOffsets[2],
                kPass1NeighborOffsets[3],
                shade_mask,
                shade_count,
                i);
        }
    }

    distribute_pass1(lane_index, pack_base(super_base), shade_mask, shade_count);
}

[shader("compute")]
[numthreads(WAVE_SIZE, 1, 1)]
void wave_pass2(uint tid: SV_DispatchThreadID, uint gtid: SV_GroupThreadID)
{
    uint lane_index = WaveGetLaneIndex();
    uint2 super_id = linear_to_grid(tid, g_super_grid_dim);
    uint2 super_base = super_id * 8;
    uint shade_mask = 0;
    uint shade_count = 0;

    if (tid < g_super_grid_dim.x * g_super_grid_dim.y)
    {
        [unroll]
        for (uint i = 0; i < 8; i++)
        {
            collect_candidate(
                super_base + kPass2Pixels[i],
                2,
                kPass2NeighborOffsets[0],
                kPass2NeighborOffsets[1],
                kPass2NeighborOffsets[2],
                kPass2NeighborOffsets[3],
                shade_mask,
                shade_count,
                i);
        }
    }

    distribute_pass2(lane_index, pack_base(super_base), shade_mask, shade_count);
}

[shader("compute")]
[numthreads(WAVE_SIZE, 1, 1)]
void wave_pass3(uint tid: SV_DispatchThreadID, uint gtid: SV_GroupThreadID)
{
    uint lane_index = WaveGetLaneIndex();
    uint2 block_id = linear_to_grid(tid, g_screen_grid_dim);
    uint2 block_base = block_id * 4;
    uint shade_mask = 0;
    uint shade_count = 0;

    if (tid < g_screen_grid_dim.x * g_screen_grid_dim.y)
    {
        [unroll]
        for (uint i = 0; i < 4; i++)
        {
            collect_candidate(
                block_base + kPass3Pixels[i],
                3,
                kPass3NeighborOffsets[0],
                kPass3NeighborOffsets[1],
                kPass3NeighborOffsets[2],
                kPass3NeighborOffsets[3],
                shade_mask,
                shade_count,
                i);
        }
    }

    distribute_pass3(lane_index, pack_base(block_base), shade_mask, shade_count);
}

[shader("compute")]
[numthreads(WAVE_SIZE, 1, 1)]
void wave_pass4(uint tid: SV_DispatchThreadID, uint gtid: SV_GroupThreadID)
{
    uint lane_index = WaveGetLaneIndex();
    uint2 block_id = linear_to_grid(tid, g_screen_grid_dim);
    uint2 block_base = block_id * 4;
    uint shade_mask = 0;
    uint shade_count = 0;

    if (tid < g_screen_grid_dim.x * g_screen_grid_dim.y)
    {
        [unroll]
        for (uint i = 0; i < 8; i++)
        {
            collect_candidate(
                block_base + kPass4Pixels[i],
                4,
                kPass4NeighborOffsets[0],
                kPass4NeighborOffsets[1],
                kPass4NeighborOffsets[2],
                kPass4NeighborOffsets[3],
                shade_mask,
                shade_count,
                i);
        }
    }

    distribute_pass4(lane_index, pack_base(block_base), shade_mask, shade_count);
}
