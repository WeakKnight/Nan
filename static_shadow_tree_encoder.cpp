#include "static_shadow_tree_encoder.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
constexpr uint32_t kNodeFlagPlane = 1u << 0;
constexpr uint32_t kNodeFlagConstant = 1u << 1;
constexpr uint32_t kInvalidRoot = 0xffffffffu;

struct Node
{
    uint32_t flags = 0;
    uint32_t child_base = 0;
    uint32_t meta = 0;
    uint32_t reserved = 0;
    float a = 0.0f;
    float b = 0.0f;
    float c = 1.0f;
    float max_error = 0.0f;
};

static_assert(sizeof(Node) == 32, "SST node ABI must match Python struct IIIIffff");

struct FitResult
{
    uint32_t flags = kNodeFlagPlane;
    float a = 0.0f;
    float b = 0.0f;
    float c = 1.0f;
    float max_error = 0.0f;
};

struct TileResult
{
    std::vector<Node> nodes;
    std::vector<uint32_t> levels;
    uint32_t root = kInvalidRoot;
    uint32_t forced_leaf_node_count = 0;
    uint64_t forced_leaf_pixel_count = 0;
    double forced_leaf_error_sum = 0.0;
    float forced_leaf_max_error = 0.0f;
};

struct PlaneEvalResult
{
    float a = 0.0f;
    float b = 0.0f;
    float c = 1.0f;
    float max_error = 0.0f;
};

struct RegionStats
{
    uint32_t x0 = 0;
    uint32_t y0 = 0;
    uint32_t x1 = 0;
    uint32_t y1 = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    double count = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_yy = 0.0;
    double sum_xy = 0.0;
    double sum_z = 0.0;
    double sum_xz = 0.0;
    double sum_yz = 0.0;
    float lower_min = std::numeric_limits<float>::infinity();
    float lower_max = -std::numeric_limits<float>::infinity();
    float upper_min = std::numeric_limits<float>::infinity();
    float leaky_upper_min = std::numeric_limits<float>::infinity();
};

uint32_t ceil_div(uint32_t a, uint32_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

uint32_t quantize_unorm(float value, uint32_t bits)
{
    const uint32_t max_value = (1u << bits) - 1u;
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<uint32_t>(std::lround(static_cast<double>(clamped) * static_cast<double>(max_value)));
}

uint32_t quantize_snorm(float value, uint32_t bits, float value_range)
{
    const uint32_t max_value = (1u << bits) - 1u;
    const float normalized = std::clamp(value / value_range, -1.0f, 1.0f) * 0.5f + 0.5f;
    return static_cast<uint32_t>(std::lround(static_cast<double>(normalized) * static_cast<double>(max_value)));
}

float decode_unorm(uint32_t value, uint32_t max_value)
{
    return static_cast<float>(static_cast<double>(value) / static_cast<double>(max_value));
}

float decode_snorm(uint32_t value, uint32_t max_value, float value_range)
{
    return static_cast<float>(((static_cast<double>(value) / static_cast<double>(max_value)) * 2.0 - 1.0) * value_range);
}

bool solve_2x2(double a00, double a01, double a11, double b0, double b1, double& x0, double& x1)
{
    const double det = a00 * a11 - a01 * a01;
    if (std::abs(det) < 1e-20)
        return false;
    x0 = (b0 * a11 - b1 * a01) / det;
    x1 = (a00 * b1 - a01 * b0) / det;
    return true;
}

bool solve_3x3(double matrix[3][3], double rhs[3], double out[3])
{
    double a[3][4] = {
        {matrix[0][0], matrix[0][1], matrix[0][2], rhs[0]},
        {matrix[1][0], matrix[1][1], matrix[1][2], rhs[1]},
        {matrix[2][0], matrix[2][1], matrix[2][2], rhs[2]},
    };

    for (int col = 0; col < 3; ++col)
    {
        int pivot = col;
        double best = std::abs(a[col][col]);
        for (int row = col + 1; row < 3; ++row)
        {
            const double candidate = std::abs(a[row][col]);
            if (candidate > best)
            {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1e-20)
            return false;
        if (pivot != col)
        {
            for (int k = col; k < 4; ++k)
                std::swap(a[col][k], a[pivot][k]);
        }

        const double inv = 1.0 / a[col][col];
        for (int k = col; k < 4; ++k)
            a[col][k] *= inv;

        for (int row = 0; row < 3; ++row)
        {
            if (row == col)
                continue;
            const double factor = a[row][col];
            if (factor == 0.0)
                continue;
            for (int k = col; k < 4; ++k)
                a[row][k] -= factor * a[col][k];
        }
    }

    out[0] = a[0][3];
    out[1] = a[1][3];
    out[2] = a[2][3];
    return true;
}

char* duplicate_message(const std::string& message)
{
    char* result = static_cast<char*>(std::malloc(message.size() + 1));
    if (!result)
        return nullptr;
    std::memcpy(result, message.c_str(), message.size() + 1);
    return result;
}

template <typename T>
T* copy_vector_to_heap(const std::vector<T>& values)
{
    if (values.empty())
        return nullptr;
    const size_t bytes = values.size() * sizeof(T);
    T* out = static_cast<T*>(std::malloc(bytes));
    if (!out)
        throw std::bad_alloc();
    std::memcpy(out, values.data(), bytes);
    return out;
}

class Encoder
{
public:
    Encoder(const float* depth, const float* second_depth, const SSTEncoderOptions& options)
        : m_depth(depth)
        , m_second_depth(second_depth)
        , m_options(options)
    {
        if (!m_depth)
            throw std::runtime_error("depth pointer is null");
        m_width = std::max(1u, options.width);
        m_height = std::max(1u, options.height);
        m_tile_size = std::max(1u, options.tile_size);
        m_min_leaf_size = std::max(1u, options.min_leaf_size);
        m_plane_error_threshold = std::max(0.0f, options.plane_error_threshold);
        m_constant_epsilon = std::max(0.0f, options.constant_epsilon);
        m_use_dual_layer = options.use_dual_layer != 0 && options.has_second_depth != 0 && m_second_depth != nullptr;
        m_dual_has_slack = options.has_dual_depth_slack != 0;
        m_dual_depth_slack = std::max(0.0f, options.dual_depth_slack);
        m_dual_conservative = options.dual_conservative != 0;
        m_has_dual_max_leak = options.has_dual_max_leak != 0;
        m_dual_max_leak = std::max(0.0f, options.dual_max_leak);
        m_dual_max_leak_guard = m_has_dual_max_leak ? std::min(1e-6f, 0.1f * m_dual_max_leak) : 0.0f;
        m_dual_visibility_tolerance = std::max(0.0f, options.dual_visibility_tolerance);
        m_shadow_bias = std::max(0.0f, options.shadow_bias);
        m_quantization_radius = std::max(0u, options.plane_quantization_search_radius);
        m_has_forced_leaf_error_cap = options.has_forced_leaf_error_cap != 0;
        m_forced_leaf_error_cap = std::max(0.0f, options.forced_leaf_error_cap);
        m_forced_split_bias_fit = options.forced_split_bias_fit != 0;
        m_max_tree_depth = compute_max_tree_depth();
        m_branch_10bit_start_level = compute_branch_10bit_start_level();

        if (m_use_dual_layer)
        {
            m_upper_depth.resize(static_cast<size_t>(m_width) * static_cast<size_t>(m_height));
            for (uint32_t y = 0; y < m_height; ++y)
            {
                for (uint32_t x = 0; x < m_width; ++x)
                {
                    const size_t index = pixel_index(x, y);
                    const float lower = m_depth[index];
                    const float raw_upper = std::max(m_second_depth[index], lower);
                    float upper = raw_upper;
                    if (m_dual_has_slack)
                        upper = std::min(upper, lower + m_dual_depth_slack);
                    m_upper_depth[index] = std::max(upper, lower);
                }
            }
        }
    }

    void encode(SSTEncoderOutput& output)
    {
        const uint32_t tile_grid_x = ceil_div(m_width, m_tile_size);
        const uint32_t tile_grid_y = ceil_div(m_height, m_tile_size);
        const uint32_t tile_count = tile_grid_x * tile_grid_y;

        std::vector<TileResult> tile_results(tile_count);
        encode_tiles(tile_grid_x, tile_grid_y, tile_results);
        merge_tiles(tile_grid_x, tile_grid_y, tile_results);
        reorder_nodes_level_order();
        m_branch_10bit_start_level = find_safe_branch_10bit_start_level();

        std::vector<uint32_t> compact_words;
        std::vector<uint32_t> compact_roots;
        uint32_t compact_overflow_count = 0;
        uint32_t max_compact_offset = 0;
        pack_compact_nodes(compact_words, compact_roots, compact_overflow_count, max_compact_offset);

        std::vector<uint32_t> fixed64_nodes;
        pack_fixed64_nodes(fixed64_nodes);
        auto fixed64_overflow = count_fixed64_branch_offset_overflow();
        auto diagnostics = compute_compact_branch_offset_diagnostics();

        SSTEncoderStats stats{};
        fill_stats(tile_grid_x, tile_grid_y, compact_words, compact_roots, fixed64_nodes, compact_overflow_count, max_compact_offset, fixed64_overflow.first, fixed64_overflow.second, diagnostics, stats);

        output.nodes_size = m_nodes.size() * sizeof(Node);
        output.nodes = reinterpret_cast<uint8_t*>(copy_vector_to_heap(m_nodes));
        output.fixed64_word_count = fixed64_nodes.size();
        output.fixed64_nodes = copy_vector_to_heap(fixed64_nodes);
        output.compact_word_count = compact_words.size();
        output.compact_words = copy_vector_to_heap(compact_words);
        output.compact_root_count = compact_roots.size();
        output.compact_roots = copy_vector_to_heap(compact_roots);
        output.tile_root_count = m_tile_roots.size();
        output.tile_roots = copy_vector_to_heap(m_tile_roots);
        output.stats = stats;
    }

private:
    size_t pixel_index(uint32_t x, uint32_t y) const
    {
        return static_cast<size_t>(y) * static_cast<size_t>(m_width) + static_cast<size_t>(x);
    }

    float lower_at(uint32_t x, uint32_t y) const
    {
        return m_depth[pixel_index(x, y)];
    }

    float upper_at(uint32_t x, uint32_t y) const
    {
        if (!m_use_dual_layer)
            return lower_at(x, y);
        return m_upper_depth[pixel_index(x, y)];
    }

    RegionStats compute_region_stats(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) const
    {
        RegionStats stats{};
        stats.x0 = x0;
        stats.y0 = y0;
        stats.x1 = x1;
        stats.y1 = y1;
        stats.width = std::max(x1 - x0, 1u);
        stats.height = std::max(y1 - y0, 1u);

        const float leaky_upper_slack = std::max(0.0f, m_dual_max_leak - m_dual_max_leak_guard);
        for (uint32_t y = y0; y < y1; ++y)
        {
            const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(m_width) + static_cast<size_t>(x0);
            const float* lower_row = m_depth + row_base;
            const float* upper_row = m_use_dual_layer ? m_upper_depth.data() + row_base : nullptr;
            for (uint32_t local_index = 0; local_index < stats.width; ++local_index)
            {
                const float lower = lower_row[local_index];
                const float upper = m_use_dual_layer ? upper_row[local_index] : lower;

                stats.lower_min = std::min(stats.lower_min, lower);
                stats.lower_max = std::max(stats.lower_max, lower);
                stats.upper_min = std::min(stats.upper_min, upper);
                stats.leaky_upper_min = std::min(stats.leaky_upper_min, std::min(upper, lower + leaky_upper_slack));
            }
        }
        return stats;
    }

    RegionStats compute_region_moments(const RegionStats& base) const
    {
        RegionStats stats = base;
        const double inv_width = 1.0 / static_cast<double>(stats.width);
        const double inv_height = 1.0 / static_cast<double>(stats.height);
        for (uint32_t y = stats.y0; y < stats.y1; ++y)
        {
            const double local_y = ((static_cast<double>(y) + 0.5 - static_cast<double>(stats.y0)) * inv_height) - 0.5;
            const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(m_width) + static_cast<size_t>(stats.x0);
            const float* lower_row = m_depth + row_base;
            for (uint32_t x = stats.x0; x < stats.x1; ++x)
            {
                const uint32_t local_index = x - stats.x0;
                const double local_x = ((static_cast<double>(x) + 0.5 - static_cast<double>(stats.x0)) * inv_width) - 0.5;
                const double z = lower_row[local_index];
                stats.sum_x += local_x;
                stats.sum_y += local_y;
                stats.sum_xx += local_x * local_x;
                stats.sum_yy += local_y * local_y;
                stats.sum_xy += local_x * local_y;
                stats.sum_z += z;
                stats.sum_xz += local_x * z;
                stats.sum_yz += local_y * z;
                stats.count += 1.0;
            }
        }
        return stats;
    }

    uint32_t compute_max_tree_depth() const
    {
        if (m_tile_size <= 1 || m_min_leaf_size >= m_tile_size)
            return 0;
        return static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(m_tile_size) / static_cast<double>(std::max(m_min_leaf_size, 1u)))));
    }

    uint32_t compute_branch_10bit_start_level() const
    {
        for (uint32_t level = 0; level <= m_max_tree_depth; ++level)
        {
            const uint32_t child_depth = m_max_tree_depth > level + 1 ? m_max_tree_depth - level - 1 : 0;
            const uint64_t child_subtree_nodes = (pow4(child_depth + 1) - 1u) / 3u;
            const uint64_t max_child_offset = 1u + 3u * child_subtree_nodes;
            if (max_child_offset <= 0x3ffu)
                return level;
        }
        return m_max_tree_depth + 1;
    }

    static uint64_t pow4(uint32_t exp)
    {
        uint64_t value = 1;
        for (uint32_t i = 0; i < exp; ++i)
            value *= 4u;
        return value;
    }

    uint32_t branch_offset_bits_for_level(uint32_t level) const
    {
        return level >= m_branch_10bit_start_level ? 10u : 13u;
    }

    uint32_t append_node(TileResult& tile, uint32_t level) const
    {
        const uint32_t index = static_cast<uint32_t>(tile.nodes.size());
        tile.nodes.emplace_back();
        tile.levels.push_back(level);
        return index;
    }

    void encode_tiles(uint32_t tile_grid_x, uint32_t tile_grid_y, std::vector<TileResult>& tile_results)
    {
        const uint32_t tile_count = tile_grid_x * tile_grid_y;
        const uint32_t requested_threads = m_options.thread_count;
        uint32_t worker_count = requested_threads != 0 ? requested_threads : std::thread::hardware_concurrency();
        worker_count = std::max(1u, std::min(worker_count, tile_count == 0 ? 1u : tile_count));

        std::atomic<uint32_t> next_tile{0};
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::string error_message;

        auto worker = [&]() {
            while (!failed.load(std::memory_order_relaxed))
            {
                const uint32_t tile_index = next_tile.fetch_add(1, std::memory_order_relaxed);
                if (tile_index >= tile_count)
                    return;

                try
                {
                    const uint32_t tile_x = tile_index % tile_grid_x;
                    const uint32_t tile_y = tile_index / tile_grid_x;
                    const uint32_t x0 = tile_x * m_tile_size;
                    const uint32_t y0 = tile_y * m_tile_size;
                    const uint32_t x1 = std::min(x0 + m_tile_size, m_width);
                    const uint32_t y1 = std::min(y0 + m_tile_size, m_height);
                    TileResult tile;
                    const uint32_t tile_width = x1 - x0;
                    const uint32_t tile_height = y1 - y0;
                    const uint32_t leaf_area = std::max(1u, m_min_leaf_size * m_min_leaf_size);
                    const uint32_t reserve_nodes = std::min(2048u, std::max(64u, (tile_width * tile_height) / (leaf_area * 4u)));
                    tile.nodes.reserve(reserve_nodes);
                    tile.levels.reserve(reserve_nodes);
                    tile.root = append_node(tile, 0);
                    encode_region_at(tile, tile.root, x0, y0, x1, y1, 0, m_dual_visibility_tolerance);
                    tile_results[tile_index] = std::move(tile);
                }
                catch (const std::exception& exc)
                {
                    bool expected = false;
                    if (failed.compare_exchange_strong(expected, true))
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        error_message = exc.what();
                    }
                    return;
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (uint32_t i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();

        if (failed.load())
            throw std::runtime_error(error_message.empty() ? "SST tile encode failed" : error_message);
    }

    void merge_tiles(uint32_t tile_grid_x, uint32_t tile_grid_y, const std::vector<TileResult>& tile_results)
    {
        const uint32_t tile_count = tile_grid_x * tile_grid_y;
        m_nodes.clear();
        m_node_levels.clear();
        m_tile_roots.assign(tile_count, kInvalidRoot);
        m_forced_leaf_node_count = 0;
        m_forced_leaf_pixel_count = 0;
        m_forced_leaf_error_sum = 0.0;
        m_forced_leaf_max_error = 0.0f;

        for (uint32_t tile_index = 0; tile_index < tile_count; ++tile_index)
        {
            const TileResult& tile = tile_results[tile_index];
            if (tile.root == kInvalidRoot || tile.nodes.empty())
                continue;

            const uint32_t base = static_cast<uint32_t>(m_nodes.size());
            m_tile_roots[tile_index] = base + tile.root;
            m_forced_leaf_node_count += tile.forced_leaf_node_count;
            m_forced_leaf_pixel_count += tile.forced_leaf_pixel_count;
            m_forced_leaf_error_sum += tile.forced_leaf_error_sum;
            m_forced_leaf_max_error = std::max(m_forced_leaf_max_error, tile.forced_leaf_max_error);

            m_nodes.reserve(m_nodes.size() + tile.nodes.size());
            m_node_levels.reserve(m_node_levels.size() + tile.levels.size());
            for (const Node& local_node : tile.nodes)
            {
                Node node = local_node;
                if ((node.flags & kNodeFlagPlane) == 0)
                    node.child_base += base;
                m_nodes.push_back(node);
            }
            for (uint32_t level : tile.levels)
                m_node_levels.push_back(level);
        }
    }

    void encode_region_at(
        TileResult& tile,
        uint32_t node_index,
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        uint32_t level,
        float visibility_tolerance) const
    {
        const uint32_t width = x1 - x0;
        const uint32_t height = y1 - y0;
        const RegionStats stats = compute_region_stats(x0, y0, x1, y1);
        const FitResult fit = fit_plane(stats, visibility_tolerance);
        const bool reached_min_leaf = width <= m_min_leaf_size || height <= m_min_leaf_size;
        const bool can_split_further = width > 1 && height > 1;
        const bool forced_leaf_blocked =
            reached_min_leaf &&
            can_split_further &&
            m_has_forced_leaf_error_cap &&
            fit.max_error > m_forced_leaf_error_cap;

        if (fit.max_error <= m_plane_error_threshold || (reached_min_leaf && !forced_leaf_blocked))
        {
            if (fit.max_error > m_plane_error_threshold)
            {
                const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
                tile.forced_leaf_node_count += 1;
                tile.forced_leaf_pixel_count += pixel_count;
                tile.forced_leaf_error_sum += static_cast<double>(fit.max_error) * static_cast<double>(pixel_count);
                tile.forced_leaf_max_error = std::max(tile.forced_leaf_max_error, fit.max_error);
            }
            tile.nodes[node_index] = make_plane_node(fit.flags, x0, y0, fit.a, fit.b, fit.c, fit.max_error);
            return;
        }

        if (stats.lower_max - stats.lower_min <= m_constant_epsilon)
        {
            tile.nodes[node_index] = make_plane_node(kNodeFlagPlane | kNodeFlagConstant, x0, y0, 0.0f, 0.0f, stats.lower_min, stats.lower_max - stats.lower_min);
            return;
        }

        const uint32_t mid_x = x0 + std::max(width / 2u, 1u);
        const uint32_t mid_y = y0 + std::max(height / 2u, 1u);
        const uint32_t child_base = static_cast<uint32_t>(tile.nodes.size());
        for (uint32_t child = 0; child < 4; ++child)
            append_node(tile, level + 1);

        const float child_visibility_tolerance = forced_leaf_blocked && m_forced_split_bias_fit ? 0.0f : visibility_tolerance;
        encode_region_at(tile, child_base + 0, x0, y0, mid_x, mid_y, level + 1, child_visibility_tolerance);
        encode_region_at(tile, child_base + 1, mid_x, y0, x1, mid_y, level + 1, child_visibility_tolerance);
        encode_region_at(tile, child_base + 2, x0, mid_y, mid_x, y1, level + 1, child_visibility_tolerance);
        encode_region_at(tile, child_base + 3, mid_x, mid_y, x1, y1, level + 1, child_visibility_tolerance);

        Node branch{};
        branch.flags = 0;
        branch.child_base = child_base;
        branch.meta = (x0 & 0xffffu) | ((y0 & 0xffffu) << 16);
        branch.max_error = fit.max_error;
        tile.nodes[node_index] = branch;
    }

    Node make_plane_node(uint32_t flags, uint32_t x0, uint32_t y0, float a, float b, float c, float max_error) const
    {
        Node node{};
        node.flags = flags;
        node.child_base = 0;
        node.meta = (x0 & 0xffffu) | ((y0 & 0xffffu) << 16);
        node.a = a;
        node.b = b;
        node.c = c;
        node.max_error = max_error;
        return node;
    }

    FitResult fit_plane(const RegionStats& stats, float visibility_tolerance) const
    {
        auto constant = fit_constant_depth(stats, visibility_tolerance);
        if (constant.second <= m_plane_error_threshold)
        {
            FitResult result{};
            result.flags = kNodeFlagPlane | kNodeFlagConstant;
            result.c = constant.first;
            result.max_error = constant.second;
            return result;
        }

        const RegionStats moments = compute_region_moments(stats);
        double coeff[3] = {0.0, 0.0, static_cast<double>(constant.first)};
        if (!fit_least_squares_plane(moments, coeff))
        {
            FitResult result{};
            result.flags = kNodeFlagPlane | kNodeFlagConstant;
            result.c = constant.first;
            result.max_error = constant.second;
            return result;
        }

        if (m_use_dual_layer && m_dual_conservative)
        {
            const float shift = std::min(min_lower_minus_predicted(stats.x0, stats.y0, stats.x1, stats.y1, coeff[0], coeff[1], coeff[2]), 0.0f);
            coeff[2] += shift;
        }
        else if (m_use_dual_layer && m_has_dual_max_leak)
        {
            auto shift_error = compute_dual_biased_shift_and_error(stats.x0, stats.y0, stats.x1, stats.y1, coeff[0], coeff[1], coeff[2], visibility_tolerance, true);
            coeff[2] += shift_error.first;
        }
        else if (m_use_dual_layer)
        {
            const float min_shift = max_lower_minus_predicted(stats.x0, stats.y0, stats.x1, stats.y1, coeff[0], coeff[1], coeff[2]);
            const float max_shift = min_upper_minus_predicted(stats.x0, stats.y0, stats.x1, stats.y1, coeff[0], coeff[1], coeff[2]);
            if (min_shift <= max_shift)
                coeff[2] += std::min(std::max(0.0f, min_shift), max_shift);
            else
                coeff[2] += 0.5f * (min_shift + max_shift);
        }
        else
        {
            const float shift = std::min(min_lower_minus_predicted(stats.x0, stats.y0, stats.x1, stats.y1, coeff[0], coeff[1], coeff[2]), 0.0f);
            coeff[2] += shift;
        }

        const PlaneEvalResult adjusted = quantize_adjust_plane(stats.x0, stats.y0, stats.x1, stats.y1, coeff[0], coeff[1], coeff[2], visibility_tolerance);
        FitResult result{};
        result.flags = kNodeFlagPlane;
        result.a = adjusted.a;
        result.b = adjusted.b;
        result.c = adjusted.c;
        result.max_error = adjusted.max_error;
        return result;
    }

    std::pair<float, float> fit_constant_depth(const RegionStats& stats, float visibility_tolerance) const
    {
        constexpr uint32_t max_unorm = (1u << 30) - 1u;

        if (m_use_dual_layer && m_dual_conservative)
        {
            const uint32_t q = static_cast<uint32_t>(std::floor(std::clamp(stats.lower_min, 0.0f, 1.0f) * static_cast<float>(max_unorm)));
            const float value = decode_unorm(q, max_unorm);
            const float max_error = std::max(std::max(stats.lower_max - value, 0.0f), std::max(value - stats.lower_min, 0.0f));
            return {value, max_error};
        }

        if (m_use_dual_layer)
        {
            float interval_min = stats.lower_max;
            float interval_max = stats.upper_min;
            if (m_has_dual_max_leak)
            {
                interval_min = stats.lower_max - std::max(0.0f, visibility_tolerance - m_dual_max_leak_guard);
                interval_max = stats.leaky_upper_min;
            }

            uint32_t q_value = 0;
            if (interval_min <= interval_max)
            {
                float preferred_value = interval_min;
                if (visibility_tolerance > 0.0f)
                    preferred_value = std::min(std::max(stats.lower_max, interval_min), interval_max);
                const uint32_t q_min = static_cast<uint32_t>(std::ceil(std::clamp(interval_min, 0.0f, 1.0f) * static_cast<float>(max_unorm)));
                const uint32_t q_max = static_cast<uint32_t>(std::floor(std::clamp(interval_max, 0.0f, 1.0f) * static_cast<float>(max_unorm)));
                const uint32_t preferred_q = quantize_unorm(preferred_value, 30);
                q_value = q_min <= q_max ? std::min(std::max(preferred_q, q_min), q_max) : quantize_unorm(preferred_value, 30);
            }
            else if (m_has_dual_max_leak)
            {
                q_value = quantize_unorm(std::min(std::max(0.0f, interval_min), interval_max), 30);
            }
            else
            {
                q_value = quantize_unorm(0.5f * (interval_min + interval_max), 30);
            }

            const float value = decode_unorm(q_value, max_unorm);
            float max_error = 0.0f;
            if (m_has_dual_max_leak)
            {
                const float lower_limit_max = stats.lower_max - std::max(0.0f, visibility_tolerance - m_dual_max_leak_guard);
                max_error = std::max(std::max(lower_limit_max - value, 0.0f), std::max(value - stats.leaky_upper_min, 0.0f));
            }
            else
            {
                max_error = std::max(std::max(stats.lower_max - value, 0.0f), std::max(value - stats.upper_min, 0.0f));
            }
            if (m_has_dual_max_leak && max_error > 1e-7f)
                return {value, m_plane_error_threshold + max_error};
            if (m_has_dual_max_leak && visibility_tolerance <= 0.0f)
            {
                const float conservative_error = std::max(stats.lower_max - value, 0.0f);
                return {value, conservative_error};
            }
            if (m_has_dual_max_leak)
                return {value, 0.0f};
            return {value, max_error};
        }

        const uint32_t q = static_cast<uint32_t>(std::floor(std::clamp(stats.lower_min, 0.0f, 1.0f) * static_cast<float>(max_unorm)));
        const float value = decode_unorm(q, max_unorm);
        const float max_error = std::max(std::max(stats.lower_max - value, 0.0f), std::max(value - stats.lower_min, 0.0f));
        return {value, max_error};
    }

    bool fit_least_squares_plane(const RegionStats& stats, double coeff[3]) const
    {
        const uint32_t width = stats.width;
        const uint32_t height = stats.height;
        if (width == 1 && height == 1)
        {
            coeff[0] = 0.0;
            coeff[1] = 0.0;
            coeff[2] = stats.sum_z;
            return true;
        }

        if (width == 1)
        {
            double b = 0.0;
            double c = 0.0;
            if (!solve_2x2(stats.sum_yy, stats.sum_y, stats.count, stats.sum_yz, stats.sum_z, b, c))
                return false;
            coeff[0] = 0.0;
            coeff[1] = b;
            coeff[2] = c;
            return true;
        }

        if (height == 1)
        {
            double a = 0.0;
            double c = 0.0;
            if (!solve_2x2(stats.sum_xx, stats.sum_x, stats.count, stats.sum_xz, stats.sum_z, a, c))
                return false;
            coeff[0] = a;
            coeff[1] = 0.0;
            coeff[2] = c;
            return true;
        }

        double matrix[3][3] = {
            {stats.sum_xx, stats.sum_xy, stats.sum_x},
            {stats.sum_xy, stats.sum_yy, stats.sum_y},
            {stats.sum_x, stats.sum_y, stats.count},
        };
        double rhs[3] = {stats.sum_xz, stats.sum_yz, stats.sum_z};
        return solve_3x3(matrix, rhs, coeff);
    }

    float predicted_at(uint32_t x, uint32_t y, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, double a, double b, double c) const
    {
        const double local_x = ((static_cast<double>(x) + 0.5 - static_cast<double>(x0)) / static_cast<double>(std::max(width, 1u))) - 0.5;
        const double local_y = ((static_cast<double>(y) + 0.5 - static_cast<double>(y0)) / static_cast<double>(std::max(height, 1u))) - 0.5;
        return static_cast<float>(a * local_x + b * local_y + c);
    }

    float min_lower_minus_predicted(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, double a, double b, double c) const
    {
        float result = std::numeric_limits<float>::infinity();
        const uint32_t width = x1 - x0;
        const uint32_t height = y1 - y0;
        for (uint32_t y = y0; y < y1; ++y)
            for (uint32_t x = x0; x < x1; ++x)
                result = std::min(result, lower_at(x, y) - predicted_at(x, y, x0, y0, width, height, a, b, c));
        return result;
    }

    float max_lower_minus_predicted(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, double a, double b, double c) const
    {
        float result = -std::numeric_limits<float>::infinity();
        const uint32_t width = x1 - x0;
        const uint32_t height = y1 - y0;
        for (uint32_t y = y0; y < y1; ++y)
            for (uint32_t x = x0; x < x1; ++x)
                result = std::max(result, lower_at(x, y) - predicted_at(x, y, x0, y0, width, height, a, b, c));
        return result;
    }

    float min_upper_minus_predicted(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, double a, double b, double c) const
    {
        float result = std::numeric_limits<float>::infinity();
        const uint32_t width = x1 - x0;
        const uint32_t height = y1 - y0;
        for (uint32_t y = y0; y < y1; ++y)
            for (uint32_t x = x0; x < x1; ++x)
                result = std::min(result, upper_at(x, y) - predicted_at(x, y, x0, y0, width, height, a, b, c));
        return result;
    }

    std::pair<float, float> compute_dual_biased_shift_and_error(
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        double a,
        double b,
        double c,
        float visibility_tolerance,
        bool allow_shift) const
    {
        const uint32_t width = x1 - x0;
        const uint32_t height = y1 - y0;
        float lower_shift = -std::numeric_limits<float>::infinity();
        float upper_shift = std::numeric_limits<float>::infinity();
        if (allow_shift)
        {
            for (uint32_t y = y0; y < y1; ++y)
            {
                for (uint32_t x = x0; x < x1; ++x)
                {
                    const float lower = lower_at(x, y);
                    const float upper = upper_at(x, y);
                    const float upper_limit = std::min(upper, lower + std::max(0.0f, m_dual_max_leak - m_dual_max_leak_guard));
                    const float lower_limit = lower - std::max(0.0f, visibility_tolerance - m_dual_max_leak_guard);
                    const float predicted = predicted_at(x, y, x0, y0, width, height, a, b, c);
                    lower_shift = std::max(lower_shift, lower_limit - predicted);
                    upper_shift = std::min(upper_shift, upper_limit - predicted);
                }
            }
        }

        float shift = 0.0f;
        if (allow_shift)
        {
            if (lower_shift <= upper_shift)
            {
                if (visibility_tolerance > 0.0f)
                {
                    const float preferred_shift = max_lower_minus_predicted(x0, y0, x1, y1, a, b, c);
                    shift = std::min(std::max(preferred_shift, lower_shift), upper_shift);
                }
                else
                {
                    shift = std::min(std::max(0.0f, lower_shift), upper_shift);
                }
            }
            else
            {
                shift = upper_shift;
            }
        }

        float practical_error = 0.0f;
        for (uint32_t y = y0; y < y1; ++y)
        {
            for (uint32_t x = x0; x < x1; ++x)
            {
                const float lower = lower_at(x, y);
                const float upper = upper_at(x, y);
                const float upper_limit = std::min(upper, lower + std::max(0.0f, m_dual_max_leak - m_dual_max_leak_guard));
                const float lower_limit = lower - std::max(0.0f, visibility_tolerance - m_dual_max_leak_guard);
                const float shifted = predicted_at(x, y, x0, y0, width, height, a, b, c) + shift;
                practical_error = std::max(practical_error, std::max(std::max(lower_limit - shifted, 0.0f), std::max(shifted - upper_limit, 0.0f)));
            }
        }

        float max_error = 0.0f;
        if (practical_error > 1e-7f)
        {
            max_error = m_plane_error_threshold + practical_error;
        }
        else if (visibility_tolerance <= 0.0f)
        {
            for (uint32_t y = y0; y < y1; ++y)
                for (uint32_t x = x0; x < x1; ++x)
                    max_error = std::max(max_error, lower_at(x, y) - (predicted_at(x, y, x0, y0, width, height, a, b, c) + shift));
        }
        else
        {
            max_error = 0.0f;
        }
        return {shift, max_error};
    }

    PlaneEvalResult quantize_adjust_plane(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, double a, double b, double c, float visibility_tolerance) const
    {
        const int qa = static_cast<int>(quantize_snorm(static_cast<float>(a), 16, 2.0f));
        const int qb = static_cast<int>(quantize_snorm(static_cast<float>(b), 16, 2.0f));
        const int max_snorm = (1 << 16) - 1;
        const int radius = static_cast<int>(m_quantization_radius);

        PlaneEvalResult best = evaluate_quantized_plane_slopes(x0, y0, x1, y1, qa, qb, static_cast<float>(c), visibility_tolerance);
        for (int da = -radius; da <= radius; ++da)
        {
            const int candidate_qa = std::clamp(qa + da, 0, max_snorm);
            for (int db = -radius; db <= radius; ++db)
            {
                if (da == 0 && db == 0)
                    continue;
                const int candidate_qb = std::clamp(qb + db, 0, max_snorm);
                const PlaneEvalResult candidate = evaluate_quantized_plane_slopes(x0, y0, x1, y1, candidate_qa, candidate_qb, static_cast<float>(c), visibility_tolerance);
                if (candidate.max_error < best.max_error)
                    best = candidate;
            }
        }
        return best;
    }

    PlaneEvalResult evaluate_quantized_plane_slopes(
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        int qa,
        int qb,
        float c,
        float visibility_tolerance) const
    {
        constexpr uint32_t max_snorm = (1u << 16) - 1u;
        constexpr uint32_t max_unorm = (1u << 30) - 1u;
        double a = decode_snorm(static_cast<uint32_t>(qa), max_snorm, 2.0f);
        double b = decode_snorm(static_cast<uint32_t>(qb), max_snorm, 2.0f);
        double quantized_c = decode_unorm(quantize_unorm(c, 30), max_unorm);

        float shift = 0.0f;
        if (m_use_dual_layer && m_dual_conservative)
            shift = std::min(min_lower_minus_predicted(x0, y0, x1, y1, a, b, quantized_c), 0.0f);
        else if (m_use_dual_layer && m_has_dual_max_leak)
            shift = compute_dual_biased_shift_and_error(x0, y0, x1, y1, a, b, quantized_c, visibility_tolerance, true).first;
        else if (m_use_dual_layer)
        {
            const float min_shift = max_lower_minus_predicted(x0, y0, x1, y1, a, b, quantized_c);
            const float max_shift = min_upper_minus_predicted(x0, y0, x1, y1, a, b, quantized_c);
            if (min_shift <= max_shift)
                shift = std::min(std::max(0.0f, min_shift), max_shift);
            else
                shift = 0.5f * (min_shift + max_shift);
        }
        else
            shift = std::min(min_lower_minus_predicted(x0, y0, x1, y1, a, b, quantized_c), 0.0f);

        quantized_c = decode_unorm(quantize_unorm(static_cast<float>(quantized_c + shift), 30), max_unorm);

        float max_error = 0.0f;
        const uint32_t width = x1 - x0;
        const uint32_t height = y1 - y0;
        if (m_use_dual_layer && m_has_dual_max_leak && !m_dual_conservative)
        {
            max_error = compute_dual_biased_shift_and_error(x0, y0, x1, y1, a, b, quantized_c, visibility_tolerance, false).second;
        }
        else
        {
            for (uint32_t y = y0; y < y1; ++y)
            {
                for (uint32_t x = x0; x < x1; ++x)
                {
                    const float predicted = predicted_at(x, y, x0, y0, width, height, a, b, quantized_c);
                    const float lower = lower_at(x, y);
                    if (m_use_dual_layer && !m_dual_conservative)
                    {
                        const float upper = upper_at(x, y);
                        max_error = std::max(max_error, std::max(std::max(lower - predicted, 0.0f), std::max(predicted - upper, 0.0f)));
                    }
                    else
                    {
                        max_error = std::max(max_error, std::max(std::max(lower - predicted, 0.0f), std::max(predicted - lower, 0.0f)));
                    }
                }
            }
        }

        PlaneEvalResult result{};
        result.a = static_cast<float>(a);
        result.b = static_cast<float>(b);
        result.c = static_cast<float>(quantized_c);
        result.max_error = max_error;
        return result;
    }

    void reorder_nodes_level_order()
    {
        if (m_nodes.empty())
            return;

        std::vector<uint8_t> queued(m_nodes.size(), 0);
        std::vector<uint32_t> order;
        order.reserve(m_nodes.size());
        std::vector<uint32_t> queue;
        queue.reserve(m_nodes.size());

        for (uint32_t root : m_tile_roots)
        {
            if (root == kInvalidRoot || root >= m_nodes.size() || queued[root])
                continue;
            queue.clear();
            queue.push_back(root);
            queued[root] = 1;
            size_t cursor = 0;
            while (cursor < queue.size())
            {
                const uint32_t old_index = queue[cursor++];
                order.push_back(old_index);
                const Node& node = m_nodes[old_index];
                if (node.flags & kNodeFlagPlane)
                    continue;
                for (uint32_t child = 0; child < 4; ++child)
                {
                    const uint32_t child_index = node.child_base + child;
                    if (child_index < m_nodes.size() && !queued[child_index])
                    {
                        queued[child_index] = 1;
                        queue.push_back(child_index);
                    }
                }
            }
        }

        if (order.size() != m_nodes.size())
        {
            for (uint32_t index = 0; index < m_nodes.size(); ++index)
            {
                if (!queued[index])
                    order.push_back(index);
            }
        }

        std::vector<uint32_t> old_to_new(m_nodes.size(), kInvalidRoot);
        for (uint32_t new_index = 0; new_index < order.size(); ++new_index)
            old_to_new[order[new_index]] = new_index;

        std::vector<Node> reordered_nodes;
        std::vector<uint32_t> reordered_levels;
        reordered_nodes.reserve(m_nodes.size());
        reordered_levels.reserve(m_node_levels.size());
        for (uint32_t old_index : order)
        {
            Node node = m_nodes[old_index];
            if ((node.flags & kNodeFlagPlane) == 0 && node.child_base < old_to_new.size())
                node.child_base = old_to_new[node.child_base];
            reordered_nodes.push_back(node);
            reordered_levels.push_back(m_node_levels[old_index]);
        }

        for (uint32_t& root : m_tile_roots)
        {
            if (root != kInvalidRoot && root < old_to_new.size())
                root = old_to_new[root];
        }

        m_nodes = std::move(reordered_nodes);
        m_node_levels = std::move(reordered_levels);
    }

    std::vector<uint32_t> compute_node_word_offsets() const
    {
        std::vector<uint32_t> offsets;
        offsets.reserve(m_nodes.size());
        uint32_t word_count = 0;
        for (const Node& node : m_nodes)
        {
            offsets.push_back(word_count);
            word_count += (node.flags & kNodeFlagConstant) ? 1u : 2u;
        }
        return offsets;
    }

    uint32_t find_safe_branch_10bit_start_level() const
    {
        const std::vector<uint32_t> word_offsets = compute_node_word_offsets();
        for (uint32_t start_level = 0; start_level <= m_max_tree_depth + 1; ++start_level)
        {
            bool has_overflow = false;
            for (uint32_t node_index = 0; node_index < m_nodes.size() && !has_overflow; ++node_index)
            {
                const Node& node = m_nodes[node_index];
                if (node.flags & kNodeFlagPlane)
                    continue;
                const uint32_t bits = m_node_levels[node_index] >= start_level ? 10u : 13u;
                const uint32_t max_value = (1u << bits) - 1u;
                for (uint32_t child = 0; child < 4; ++child)
                {
                    const uint32_t child_index = node.child_base + child;
                    const uint32_t node_relative_offset = child_index - node_index;
                    const uint32_t word_relative_offset = word_offsets[child_index] - word_offsets[node_index];
                    if (node_relative_offset > max_value || word_relative_offset > max_value)
                    {
                        has_overflow = true;
                        break;
                    }
                }
            }
            if (!has_overflow)
                return start_level;
        }
        return m_max_tree_depth + 1;
    }

    void pack_compact_nodes(std::vector<uint32_t>& words, std::vector<uint32_t>& compact_roots, uint32_t& overflow_count, uint32_t& max_branch_offset) const
    {
        const std::vector<uint32_t> word_offsets = compute_node_word_offsets();
        uint32_t word_count = 0;
        if (!word_offsets.empty())
            word_count = word_offsets.back() + ((m_nodes.back().flags & kNodeFlagConstant) ? 1u : 2u);
        words.assign(std::max(word_count, 1u), 0u);
        compact_roots.assign(m_tile_roots.size(), kInvalidRoot);
        for (size_t i = 0; i < m_tile_roots.size(); ++i)
        {
            const uint32_t root = m_tile_roots[i];
            if (root != kInvalidRoot && root < word_offsets.size())
                compact_roots[i] = word_offsets[root];
        }

        overflow_count = 0;
        max_branch_offset = 0;
        for (uint32_t node_index = 0; node_index < m_nodes.size(); ++node_index)
        {
            const Node& node = m_nodes[node_index];
            const uint32_t word_offset = word_offsets[node_index];
            if (node.flags & kNodeFlagConstant)
            {
                const uint32_t qz = quantize_unorm(node.c, 30);
                words[word_offset] = (qz << 2) | 0x1u;
            }
            else if (node.flags & kNodeFlagPlane)
            {
                const uint32_t qz = quantize_unorm(node.c, 30);
                const uint32_t qx = quantize_snorm(node.a, 16, 2.0f);
                const uint32_t qy = quantize_snorm(node.b, 16, 2.0f);
                words[word_offset] = (qz << 2) | 0x2u;
                words[word_offset + 1] = qx | (qy << 16);
            }
            else
            {
                const uint32_t bits = branch_offset_bits_for_level(m_node_levels[node_index]);
                const uint64_t max_value = (uint64_t{1} << bits) - 1u;
                uint64_t payload = 0;
                for (uint32_t child = 0; child < 4; ++child)
                {
                    const uint32_t child_index = node.child_base + child;
                    const uint32_t relative_offset = word_offsets[child_index] - word_offset;
                    max_branch_offset = std::max(max_branch_offset, relative_offset);
                    if (relative_offset > max_value)
                        overflow_count += 1;
                    payload |= std::min<uint64_t>(relative_offset, max_value) << (bits * child);
                }
                const uint64_t packed = payload << 2;
                words[word_offset] = static_cast<uint32_t>(packed & 0xffffffffu);
                words[word_offset + 1] = static_cast<uint32_t>((packed >> 32) & 0xffffffffu);
            }
        }
    }

    void pack_fixed64_nodes(std::vector<uint32_t>& packed) const
    {
        packed.assign(m_nodes.size() * 2, 0u);
        for (uint32_t node_index = 0; node_index < m_nodes.size(); ++node_index)
        {
            const Node& node = m_nodes[node_index];
            uint32_t word0 = 0;
            uint32_t word1 = 0;
            if (node.flags & kNodeFlagConstant)
            {
                const uint32_t qz = quantize_unorm(node.c, 30);
                word0 = (qz << 2) | 0x1u;
            }
            else if (node.flags & kNodeFlagPlane)
            {
                const uint32_t qz = quantize_unorm(node.c, 30);
                const uint32_t qx = quantize_snorm(node.a, 16, 2.0f);
                const uint32_t qy = quantize_snorm(node.b, 16, 2.0f);
                word0 = (qz << 2) | 0x2u;
                word1 = qx | (qy << 16);
            }
            else
            {
                const uint32_t bits = branch_offset_bits_for_level(m_node_levels[node_index]);
                const uint64_t max_value = (uint64_t{1} << bits) - 1u;
                uint64_t payload = 0;
                for (uint32_t child = 0; child < 4; ++child)
                {
                    const uint32_t relative_offset = node.child_base + child - node_index;
                    payload |= std::min<uint64_t>(relative_offset, max_value) << (bits * child);
                }
                const uint64_t packed_bits = payload << 2;
                word0 = static_cast<uint32_t>(packed_bits & 0xffffffffu);
                word1 = static_cast<uint32_t>((packed_bits >> 32) & 0xffffffffu);
            }
            packed[node_index * 2] = word0;
            packed[node_index * 2 + 1] = word1;
        }
    }

    std::pair<uint32_t, uint32_t> count_fixed64_branch_offset_overflow() const
    {
        uint32_t overflow_count = 0;
        uint32_t max_branch_offset = 0;
        for (uint32_t node_index = 0; node_index < m_nodes.size(); ++node_index)
        {
            const Node& node = m_nodes[node_index];
            if (node.flags & kNodeFlagPlane)
                continue;
            const uint32_t bits = branch_offset_bits_for_level(m_node_levels[node_index]);
            const uint32_t max_value = (1u << bits) - 1u;
            for (uint32_t child = 0; child < 4; ++child)
            {
                const uint32_t relative_offset = node.child_base + child - node_index;
                max_branch_offset = std::max(max_branch_offset, relative_offset);
                if (relative_offset > max_value)
                    overflow_count += 1;
            }
        }
        return {overflow_count, max_branch_offset};
    }

    struct BranchDiagnostics
    {
        uint32_t max_13bit_offset = 0;
        uint32_t max_10bit_offset = 0;
    };

    BranchDiagnostics compute_compact_branch_offset_diagnostics() const
    {
        const std::vector<uint32_t> word_offsets = compute_node_word_offsets();
        BranchDiagnostics diagnostics{};
        for (uint32_t node_index = 0; node_index < m_nodes.size(); ++node_index)
        {
            const Node& node = m_nodes[node_index];
            if (node.flags & kNodeFlagPlane)
                continue;
            const uint32_t bits = branch_offset_bits_for_level(m_node_levels[node_index]);
            const uint32_t word_offset = word_offsets[node_index];
            for (uint32_t child = 0; child < 4; ++child)
            {
                const uint32_t child_index = node.child_base + child;
                const uint32_t relative_offset = word_offsets[child_index] - word_offset;
                if (bits == 10)
                    diagnostics.max_10bit_offset = std::max(diagnostics.max_10bit_offset, relative_offset);
                else
                    diagnostics.max_13bit_offset = std::max(diagnostics.max_13bit_offset, relative_offset);
            }
        }
        return diagnostics;
    }

    void fill_stats(
        uint32_t tile_grid_x,
        uint32_t tile_grid_y,
        const std::vector<uint32_t>& compact_words,
        const std::vector<uint32_t>& compact_roots,
        const std::vector<uint32_t>& fixed64_nodes,
        uint32_t compact_overflow_count,
        uint32_t max_compact_offset,
        uint32_t fixed64_overflow_count,
        uint32_t max_fixed64_offset,
        BranchDiagnostics diagnostics,
        SSTEncoderStats& stats) const
    {
        uint32_t branch_count = 0;
        uint32_t branch_13bit_count = 0;
        uint32_t uniform_count = 0;
        float max_error = 0.0f;
        for (uint32_t node_index = 0; node_index < m_nodes.size(); ++node_index)
        {
            const Node& node = m_nodes[node_index];
            if ((node.flags & kNodeFlagPlane) == 0)
            {
                branch_count += 1;
                if (branch_offset_bits_for_level(m_node_levels[node_index]) == 13)
                    branch_13bit_count += 1;
            }
            else
            {
                max_error = std::max(max_error, node.max_error);
                if (node.flags & kNodeFlagConstant)
                    uniform_count += 1;
            }
        }
        const uint32_t branch_10bit_count = branch_count - branch_13bit_count;
        const uint32_t plane_count = static_cast<uint32_t>(m_nodes.size()) - branch_count;
        const uint32_t plane62_count = plane_count - uniform_count;
        const uint64_t original_bytes = static_cast<uint64_t>(m_width) * static_cast<uint64_t>(m_height) * 4u;
        const uint64_t encoded_bytes = static_cast<uint64_t>(m_nodes.size()) * sizeof(Node) + static_cast<uint64_t>(m_tile_roots.size()) * sizeof(uint32_t);
        const uint64_t packed_encoded_bytes = static_cast<uint64_t>(compact_words.size() + compact_roots.size()) * sizeof(uint32_t);
        const uint64_t fixed64_encoded_bytes = static_cast<uint64_t>(fixed64_nodes.size() + m_tile_roots.size()) * sizeof(uint32_t);
        const uint64_t decompressed_depth_bytes = original_bytes;
        const uint64_t packed_decompressed_working_set_bytes = packed_encoded_bytes + decompressed_depth_bytes;

        stats.width = m_width;
        stats.height = m_height;
        stats.tile_grid_x = tile_grid_x;
        stats.tile_grid_y = tile_grid_y;
        stats.tile_size = m_tile_size;
        stats.min_leaf_size = m_min_leaf_size;
        stats.max_tree_depth = m_max_tree_depth;
        stats.max_traversal_steps = m_max_tree_depth + 1;
        stats.branch_10bit_start_level = m_branch_10bit_start_level;
        stats.tile_count = tile_grid_x * tile_grid_y;
        stats.node_count = static_cast<uint32_t>(m_nodes.size());
        stats.branch_node_count = branch_count;
        stats.branch_13bit_node_count = branch_13bit_count;
        stats.branch_10bit_node_count = branch_10bit_count;
        stats.plane_node_count = plane_count;
        stats.uniform_plane_node_count = uniform_count;
        stats.compact_branch_words = branch_count * 2;
        stats.compact_30bit_plane_words = uniform_count;
        stats.compact_62bit_plane_words = plane62_count * 2;
        stats.compact_node_words = static_cast<uint32_t>(compact_words.size());
        stats.compact_tile_root_bytes = static_cast<uint32_t>(compact_roots.size() * sizeof(uint32_t));
        stats.compact_branch_offset_overflow_count = compact_overflow_count;
        stats.fixed64_branch_offset_overflow_count = fixed64_overflow_count;
        stats.max_compact_branch_offset = max_compact_offset;
        stats.max_fixed64_branch_offset = max_fixed64_offset;
        stats.compact_branch_13bit_max_offset = diagnostics.max_13bit_offset;
        stats.compact_branch_13bit_capacity_percent = diagnostics.max_13bit_offset * 100.0f / static_cast<float>((1u << 13) - 1u);
        stats.compact_branch_10bit_max_offset = diagnostics.max_10bit_offset;
        stats.compact_branch_10bit_capacity_percent = diagnostics.max_10bit_offset * 100.0f / static_cast<float>((1u << 10) - 1u);
        stats.forced_leaf_node_count = m_forced_leaf_node_count;
        stats.forced_leaf_pixel_count = m_forced_leaf_pixel_count;
        stats.forced_leaf_max_error = m_forced_leaf_max_error;
        stats.forced_leaf_error_sum = m_forced_leaf_error_sum;
        stats.original_bytes = original_bytes;
        stats.encoded_bytes = encoded_bytes;
        stats.packed_encoded_bytes = packed_encoded_bytes;
        stats.fixed64_encoded_bytes = fixed64_encoded_bytes;
        stats.decompressed_depth_bytes = decompressed_depth_bytes;
        stats.packed_decompressed_working_set_bytes = packed_decompressed_working_set_bytes;
        stats.compression_ratio = encoded_bytes > 0 ? static_cast<float>(static_cast<double>(original_bytes) / static_cast<double>(encoded_bytes)) : 0.0f;
        stats.packed_compression_ratio = packed_encoded_bytes > 0 ? static_cast<float>(static_cast<double>(original_bytes) / static_cast<double>(packed_encoded_bytes)) : 0.0f;
        stats.fixed64_compression_ratio = fixed64_encoded_bytes > 0 ? static_cast<float>(static_cast<double>(original_bytes) / static_cast<double>(fixed64_encoded_bytes)) : 0.0f;
        stats.packed_decompressed_working_set_ratio = packed_decompressed_working_set_bytes > 0 ? static_cast<float>(static_cast<double>(original_bytes) / static_cast<double>(packed_decompressed_working_set_bytes)) : 0.0f;
        stats.packed_decode_valid = compact_overflow_count == 0 ? 1u : 0u;
        stats.max_error = max_error;
        stats.mean_error = 0.0f;
        stats.rmse_error = 0.0f;
    }

    const float* m_depth = nullptr;
    const float* m_second_depth = nullptr;
    SSTEncoderOptions m_options{};
    uint32_t m_width = 1;
    uint32_t m_height = 1;
    uint32_t m_tile_size = 128;
    uint32_t m_min_leaf_size = 1;
    float m_plane_error_threshold = 0.0015f;
    float m_constant_epsilon = 0.0005f;
    bool m_use_dual_layer = true;
    bool m_dual_has_slack = true;
    float m_dual_depth_slack = 0.0015f;
    bool m_dual_conservative = false;
    bool m_has_dual_max_leak = false;
    float m_dual_max_leak = 0.0f;
    float m_dual_max_leak_guard = 0.0f;
    float m_dual_visibility_tolerance = 0.0f;
    float m_shadow_bias = 0.0015f;
    uint32_t m_quantization_radius = 0;
    bool m_has_forced_leaf_error_cap = false;
    float m_forced_leaf_error_cap = 0.0f;
    bool m_forced_split_bias_fit = false;
    uint32_t m_max_tree_depth = 0;
    uint32_t m_branch_10bit_start_level = 0;
    std::vector<float> m_upper_depth;
    std::vector<Node> m_nodes;
    std::vector<uint32_t> m_node_levels;
    std::vector<uint32_t> m_tile_roots;
    uint32_t m_forced_leaf_node_count = 0;
    uint64_t m_forced_leaf_pixel_count = 0;
    double m_forced_leaf_error_sum = 0.0;
    float m_forced_leaf_max_error = 0.0f;
};
} // namespace

extern "C" {

SST_API int sst_encode(
    const float* depth,
    const float* second_depth,
    const SSTEncoderOptions* options,
    SSTEncoderOutput* output)
{
    if (!output)
        return 0;
    std::memset(output, 0, sizeof(SSTEncoderOutput));
    try
    {
        if (!options)
            throw std::runtime_error("options pointer is null");
        Encoder encoder(depth, second_depth, *options);
        encoder.encode(*output);
        return 1;
    }
    catch (const std::exception& exc)
    {
        output->error_message = duplicate_message(exc.what());
        return 0;
    }
    catch (...)
    {
        output->error_message = duplicate_message("unknown C++ SST encode failure");
        return 0;
    }
}

SST_API void sst_free_output(SSTEncoderOutput* output)
{
    if (!output)
        return;
    std::free(output->nodes);
    std::free(output->fixed64_nodes);
    std::free(output->compact_words);
    std::free(output->compact_roots);
    std::free(output->tile_roots);
    std::free(output->error_message);
    std::memset(output, 0, sizeof(SSTEncoderOutput));
}

SST_API const char* sst_version()
{
    return "static-shadow-tree-encoder-cpp-v1";
}

}
