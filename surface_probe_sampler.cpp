#include "surface_probe_sampler.h"

#include "cySampleElim.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
constexpr const char* kVersion =
    "cyCodeBase/62da186e0b2f2d3673d1f18386c66caf5798cd9b+parallel-adaptive-v3+parallel-repair-v2+parallel-filter-v1+parallel-octree-v1";

std::uint32_t parallel_worker_count(std::uint32_t count)
{
    return std::min(
        count,
        std::max(
            1u,
            static_cast<std::uint32_t>(std::thread::hardware_concurrency())));
}

template <typename Function>
void parallel_for_indices(std::uint32_t count, Function&& function)
{
    if (count == 0)
    {
        return;
    }
    const std::uint32_t thread_count =
        count < 1024 ? 1u : parallel_worker_count(count);
    if (thread_count == 1 || count < 1024)
    {
        for (std::uint32_t index = 0; index < count; ++index)
        {
            function(index);
        }
        return;
    }
    constexpr std::uint32_t kChunkSize = 64;
    std::atomic<std::uint32_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::uint32_t thread = 0; thread < thread_count; ++thread)
    {
        workers.emplace_back([&]() {
            try
            {
                while (!failed.load(std::memory_order_relaxed))
                {
                    const std::uint32_t begin = next.fetch_add(
                        kChunkSize, std::memory_order_relaxed);
                    if (begin >= count)
                    {
                        break;
                    }
                    const std::uint32_t end = std::min(count, begin + kChunkSize);
                    for (std::uint32_t index = begin; index < end; ++index)
                    {
                        function(index);
                    }
                }
            }
            catch (...)
            {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(exception_mutex);
                if (first_exception == nullptr)
                {
                    first_exception = std::current_exception();
                }
            }
        });
    }
    for (std::thread& worker : workers)
    {
        worker.join();
    }
    if (first_exception != nullptr)
    {
        std::rethrow_exception(first_exception);
    }
}

// Adaptive WSE partitions are few but individually expensive. The generic
// parallel loop deliberately keeps small index ranges serial because most of
// its callers perform very little work per index. Partition elimination is the
// opposite: 8-64 independent tasks can each take hundreds of milliseconds, so
// schedule them one at a time across the available workers.
template <typename Function>
void parallel_for_coarse_indices(std::uint32_t count, Function&& function)
{
    if (count == 0)
    {
        return;
    }
    const std::uint32_t thread_count = parallel_worker_count(count);
    if (thread_count == 1)
    {
        function(0);
        return;
    }
    std::atomic<std::uint32_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::uint32_t thread = 0; thread < thread_count; ++thread)
    {
        workers.emplace_back([&]() {
            try
            {
                while (!failed.load(std::memory_order_relaxed))
                {
                    const std::uint32_t index =
                        next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= count)
                    {
                        break;
                    }
                    function(index);
                }
            }
            catch (...)
            {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(exception_mutex);
                if (first_exception == nullptr)
                {
                    first_exception = std::current_exception();
                }
            }
        });
    }
    for (std::thread& worker : workers)
    {
        worker.join();
    }
    if (first_exception != nullptr)
    {
        std::rethrow_exception(first_exception);
    }
}

template <typename Value, typename Function>
std::vector<std::vector<Value>> parallel_collect_indices(
    std::uint32_t count,
    Function&& function)
{
    if (count == 0)
    {
        return {};
    }
    const std::uint32_t thread_count =
        count < 1024 ? 1u : parallel_worker_count(count);
    std::vector<std::vector<Value>> outputs(thread_count);
    std::atomic<bool> failed{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::uint32_t thread = 0; thread < thread_count; ++thread)
    {
        workers.emplace_back([&, thread]() {
            try
            {
                for (std::uint32_t index = thread;
                     index < count && !failed.load(std::memory_order_relaxed);
                     index += thread_count)
                {
                    function(index, outputs[thread]);
                }
            }
            catch (...)
            {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(exception_mutex);
                if (first_exception == nullptr)
                {
                    first_exception = std::current_exception();
                }
            }
        });
    }
    for (std::thread& worker : workers)
    {
        worker.join();
    }
    if (first_exception != nullptr)
    {
        std::rethrow_exception(first_exception);
    }
    return outputs;
}

struct SurfaceSample
{
    float position[3];
    float normal[3];
    std::uint32_t source_index;

    explicit SurfaceSample(float value = 0.0f)
        : position{value, value, value}, normal{0.0f, 0.0f, 0.0f}, source_index(0)
    {
    }

    float operator[](int dimension) const
    {
        return position[dimension];
    }

    float& operator[](int dimension)
    {
        return position[dimension];
    }

    float LengthSquared() const
    {
        return position[0] * position[0] + position[1] * position[1] +
            position[2] * position[2];
    }
};

SurfaceSample operator-(const SurfaceSample& lhs, const SurfaceSample& rhs)
{
    SurfaceSample result;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        result.position[dimension] =
            lhs.position[dimension] - rhs.position[dimension];
    }
    return result;
}

void set_error(char* destination, std::size_t capacity, const std::string& message)
{
    if (destination == nullptr || capacity == 0)
    {
        return;
    }
    const std::size_t count = std::min(capacity - 1, message.size());
    std::memcpy(destination, message.data(), count);
    destination[count] = '\0';
}

float dot3(const float* lhs, const float* rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

void validate_options(
    std::uint32_t input_count,
    std::uint32_t output_count,
    const SurfaceProbeWSEOptions& options)
{
    if (input_count == 0)
    {
        throw std::invalid_argument("input_count must be positive");
    }
    if (output_count == 0 || output_count > input_count)
    {
        throw std::invalid_argument(
            "output_count must be positive and no larger than input_count");
    }
    if (!std::isfinite(options.surface_area) || options.surface_area <= 0.0f)
    {
        throw std::invalid_argument("surface_area must be finite and positive");
    }
    if (!std::isfinite(options.normal_cosine_threshold) ||
        options.normal_cosine_threshold < -1.0f ||
        options.normal_cosine_threshold > 1.0f)
    {
        throw std::invalid_argument(
            "normal_cosine_threshold must be finite and within [-1, 1]");
    }
    if (!std::isfinite(options.plane_distance_scale) ||
        options.plane_distance_scale < 0.0f)
    {
        throw std::invalid_argument(
            "plane_distance_scale must be finite and non-negative");
    }
}

struct GridKey
{
    std::uint32_t instance;
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;

    bool operator==(const GridKey& other) const
    {
        return instance == other.instance && x == other.x && y == other.y &&
            z == other.z;
    }
};

struct GridKeyHash
{
    std::size_t operator()(const GridKey& key) const
    {
        std::size_t value = static_cast<std::size_t>(key.instance);
        const auto combine = [&value](std::uint64_t component) {
            value ^= static_cast<std::size_t>(component) +
                0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        };
        combine(static_cast<std::uint64_t>(key.x));
        combine(static_cast<std::uint64_t>(key.y));
        combine(static_cast<std::uint64_t>(key.z));
        return value;
    }
};

struct ExactSurfaceKey
{
    std::array<std::uint32_t, 6> values{};

    bool operator==(const ExactSurfaceKey& other) const
    {
        return values == other.values;
    }
};

struct ExactSurfaceKeyHash
{
    std::size_t operator()(const ExactSurfaceKey& key) const
    {
        std::size_t value = 0;
        for (const std::uint32_t component : key.values)
        {
            value ^= static_cast<std::size_t>(component) +
                0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        }
        return value;
    }
};

ExactSurfaceKey exact_surface_key(
    const float* positions,
    const float* normals,
    std::uint32_t index)
{
    ExactSurfaceKey key;
    const float* position = positions + static_cast<std::size_t>(index) * 3;
    const float* normal = normals + static_cast<std::size_t>(index) * 3;
    std::memcpy(key.values.data(), position, 3 * sizeof(float));
    std::memcpy(key.values.data() + 3, normal, 3 * sizeof(float));
    return key;
}

class PointGrid
{
public:
    PointGrid(
        const float* positions,
        const std::uint32_t* instances,
        std::uint32_t count,
        float cell_size)
        : positions_(positions),
          instances_(instances),
          cell_size_(std::max(cell_size, 1.0e-8f)),
          inverse_cell_size_(1.0f / cell_size_)
    {
        for (std::uint32_t index = 0; index < count; ++index)
        {
            cells_[key(index)].push_back(index);
        }
    }

    template <typename Visitor>
    void visit_nearby(
        const float* position,
        std::uint32_t instance,
        Visitor&& visitor) const
    {
        const std::int64_t x = coordinate(position[0]);
        const std::int64_t y = coordinate(position[1]);
        const std::int64_t z = coordinate(position[2]);
        for (std::int64_t dz = -1; dz <= 1; ++dz)
        {
            for (std::int64_t dy = -1; dy <= 1; ++dy)
            {
                for (std::int64_t dx = -1; dx <= 1; ++dx)
                {
                    const auto found = cells_.find(
                        GridKey{instance, x + dx, y + dy, z + dz});
                    if (found == cells_.end())
                    {
                        continue;
                    }
                    for (const std::uint32_t index : found->second)
                    {
                        visitor(index);
                    }
                }
            }
        }
    }

private:
    std::int64_t coordinate(float value) const
    {
        return static_cast<std::int64_t>(
            std::floor(static_cast<double>(value) * inverse_cell_size_));
    }

    GridKey key(std::uint32_t index) const
    {
        const float* position = positions_ + static_cast<std::size_t>(index) * 3;
        return GridKey{
            instances_ != nullptr ? instances_[index] : 0,
            coordinate(position[0]),
            coordinate(position[1]),
            coordinate(position[2]),
        };
    }

    const float* positions_;
    const std::uint32_t* instances_;
    float cell_size_;
    float inverse_cell_size_;
    std::unordered_map<GridKey, std::vector<std::uint32_t>, GridKeyHash> cells_;
};

float gather_weight(
    const float* query_position,
    const float* query_normal,
    const float* probe_position,
    const float* probe_normal,
    float radius,
    float normal_cosine_threshold,
    float weight_epsilon)
{
    const float delta[3] = {
        probe_position[0] - query_position[0],
        probe_position[1] - query_position[1],
        probe_position[2] - query_position[2],
    };
    const float distance_squared = dot3(delta, delta);
    const float radius_squared = radius * radius;
    if (distance_squared > radius_squared)
    {
        return 0.0f;
    }
    const float normal_cosine = dot3(query_normal, probe_normal);
    if (normal_cosine < normal_cosine_threshold)
    {
        return 0.0f;
    }
    const float plane_sigma = std::max(radius * 0.25f, 1.0e-6f);
    const float plane_distance = std::abs(dot3(delta, query_normal));
    const float plane_ratio = plane_distance / plane_sigma;
    const float plane_weight = std::exp(-0.5f * plane_ratio * plane_ratio);
    const float spatial = std::max(1.0f - distance_squared / radius_squared, 0.0f);
    const float weight = spatial * spatial * plane_weight *
        std::max(normal_cosine, 0.0f);
    return weight > weight_epsilon ? weight : 0.0f;
}

float adaptive_elimination_weight(
    const float* query_position,
    const float* query_normal,
    const float* neighbor_position,
    const float* neighbor_normal,
    float local_d_max,
    float normal_cosine_threshold,
    float plane_distance_scale)
{
    const float delta[3] = {
        neighbor_position[0] - query_position[0],
        neighbor_position[1] - query_position[1],
        neighbor_position[2] - query_position[2],
    };
    const float distance_squared = dot3(delta, delta);
    const float maximum_distance_squared = local_d_max * local_d_max;
    if (distance_squared <= 1.0e-20f ||
        distance_squared >= maximum_distance_squared ||
        dot3(query_normal, neighbor_normal) < normal_cosine_threshold)
    {
        return 0.0f;
    }
    const float plane_limit = std::max(
        local_d_max * plane_distance_scale, 1.0e-8f);
    if (std::abs(dot3(delta, query_normal)) > plane_limit ||
        std::abs(dot3(delta, neighbor_normal)) > plane_limit)
    {
        return 0.0f;
    }
    const float normalized_distance =
        std::sqrt(distance_squared) / local_d_max;
    const float base = std::max(1.0f - normalized_distance, 0.0f);
    const float square = base * base;
    return square * square * square * square;
}

using SampleIndices = std::vector<std::uint32_t>;

struct AdaptiveSubsetProfile
{
    double pack_ms = 0.0;
    double grid_ms = 0.0;
    double weights_ms = 0.0;
    double heap_ms = 0.0;
};

struct AdaptivePartitionStageProfile
{
    double partition_ms = 0.0;
    double eliminate_wall_ms = 0.0;
    double pack_cpu_ms = 0.0;
    double grid_cpu_ms = 0.0;
    double weights_cpu_ms = 0.0;
    double heap_cpu_ms = 0.0;
    std::uint32_t input_count = 0;
    std::uint32_t output_count = 0;
    std::uint32_t partition_count = 0;
};

void build_balanced_partitions_recursive(
    const float* positions,
    SampleIndices& indices,
    std::size_t begin,
    std::size_t end,
    std::uint32_t leaf_count,
    std::uint32_t output_offset,
    std::uint32_t worker_budget,
    std::vector<SampleIndices>& output)
{
    if (begin == end)
    {
        return;
    }
    if (leaf_count <= 1 || end - begin <= 1)
    {
        output[output_offset] = SampleIndices(
            indices.begin() + static_cast<std::ptrdiff_t>(begin),
            indices.begin() + static_cast<std::ptrdiff_t>(end));
        return;
    }
    float minimum[3] = {
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    float maximum[3] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (std::size_t offset = begin; offset < end; ++offset)
    {
        const std::uint32_t index = indices[offset];
        const float* position = positions + static_cast<std::size_t>(index) * 3;
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }
    int split_axis = 0;
    for (int axis = 1; axis < 3; ++axis)
    {
        if (maximum[axis] - minimum[axis] >
            maximum[split_axis] - minimum[split_axis])
        {
            split_axis = axis;
        }
    }
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(
        indices.begin() + static_cast<std::ptrdiff_t>(begin),
        indices.begin() + static_cast<std::ptrdiff_t>(middle),
        indices.begin() + static_cast<std::ptrdiff_t>(end),
        [positions, split_axis](std::uint32_t lhs, std::uint32_t rhs) {
            const float left = positions[static_cast<std::size_t>(lhs) * 3 + split_axis];
            const float right = positions[static_cast<std::size_t>(rhs) * 3 + split_axis];
            return left != right ? left < right : lhs < rhs;
        });
    const std::uint32_t left_leaves = leaf_count / 2;
    const std::uint32_t right_leaves = leaf_count - left_leaves;
    if (worker_budget <= 1)
    {
        build_balanced_partitions_recursive(
            positions,
            indices,
            begin,
            middle,
            left_leaves,
            output_offset,
            1,
            output);
        build_balanced_partitions_recursive(
            positions,
            indices,
            middle,
            end,
            right_leaves,
            output_offset + left_leaves,
            1,
            output);
        return;
    }

    const std::uint32_t left_workers = std::max(
        1u,
        std::min(
            left_leaves,
            static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(worker_budget) * left_leaves /
                leaf_count)));
    const std::uint32_t right_workers = std::max(
        1u, std::min(right_leaves, worker_budget - left_workers));
    std::exception_ptr left_exception;
    std::thread left_worker([&]() {
        try
        {
            build_balanced_partitions_recursive(
                positions,
                indices,
                begin,
                middle,
                left_leaves,
                output_offset,
                left_workers,
                output);
        }
        catch (...)
        {
            left_exception = std::current_exception();
        }
    });
    std::exception_ptr right_exception;
    try
    {
        build_balanced_partitions_recursive(
            positions,
            indices,
            middle,
            end,
            right_leaves,
            output_offset + left_leaves,
            right_workers,
            output);
    }
    catch (...)
    {
        right_exception = std::current_exception();
    }
    left_worker.join();
    if (left_exception != nullptr)
    {
        std::rethrow_exception(left_exception);
    }
    if (right_exception != nullptr)
    {
        std::rethrow_exception(right_exception);
    }
}

void build_balanced_partitions(
    const float* positions,
    SampleIndices indices,
    std::uint32_t leaf_count,
    std::vector<SampleIndices>& output)
{
    if (indices.empty())
    {
        return;
    }
    leaf_count = std::max(
        1u,
        std::min(
            leaf_count, static_cast<std::uint32_t>(indices.size())));
    output.clear();
    output.resize(leaf_count);
    build_balanced_partitions_recursive(
        positions,
        indices,
        0,
        indices.size(),
        leaf_count,
        0,
        parallel_worker_count(leaf_count),
        output);
}

std::vector<std::uint32_t> allocate_partition_targets(
    const std::vector<SampleIndices>& partitions,
    const std::vector<float>& partition_masses,
    std::uint32_t total_target)
{
    double total_mass = 0.0;
    for (const SampleIndices& partition : partitions)
    {
        for (const std::uint32_t index : partition)
        {
            total_mass += partition_masses[index];
        }
    }
    std::vector<std::uint32_t> targets(partitions.size(), 0);
    std::vector<std::pair<double, std::uint32_t>> remainders;
    remainders.reserve(partitions.size());
    std::uint32_t assigned = 0;
    for (std::uint32_t partition = 0;
         partition < static_cast<std::uint32_t>(partitions.size());
         ++partition)
    {
        double mass = 0.0;
        for (const std::uint32_t index : partitions[partition])
        {
            mass += partition_masses[index];
        }
        const double exact = static_cast<double>(total_target) * mass /
            std::max(total_mass, 1.0e-30);
        targets[partition] = std::min(
            static_cast<std::uint32_t>(std::floor(exact)),
            static_cast<std::uint32_t>(partitions[partition].size()));
        assigned += targets[partition];
        remainders.emplace_back(exact - std::floor(exact), partition);
    }
    std::sort(
        remainders.begin(),
        remainders.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first != rhs.first
                ? lhs.first > rhs.first
                : lhs.second < rhs.second;
        });
    for (std::uint32_t offset = 0; assigned < total_target; ++offset)
    {
        const std::uint32_t partition =
            remainders[offset % remainders.size()].second;
        if (targets[partition] < partitions[partition].size())
        {
            ++targets[partition];
            ++assigned;
        }
    }
    return targets;
}

SampleIndices eliminate_adaptive_subset(
    const float* positions,
    const float* normals,
    const std::vector<float>& global_local_d_max,
    const SampleIndices& input_indices,
    std::uint32_t output_count,
    float normal_cosine_threshold,
    float plane_distance_scale,
    bool parallel_weights,
    AdaptiveSubsetProfile* profile = nullptr)
{
    using ProfileClock = std::chrono::steady_clock;
    ProfileClock::time_point phase_start;
    if (profile != nullptr)
    {
        *profile = {};
        phase_start = ProfileClock::now();
    }
    const std::uint32_t input_count =
        static_cast<std::uint32_t>(input_indices.size());
    output_count = std::min(output_count, input_count);
    if (output_count == input_count)
    {
        return input_indices;
    }
    if (output_count == 0)
    {
        return {};
    }
    std::vector<float> local_positions(static_cast<std::size_t>(input_count) * 3);
    std::vector<float> local_normals(static_cast<std::size_t>(input_count) * 3);
    std::vector<float> local_d_max(input_count);
    float maximum_d_max = 1.0e-6f;
    for (std::uint32_t local = 0; local < input_count; ++local)
    {
        const std::uint32_t global = input_indices[local];
        std::copy_n(
            positions + static_cast<std::size_t>(global) * 3,
            3,
            local_positions.data() + static_cast<std::size_t>(local) * 3);
        std::copy_n(
            normals + static_cast<std::size_t>(global) * 3,
            3,
            local_normals.data() + static_cast<std::size_t>(local) * 3);
        local_d_max[local] = global_local_d_max[global];
        maximum_d_max = std::max(maximum_d_max, local_d_max[local]);
    }
    if (profile != nullptr)
    {
        const auto now = ProfileClock::now();
        profile->pack_ms = std::chrono::duration<double, std::milli>(
            now - phase_start).count();
        phase_start = now;
    }
    const PointGrid grid(
        local_positions.data(), nullptr, input_count, maximum_d_max);
    if (profile != nullptr)
    {
        const auto now = ProfileClock::now();
        profile->grid_ms = std::chrono::duration<double, std::milli>(
            now - phase_start).count();
        phase_start = now;
    }
    std::vector<float> weights(input_count, 0.0f);
    const auto compute_weight = [&](std::uint32_t index) {
        const float* position =
            local_positions.data() + static_cast<std::size_t>(index) * 3;
        const float* normal =
            local_normals.data() + static_cast<std::size_t>(index) * 3;
        double weight_sum = 0.0;
        grid.visit_nearby(position, 0, [&](std::uint32_t neighbor) {
            if (neighbor == index)
            {
                return;
            }
            weight_sum += adaptive_elimination_weight(
                position,
                normal,
                local_positions.data() + static_cast<std::size_t>(neighbor) * 3,
                local_normals.data() + static_cast<std::size_t>(neighbor) * 3,
                local_d_max[index],
                normal_cosine_threshold,
                plane_distance_scale);
        });
        weights[index] = static_cast<float>(weight_sum);
    };
    if (parallel_weights)
    {
        parallel_for_indices(input_count, compute_weight);
    }
    else
    {
        for (std::uint32_t index = 0; index < input_count; ++index)
        {
            compute_weight(index);
        }
    }
    if (profile != nullptr)
    {
        const auto now = ProfileClock::now();
        profile->weights_ms = std::chrono::duration<double, std::milli>(
            now - phase_start).count();
        phase_start = now;
    }

    cy::MaxHeap<float, std::uint32_t> heap;
    heap.SetDataPointer(weights.data(), input_count);
    heap.Build();
    while (heap.NumItemsInHeap() > output_count)
    {
        const std::uint32_t removed = heap.GetTopItemID();
        heap.Pop();
        const float* removed_position = local_positions.data() +
            static_cast<std::size_t>(removed) * 3;
        const float* removed_normal = local_normals.data() +
            static_cast<std::size_t>(removed) * 3;
        grid.visit_nearby(removed_position, 0, [&](std::uint32_t neighbor) {
            if (neighbor == removed || !heap.IsInHeap(neighbor))
            {
                return;
            }
            const float contribution = adaptive_elimination_weight(
                local_positions.data() + static_cast<std::size_t>(neighbor) * 3,
                local_normals.data() + static_cast<std::size_t>(neighbor) * 3,
                removed_position,
                removed_normal,
                local_d_max[neighbor],
                normal_cosine_threshold,
                plane_distance_scale);
            if (contribution <= 0.0f)
            {
                return;
            }
            weights[neighbor] = std::max(
                0.0f, weights[neighbor] - contribution);
            heap.MoveItemDown(neighbor);
        });
    }
    SampleIndices output(output_count);
    for (std::uint32_t index = 0; index < output_count; ++index)
    {
        output[index] = input_indices[heap.GetIDFromHeap(index)];
    }
    std::sort(output.begin(), output.end());
    if (profile != nullptr)
    {
        profile->heap_ms = std::chrono::duration<double, std::milli>(
            ProfileClock::now() - phase_start).count();
    }
    return output;
}

SampleIndices eliminate_adaptive_partition_stage(
    const float* positions,
    const float* normals,
    const std::vector<float>& local_d_max,
    const std::vector<float>& partition_masses,
    const SampleIndices& input_indices,
    std::uint32_t output_count,
    std::uint32_t requested_partition_count,
    float normal_cosine_threshold,
    float plane_distance_scale,
    AdaptivePartitionStageProfile* profile = nullptr)
{
    using ProfileClock = std::chrono::steady_clock;
    ProfileClock::time_point stage_start;
    if (profile != nullptr)
    {
        *profile = {};
        profile->input_count = static_cast<std::uint32_t>(input_indices.size());
        profile->output_count = output_count;
        stage_start = ProfileClock::now();
    }
    const std::uint32_t partition_count = std::max(
        1u,
        std::min(
            requested_partition_count,
            std::min(
                output_count,
                static_cast<std::uint32_t>(input_indices.size()))));
    std::vector<SampleIndices> partitions;
    partitions.reserve(partition_count);
    build_balanced_partitions(
        positions, input_indices, partition_count, partitions);
    const std::vector<std::uint32_t> targets =
        allocate_partition_targets(partitions, partition_masses, output_count);
    if (profile != nullptr)
    {
        const auto now = ProfileClock::now();
        profile->partition_ms = std::chrono::duration<double, std::milli>(
            now - stage_start).count();
        profile->partition_count =
            static_cast<std::uint32_t>(partitions.size());
        stage_start = now;
    }
    std::vector<SampleIndices> outputs(partitions.size());
    std::vector<AdaptiveSubsetProfile> subset_profiles(
        profile != nullptr ? partitions.size() : 0);
    parallel_for_coarse_indices(
        static_cast<std::uint32_t>(partitions.size()),
        [&](std::uint32_t partition) {
            outputs[partition] = eliminate_adaptive_subset(
                positions,
                normals,
                local_d_max,
                partitions[partition],
                targets[partition],
                normal_cosine_threshold,
                plane_distance_scale,
                false,
                profile != nullptr ? &subset_profiles[partition] : nullptr);
        });
    SampleIndices output;
    output.reserve(output_count);
    for (SampleIndices& partition_output : outputs)
    {
        output.insert(
            output.end(), partition_output.begin(), partition_output.end());
    }
    if (output.size() != output_count)
    {
        throw std::runtime_error(
            "parallel adaptive WSE partition count mismatch");
    }
    if (profile != nullptr)
    {
        profile->eliminate_wall_ms =
            std::chrono::duration<double, std::milli>(
                ProfileClock::now() - stage_start).count();
        for (const AdaptiveSubsetProfile& subset : subset_profiles)
        {
            profile->pack_cpu_ms += subset.pack_ms;
            profile->grid_cpu_ms += subset.grid_ms;
            profile->weights_cpu_ms += subset.weights_ms;
            profile->heap_cpu_ms += subset.heap_ms;
        }
    }
    return output;
}

SampleIndices eliminate_adaptive_parallel(
    const float* positions,
    const float* normals,
    const std::vector<float>& local_d_max,
    const std::vector<float>& partition_masses,
    std::uint32_t input_count,
    std::uint32_t output_count,
    float normal_cosine_threshold,
    float plane_distance_scale,
    SurfaceProbeAdaptiveWSEProfile* profile)
{
    SampleIndices active(input_count);
    for (std::uint32_t index = 0; index < input_count; ++index)
    {
        active[index] = index;
    }
    constexpr std::uint32_t kParallelThreshold = 8192;
    if (input_count < kParallelThreshold ||
        input_count <= output_count + output_count / 2)
    {
        AdaptiveSubsetProfile final_profile;
        SampleIndices result = eliminate_adaptive_subset(
            positions,
            normals,
            local_d_max,
            active,
            output_count,
            normal_cosine_threshold,
            plane_distance_scale,
            true,
            profile != nullptr ? &final_profile : nullptr);
        if (profile != nullptr)
        {
            profile->parallel_path = 0;
            profile->final_output_count = output_count;
            profile->final_pack_ms = final_profile.pack_ms;
            profile->final_grid_ms = final_profile.grid_ms;
            profile->final_weights_ms = final_profile.weights_ms;
            profile->final_heap_ms = final_profile.heap_ms;
        }
        return result;
    }
    const auto intermediate_target = [output_count](std::uint32_t current) {
        const std::uint64_t excess = current - output_count;
        return output_count + static_cast<std::uint32_t>((excess + 4) / 5);
    };
    AdaptivePartitionStageProfile stage1_profile;
    active = eliminate_adaptive_partition_stage(
        positions,
        normals,
        local_d_max,
        partition_masses,
        active,
        intermediate_target(static_cast<std::uint32_t>(active.size())),
        64,
        normal_cosine_threshold,
        plane_distance_scale,
        profile != nullptr ? &stage1_profile : nullptr);
    // The first stage establishes the requested spatial target mass. Later
    // stages must preserve that survivor distribution instead of multiplying
    // the mass field into it again.
    const std::vector<float> survivor_masses(input_count, 1.0f);
    AdaptivePartitionStageProfile stage2_profile;
    active = eliminate_adaptive_partition_stage(
        positions,
        normals,
        local_d_max,
        survivor_masses,
        active,
        intermediate_target(static_cast<std::uint32_t>(active.size())),
        8,
        normal_cosine_threshold,
        plane_distance_scale,
        profile != nullptr ? &stage2_profile : nullptr);
    AdaptiveSubsetProfile final_profile;
    SampleIndices result = eliminate_adaptive_subset(
        positions,
        normals,
        local_d_max,
        active,
        output_count,
        normal_cosine_threshold,
        plane_distance_scale,
        true,
        profile != nullptr ? &final_profile : nullptr);
    if (profile != nullptr)
    {
        profile->parallel_path = 1;
        profile->stage1_input_count = stage1_profile.input_count;
        profile->stage1_output_count = stage1_profile.output_count;
        profile->stage2_output_count = stage2_profile.output_count;
        profile->final_output_count = output_count;
        profile->stage1_partition_count = stage1_profile.partition_count;
        profile->stage2_partition_count = stage2_profile.partition_count;
        profile->stage1_partition_ms = stage1_profile.partition_ms;
        profile->stage1_eliminate_wall_ms = stage1_profile.eliminate_wall_ms;
        profile->stage1_pack_cpu_ms = stage1_profile.pack_cpu_ms;
        profile->stage1_grid_cpu_ms = stage1_profile.grid_cpu_ms;
        profile->stage1_weights_cpu_ms = stage1_profile.weights_cpu_ms;
        profile->stage1_heap_cpu_ms = stage1_profile.heap_cpu_ms;
        profile->stage2_partition_ms = stage2_profile.partition_ms;
        profile->stage2_eliminate_wall_ms = stage2_profile.eliminate_wall_ms;
        profile->stage2_pack_cpu_ms = stage2_profile.pack_cpu_ms;
        profile->stage2_grid_cpu_ms = stage2_profile.grid_cpu_ms;
        profile->stage2_weights_cpu_ms = stage2_profile.weights_cpu_ms;
        profile->stage2_heap_cpu_ms = stage2_profile.heap_cpu_ms;
        profile->final_pack_ms = final_profile.pack_ms;
        profile->final_grid_ms = final_profile.grid_ms;
        profile->final_weights_ms = final_profile.weights_ms;
        profile->final_heap_ms = final_profile.heap_ms;
    }
    return result;
}

struct GatherStats
{
    std::uint32_t count = 0;
    float weight_sum = 0.0f;
    float ess = 0.0f;
};

struct WeightedProbe
{
    float weight;
    std::uint32_t index;
};

std::vector<float> select_gather_weights(
    std::vector<WeightedProbe>& candidates)
{
    constexpr std::size_t kCandidatePool = 96;
    constexpr std::size_t kMaxGather = 32;
    const auto stronger = [](const WeightedProbe& lhs, const WeightedProbe& rhs) {
        if (lhs.weight != rhs.weight)
        {
            return lhs.weight > rhs.weight;
        }
        return lhs.index < rhs.index;
    };
    if (candidates.size() > kCandidatePool)
    {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + kCandidatePool,
            candidates.end(),
            stronger);
        candidates.resize(kCandidatePool);
    }
    std::sort(candidates.begin(), candidates.end(), stronger);
    std::vector<float> weights;
    weights.reserve(std::min(kMaxGather, candidates.size()));
    for (const WeightedProbe& candidate : candidates)
    {
        weights.push_back(candidate.weight);
        if (weights.size() == kMaxGather)
        {
            break;
        }
    }
    return weights;
}

GatherStats summarize_weights(std::vector<float>& weights)
{
    constexpr std::size_t kMaxGather = 32;
    if (weights.size() > kMaxGather)
    {
        std::nth_element(
            weights.begin(),
            weights.begin() + static_cast<std::ptrdiff_t>(kMaxGather),
            weights.end(),
            std::greater<float>());
        weights.resize(kMaxGather);
    }
    double sum = 0.0;
    double squared_sum = 0.0;
    for (const float weight : weights)
    {
        sum += weight;
        squared_sum += static_cast<double>(weight) * weight;
    }
    GatherStats result;
    result.count = static_cast<std::uint32_t>(weights.size());
    result.weight_sum = static_cast<float>(sum);
    result.ess = squared_sum > 0.0
        ? static_cast<float>(sum * sum / squared_sum)
        : 0.0f;
    return result;
}

struct RepairHeapEntry
{
    std::uint64_t score;
    std::uint32_t candidate;
    std::uint32_t coverage_slot;
};

struct RepairHeapCompare
{
    bool operator()(const RepairHeapEntry& lhs, const RepairHeapEntry& rhs) const
    {
        if (lhs.score != rhs.score)
        {
            return lhs.score < rhs.score;
        }
        return lhs.candidate > rhs.candidate;
    }
};

struct PointOctreeBuildNode
{
    std::array<std::unique_ptr<PointOctreeBuildNode>, 8> children;
    std::size_t begin = 0;
    std::size_t end = 0;
    std::uint32_t child_mask = 0;
    bool leaf = false;
};

std::uint8_t point_octant(
    const float* positions,
    std::uint32_t index,
    const std::array<double, 3>& center)
{
    const float* position = positions + static_cast<std::size_t>(index) * 3;
    std::uint8_t octant = 0;
    for (std::uint32_t axis = 0; axis < 3; ++axis)
    {
        if (position[axis] > static_cast<float>(center[axis]))
        {
            octant |= static_cast<std::uint8_t>(1u << axis);
        }
    }
    return octant;
}

void build_point_octree_node(
    const float* positions,
    std::vector<std::uint32_t>& indices,
    std::vector<std::uint32_t>& scratch,
    std::vector<std::uint8_t>& octant_codes,
    PointOctreeBuildNode& node,
    const std::array<double, 3>& center,
    double extent,
    std::uint32_t depth,
    std::uint32_t leaf_capacity,
    std::uint32_t max_depth,
    std::uint32_t worker_budget)
{
    const std::size_t point_count = node.end - node.begin;
    if (point_count <= leaf_capacity || depth >= max_depth)
    {
        node.leaf = true;
        return;
    }

    std::array<std::size_t, 8> counts{};
    for (std::size_t offset = node.begin; offset < node.end; ++offset)
    {
        const std::uint8_t octant = point_octant(
            positions, indices[offset], center);
        octant_codes[offset] = octant;
        ++counts[octant];
    }
    std::array<std::size_t, 8> starts{};
    std::size_t cursor = node.begin;
    for (std::uint32_t octant = 0; octant < 8; ++octant)
    {
        starts[octant] = cursor;
        cursor += counts[octant];
    }
    std::array<std::size_t, 8> write_offsets = starts;
    for (std::size_t offset = node.begin; offset < node.end; ++offset)
    {
        const std::uint8_t octant = octant_codes[offset];
        scratch[write_offsets[octant]++] = indices[offset];
    }
    std::copy(
        scratch.begin() + static_cast<std::ptrdiff_t>(node.begin),
        scratch.begin() + static_cast<std::ptrdiff_t>(node.end),
        indices.begin() + static_cast<std::ptrdiff_t>(node.begin));

    std::vector<std::uint32_t> present_octants;
    present_octants.reserve(8);
    for (std::uint32_t octant = 0; octant < 8; ++octant)
    {
        if (counts[octant] == 0)
        {
            continue;
        }
        present_octants.push_back(octant);
        node.child_mask |= 1u << octant;
        auto child = std::make_unique<PointOctreeBuildNode>();
        child->begin = starts[octant];
        child->end = starts[octant] + counts[octant];
        node.children[octant] = std::move(child);
    }

    const double child_extent = extent * 0.5;
    const auto build_child = [&](std::uint32_t slot, std::uint32_t budget) {
        const std::uint32_t octant = present_octants[slot];
        std::array<double, 3> child_center{};
        for (std::uint32_t axis = 0; axis < 3; ++axis)
        {
            child_center[axis] = center[axis] +
                ((octant & (1u << axis)) ? child_extent : -child_extent);
        }
        build_point_octree_node(
            positions,
            indices,
            scratch,
            octant_codes,
            *node.children[octant],
            child_center,
            child_extent,
            depth + 1,
            leaf_capacity,
            max_depth,
            budget);
    };

    if (present_octants.size() == 1)
    {
        build_child(0, worker_budget);
        return;
    }
    constexpr std::size_t kParallelPointThreshold = 16384;
    if (worker_budget <= 1 || point_count < kParallelPointThreshold)
    {
        for (std::uint32_t slot = 0;
             slot < static_cast<std::uint32_t>(present_octants.size());
             ++slot)
        {
            build_child(slot, 1);
        }
        return;
    }

    std::vector<std::uint32_t> child_budgets(present_octants.size(), 1u);
    if (worker_budget >= present_octants.size())
    {
        std::uint32_t remaining = worker_budget -
            static_cast<std::uint32_t>(present_octants.size());
        while (remaining > 0)
        {
            std::uint32_t best_slot = 0;
            double best_load = -1.0;
            for (std::uint32_t slot = 0;
                 slot < static_cast<std::uint32_t>(present_octants.size());
                 ++slot)
            {
                const std::uint32_t octant = present_octants[slot];
                const double load = static_cast<double>(counts[octant]) /
                    child_budgets[slot];
                if (load > best_load)
                {
                    best_load = load;
                    best_slot = slot;
                }
            }
            ++child_budgets[best_slot];
            --remaining;
        }
    }

    const std::uint32_t thread_count = std::min(
        worker_budget,
        static_cast<std::uint32_t>(present_octants.size()));
    std::atomic<std::uint32_t> next_child{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    const auto worker = [&]() {
        try
        {
            while (!failed.load(std::memory_order_relaxed))
            {
                const std::uint32_t slot = next_child.fetch_add(
                    1, std::memory_order_relaxed);
                if (slot >= present_octants.size())
                {
                    break;
                }
                build_child(slot, child_budgets[slot]);
            }
        }
        catch (...)
        {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (first_exception == nullptr)
            {
                first_exception = std::current_exception();
            }
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(thread_count - 1);
    for (std::uint32_t thread = 1; thread < thread_count; ++thread)
    {
        workers.emplace_back(worker);
    }
    worker();
    for (std::thread& thread : workers)
    {
        thread.join();
    }
    if (first_exception != nullptr)
    {
        std::rethrow_exception(first_exception);
    }
}

void flatten_point_octree(
    const PointOctreeBuildNode& node,
    const std::vector<std::uint32_t>& indices,
    std::uint32_t flat_index,
    std::vector<std::array<std::uint32_t, 4>>& nodes,
    std::vector<std::uint32_t>& probe_order)
{
    if (node.leaf)
    {
        if (probe_order.size() > std::numeric_limits<std::uint32_t>::max() -
                (node.end - node.begin))
        {
            throw std::overflow_error("point octree probe order exceeds uint32");
        }
        const std::uint32_t probe_start =
            static_cast<std::uint32_t>(probe_order.size());
        probe_order.insert(
            probe_order.end(),
            indices.begin() + static_cast<std::ptrdiff_t>(node.begin),
            indices.begin() + static_cast<std::ptrdiff_t>(node.end));
        nodes[flat_index] = {
            0,
            0,
            probe_start,
            static_cast<std::uint32_t>(node.end - node.begin)};
        return;
    }

    const std::uint32_t child_count = static_cast<std::uint32_t>(
        std::count_if(
            node.children.begin(),
            node.children.end(),
            [](const auto& child) { return child != nullptr; }));
    if (nodes.size() >
        std::numeric_limits<std::uint32_t>::max() - child_count)
    {
        throw std::overflow_error("point octree node count exceeds uint32");
    }
    const std::uint32_t child_base =
        static_cast<std::uint32_t>(nodes.size());
    nodes.resize(nodes.size() + child_count);
    nodes[flat_index] = {child_base, node.child_mask, 0, 0};
    std::uint32_t compact_index = 0;
    for (std::uint32_t octant = 0; octant < 8; ++octant)
    {
        if (node.children[octant] == nullptr)
        {
            continue;
        }
        flatten_point_octree(
            *node.children[octant],
            indices,
            child_base + compact_index,
            nodes,
            probe_order);
        ++compact_index;
    }
}
}

extern "C" int surface_probe_wse_eliminate(
    const float* positions,
    const float* normals,
    std::uint32_t input_count,
    std::uint32_t output_count,
    const SurfaceProbeWSEOptions* options,
    std::uint32_t* output_indices,
    float* output_poisson_radius,
    char* error_message,
    std::size_t error_message_capacity)
{
    try
    {
        if (positions == nullptr || normals == nullptr || options == nullptr ||
            output_indices == nullptr || output_poisson_radius == nullptr)
        {
            throw std::invalid_argument("surface probe WSE received a null pointer");
        }
        validate_options(input_count, output_count, *options);

        if (output_count == input_count)
        {
            for (std::uint32_t index = 0; index < input_count; ++index)
            {
                output_indices[index] = index;
            }
            *output_poisson_radius = std::sqrt(
                options->surface_area / static_cast<float>(output_count));
            set_error(error_message, error_message_capacity, "");
            return 1;
        }

        std::vector<SurfaceSample> input(input_count);
        for (std::uint32_t index = 0; index < input_count; ++index)
        {
            SurfaceSample& sample = input[index];
            sample.source_index = index;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                sample.position[dimension] = positions[index * 3 + dimension];
                sample.normal[dimension] = normals[index * 3 + dimension];
                if (!std::isfinite(sample.position[dimension]) ||
                    !std::isfinite(sample.normal[dimension]))
                {
                    throw std::invalid_argument(
                        "positions and normals must contain only finite values");
                }
            }
        }

        const float sqrt_three = std::sqrt(3.0f);
        const float max_poisson_radius = std::sqrt(
            options->surface_area /
            (2.0f * sqrt_three * static_cast<float>(output_count)));
        const float d_max = std::max(2.0f * max_poisson_radius, 1.0e-6f);
        const float normal_threshold = options->normal_cosine_threshold;
        const float plane_limit = std::max(
            d_max * options->plane_distance_scale, 1.0e-8f);

        std::vector<SurfaceSample> output(output_count);
        cy::WeightedSampleElimination<
            SurfaceSample,
            float,
            3,
            std::uint32_t>
            eliminator;
        eliminator.Eliminate(
            input.data(),
            input_count,
            output.data(),
            output_count,
            false,
            d_max,
            2,
            [normal_threshold, plane_limit](
                const SurfaceSample& first,
                const SurfaceSample& second,
                float distance_squared,
                float maximum_distance) {
                if (distance_squared <= 1.0e-20f ||
                    distance_squared >= maximum_distance * maximum_distance)
                {
                    return 0.0f;
                }
                if (dot3(first.normal, second.normal) < normal_threshold)
                {
                    return 0.0f;
                }
                const float delta[3] = {
                    second.position[0] - first.position[0],
                    second.position[1] - first.position[1],
                    second.position[2] - first.position[2],
                };
                if (std::abs(dot3(delta, first.normal)) > plane_limit ||
                    std::abs(dot3(delta, second.normal)) > plane_limit)
                {
                    return 0.0f;
                }
                const float normalized_distance =
                    std::sqrt(distance_squared) / maximum_distance;
                const float base = std::max(1.0f - normalized_distance, 0.0f);
                const float square = base * base;
                return square * square * square * square;
            });

        for (std::uint32_t index = 0; index < output_count; ++index)
        {
            output_indices[index] = output[index].source_index;
        }
        *output_poisson_radius = d_max * 0.5f;
        set_error(error_message, error_message_capacity, "");
        return 1;
    }
    catch (const std::exception& exception)
    {
        set_error(error_message, error_message_capacity, exception.what());
        return 0;
    }
    catch (...)
    {
        set_error(error_message, error_message_capacity, "unknown C++ exception");
        return 0;
    }
}

extern "C" int surface_probe_wse_eliminate_adaptive(
    const float* positions,
    const float* normals,
    const float* relative_densities,
    const float* partition_masses,
    std::uint32_t input_count,
    std::uint32_t output_count,
    const SurfaceProbeWSEOptions* options,
    std::uint32_t* output_indices,
    float* output_poisson_radius,
    SurfaceProbeAdaptiveWSEProfile* output_profile,
    char* error_message,
    std::size_t error_message_capacity)
{
    try
    {
        using ProfileClock = std::chrono::steady_clock;
        ProfileClock::time_point profile_origin;
        if (output_profile != nullptr)
        {
            *output_profile = {};
            profile_origin = ProfileClock::now();
        }
        if (positions == nullptr || normals == nullptr ||
            relative_densities == nullptr || partition_masses == nullptr ||
            options == nullptr ||
            output_indices == nullptr || output_poisson_radius == nullptr)
        {
            throw std::invalid_argument(
                "adaptive surface probe WSE received a null pointer");
        }
        validate_options(input_count, output_count, *options);

        const float sqrt_three = std::sqrt(3.0f);
        const float base_poisson_radius = std::sqrt(
            options->surface_area /
            (2.0f * sqrt_three * static_cast<float>(output_count)));
        const float base_d_max = std::max(
            2.0f * base_poisson_radius, 1.0e-6f);
        *output_poisson_radius = base_poisson_radius;

        std::vector<float> local_d_max(input_count);
        std::vector<float> local_partition_masses(input_count);
        for (std::uint32_t index = 0; index < input_count; ++index)
        {
            const float density = relative_densities[index];
            if (!std::isfinite(density) || density <= 0.0f)
            {
                throw std::invalid_argument(
                    "relative_densities must be finite and positive");
            }
            const float partition_mass = partition_masses[index];
            if (!std::isfinite(partition_mass) || partition_mass <= 0.0f)
            {
                throw std::invalid_argument(
                    "partition_masses must be finite and positive");
            }
            local_partition_masses[index] = partition_mass;
            local_d_max[index] = std::max(
                base_d_max / std::sqrt(density), 1.0e-6f);
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                if (!std::isfinite(positions[index * 3 + dimension]) ||
                    !std::isfinite(normals[index * 3 + dimension]))
                {
                    throw std::invalid_argument(
                        "positions and normals must contain only finite values");
                }
            }
        }
        if (output_profile != nullptr)
        {
            output_profile->setup_ms =
                std::chrono::duration<double, std::milli>(
                    ProfileClock::now() - profile_origin).count();
        }

        if (output_count == input_count)
        {
            for (std::uint32_t index = 0; index < input_count; ++index)
            {
                output_indices[index] = index;
            }
            if (output_profile != nullptr)
            {
                output_profile->final_output_count = output_count;
                output_profile->total_ms =
                    std::chrono::duration<double, std::milli>(
                        ProfileClock::now() - profile_origin).count();
            }
            set_error(error_message, error_message_capacity, "");
            return 1;
        }

        const SampleIndices selected = eliminate_adaptive_parallel(
            positions,
            normals,
            local_d_max,
            local_partition_masses,
            input_count,
            output_count,
            options->normal_cosine_threshold,
            options->plane_distance_scale,
            output_profile);
        for (std::uint32_t index = 0; index < output_count; ++index)
        {
            output_indices[index] = selected[index];
        }
        if (output_profile != nullptr)
        {
            output_profile->total_ms =
                std::chrono::duration<double, std::milli>(
                    ProfileClock::now() - profile_origin).count();
        }
        set_error(error_message, error_message_capacity, "");
        return 1;
    }
    catch (const std::exception& exception)
    {
        set_error(error_message, error_message_capacity, exception.what());
        return 0;
    }
    catch (...)
    {
        set_error(error_message, error_message_capacity, "unknown C++ exception");
        return 0;
    }
}

extern "C" int surface_probe_filter_audit_repair_candidates(
    const float* candidate_positions,
    const float* candidate_normals,
    std::uint32_t candidate_count,
    const float* base_positions,
    const float* base_normals,
    std::uint32_t base_count,
    const std::uint32_t* base_selected_indices,
    std::uint32_t base_selected_count,
    double audit_cell_size,
    double normal_cosine_threshold,
    std::uint32_t* output_audit_indices,
    std::uint32_t* output_audit_count,
    std::uint32_t* output_repair_indices,
    std::uint32_t* output_repair_count,
    SurfaceProbeCandidateFilterProfile* output_profile,
    char* error_message,
    std::size_t error_message_capacity)
{
    try
    {
        using FilterClock = std::chrono::steady_clock;
        const auto profile_origin = FilterClock::now();
        auto profile_previous = profile_origin;
        if (candidate_positions == nullptr || candidate_normals == nullptr ||
            base_positions == nullptr || base_normals == nullptr ||
            output_audit_indices == nullptr || output_audit_count == nullptr ||
            output_repair_indices == nullptr ||
            output_repair_count == nullptr)
        {
            throw std::invalid_argument(
                "surface probe candidate filter received a null pointer");
        }
        if (candidate_count == 0 || base_count == 0)
        {
            throw std::invalid_argument(
                "candidate_count and base_count must be positive");
        }
        if (base_selected_count > 0 && base_selected_indices == nullptr)
        {
            throw std::invalid_argument(
                "base_selected_indices cannot be null when selections exist");
        }
        if (!std::isfinite(audit_cell_size) || audit_cell_size <= 0.0f ||
            !std::isfinite(normal_cosine_threshold) ||
            normal_cosine_threshold < -1.0f ||
            normal_cosine_threshold > 1.0f)
        {
            throw std::invalid_argument(
                "invalid candidate filter cell size or normal threshold");
        }
        if (output_profile != nullptr)
        {
            *output_profile = {};
        }
        const auto profile_mark = [&profile_previous](double* destination) {
            const auto now = FilterClock::now();
            if (destination != nullptr)
            {
                *destination = std::chrono::duration<double, std::milli>(
                    now - profile_previous).count();
            }
            profile_previous = now;
        };

        constexpr std::uint32_t kParallelShardCount = 1024;
        const std::uint32_t shard_count =
            candidate_count >= 8192 ? kParallelShardCount : 1u;
        if (output_profile != nullptr)
        {
            output_profile->shard_count = shard_count;
            output_profile->worker_count = shard_count == 1
                ? 1u
                : parallel_worker_count(shard_count);
        }
        const double inverse_cell_size = 1.0 /
            std::max(audit_cell_size, 1.0e-8);
        const auto audit_key = [candidate_positions, inverse_cell_size](
            std::uint32_t index) {
            const float* position = candidate_positions +
                static_cast<std::size_t>(index) * 3;
            return GridKey{
                0,
                static_cast<std::int64_t>(std::floor(
                    static_cast<double>(position[0]) * inverse_cell_size)),
                static_cast<std::int64_t>(std::floor(
                    static_cast<double>(position[1]) * inverse_cell_size)),
                static_cast<std::int64_t>(std::floor(
                    static_cast<double>(position[2]) * inverse_cell_size)),
            };
        };

        std::vector<std::vector<std::uint32_t>> shards(shard_count);
        const std::size_t average_shard_size =
            static_cast<std::size_t>(candidate_count) / shard_count + 1;
        for (auto& shard : shards)
        {
            shard.reserve(average_shard_size);
        }
        for (std::uint32_t index = 0; index < candidate_count; ++index)
        {
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                if (!std::isfinite(candidate_positions[index * 3 + dimension]) ||
                    !std::isfinite(candidate_normals[index * 3 + dimension]))
                {
                    throw std::invalid_argument(
                        "candidate positions and normals must be finite");
                }
            }
            const GridKey key = audit_key(index);
            shards[GridKeyHash{}(key) % shard_count].push_back(index);
        }
        profile_mark(
            output_profile != nullptr
                ? &output_profile->audit_partition_ms
                : nullptr);

        std::vector<std::uint8_t> audit_selected(candidate_count, 0);
        parallel_for_indices(shard_count, [&](std::uint32_t shard_index) {
            const auto& indices = shards[shard_index];
            std::unordered_map<
                GridKey,
                std::vector<std::uint32_t>,
                GridKeyHash>
                representatives;
            representatives.reserve(indices.size() / 4 + 1);
            for (const std::uint32_t index : indices)
            {
                const GridKey key = audit_key(index);
                auto& cell_representatives = representatives[key];
                const float* normal = candidate_normals +
                    static_cast<std::size_t>(index) * 3;
                bool duplicate = false;
                for (const std::uint32_t representative : cell_representatives)
                {
                    if (dot3(
                            normal,
                            candidate_normals +
                                static_cast<std::size_t>(representative) * 3) >=
                        normal_cosine_threshold)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate)
                {
                    cell_representatives.push_back(index);
                    audit_selected[index] = 1;
                }
            }
        });
        profile_mark(
            output_profile != nullptr
                ? &output_profile->audit_deduplicate_ms
                : nullptr);

        std::vector<std::vector<std::uint32_t>>().swap(shards);
        shards.resize(shard_count);
        for (auto& shard : shards)
        {
            shard.reserve(average_shard_size);
        }
        std::vector<std::vector<std::uint32_t>> base_shards(shard_count);
        const std::size_t average_base_shard_size =
            static_cast<std::size_t>(base_selected_count) / shard_count + 1;
        for (auto& shard : base_shards)
        {
            shard.reserve(average_base_shard_size);
        }
        const ExactSurfaceKeyHash exact_hash;
        for (std::uint32_t selected = 0;
             selected < base_selected_count;
             ++selected)
        {
            const std::uint32_t index = base_selected_indices[selected];
            if (index >= base_count)
            {
                throw std::invalid_argument(
                    "base_selected_indices contains an invalid index");
            }
            const ExactSurfaceKey key = exact_surface_key(
                base_positions, base_normals, index);
            base_shards[exact_hash(key) % shard_count].push_back(index);
        }
        for (std::uint32_t index = 0; index < candidate_count; ++index)
        {
            const ExactSurfaceKey key = exact_surface_key(
                candidate_positions, candidate_normals, index);
            shards[exact_hash(key) % shard_count].push_back(index);
        }
        profile_mark(
            output_profile != nullptr
                ? &output_profile->repair_partition_ms
                : nullptr);

        std::vector<std::uint8_t> repair_selected(candidate_count, 0);
        parallel_for_indices(shard_count, [&](std::uint32_t shard_index) {
            std::unordered_set<ExactSurfaceKey, ExactSurfaceKeyHash> occupied;
            occupied.reserve(
                base_shards[shard_index].size() +
                shards[shard_index].size());
            for (const std::uint32_t index : base_shards[shard_index])
            {
                occupied.insert(exact_surface_key(
                    base_positions, base_normals, index));
            }
            for (const std::uint32_t index : shards[shard_index])
            {
                const ExactSurfaceKey key = exact_surface_key(
                    candidate_positions, candidate_normals, index);
                if (occupied.insert(key).second)
                {
                    repair_selected[index] = 1;
                }
            }
        });
        profile_mark(
            output_profile != nullptr
                ? &output_profile->repair_exclude_ms
                : nullptr);

        std::uint32_t audit_output_count = 0;
        std::uint32_t repair_output_count = 0;
        for (std::uint32_t index = 0; index < candidate_count; ++index)
        {
            if (audit_selected[index] != 0)
            {
                output_audit_indices[audit_output_count++] = index;
            }
            if (repair_selected[index] != 0)
            {
                output_repair_indices[repair_output_count++] = index;
            }
        }
        *output_audit_count = audit_output_count;
        *output_repair_count = repair_output_count;
        profile_mark(
            output_profile != nullptr ? &output_profile->compact_ms : nullptr);
        if (output_profile != nullptr)
        {
            output_profile->audit_output_count = audit_output_count;
            output_profile->repair_output_count = repair_output_count;
            output_profile->total_ms =
                std::chrono::duration<double, std::milli>(
                    FilterClock::now() - profile_origin).count();
        }
        set_error(error_message, error_message_capacity, "");
        return 1;
    }
    catch (const std::exception& exception)
    {
        set_error(error_message, error_message_capacity, exception.what());
        return 0;
    }
    catch (...)
    {
        set_error(error_message, error_message_capacity, "unknown C++ exception");
        return 0;
    }
}

extern "C" int surface_probe_deficit_repair(
    const float* base_positions,
    const float* base_normals,
    const std::uint32_t* base_instances,
    std::uint32_t base_count,
    const float* candidate_positions,
    const float* candidate_normals,
    const std::uint32_t* candidate_instances,
    std::uint32_t candidate_count,
    const float* audit_positions,
    const float* audit_normals,
    const std::uint32_t* audit_instances,
    std::uint32_t audit_count,
    const float* instance_radii,
    std::uint32_t instance_count,
    const SurfaceProbeRepairOptions* options,
    std::uint32_t* output_candidate_indices,
    std::uint32_t* output_repair_count,
    std::uint32_t* output_counts_before,
    std::uint32_t* output_counts_after,
    float* output_weight_sums_before,
    float* output_weight_sums_after,
    float* output_ess_before,
    float* output_ess_after,
    SurfaceProbeRepairProfile* output_profile,
    char* error_message,
    std::size_t error_message_capacity)
{
    try
    {
        using RepairClock = std::chrono::steady_clock;
        const auto profile_origin = RepairClock::now();
        auto profile_previous = profile_origin;
        if (output_profile != nullptr)
        {
            *output_profile = {};
            output_profile->worker_count = parallel_worker_count(
                std::max(audit_count, candidate_count));
        }
        const auto profile_mark = [&profile_previous](double* destination) {
            const auto now = RepairClock::now();
            if (destination != nullptr)
            {
                *destination = std::chrono::duration<double, std::milli>(
                    now - profile_previous).count();
            }
            profile_previous = now;
        };
        if (base_positions == nullptr || base_normals == nullptr ||
            base_instances == nullptr || audit_positions == nullptr ||
            audit_normals == nullptr ||
            audit_instances == nullptr || instance_radii == nullptr ||
            options == nullptr || output_repair_count == nullptr ||
            output_counts_before == nullptr || output_counts_after == nullptr ||
            output_weight_sums_before == nullptr ||
            output_weight_sums_after == nullptr || output_ess_before == nullptr ||
            output_ess_after == nullptr)
        {
            throw std::invalid_argument(
                "surface probe deficit repair received a null pointer");
        }
        if (candidate_count > 0 &&
            (candidate_positions == nullptr || candidate_normals == nullptr ||
             candidate_instances == nullptr))
        {
            throw std::invalid_argument(
                "non-empty repair candidates require non-null arrays");
        }
        if (base_count == 0 || audit_count == 0 || instance_count == 0)
        {
            throw std::invalid_argument(
                "base_count, audit_count, and instance_count must be positive");
        }
        if (options->min_gather_count == 0 ||
            options->min_gather_count > 32)
        {
            throw std::invalid_argument(
                "min_gather_count must be within [1, 32]");
        }
        if (!std::isfinite(options->normal_cosine_threshold) ||
            options->normal_cosine_threshold < -1.0f ||
            options->normal_cosine_threshold > 1.0f ||
            !std::isfinite(options->weight_epsilon) ||
            options->weight_epsilon < 0.0f)
        {
            throw std::invalid_argument("invalid deficit repair thresholds");
        }
        const std::uint32_t repair_limit = std::min(
            options->max_repair_count, candidate_count);
        if (repair_limit > 0 && output_candidate_indices == nullptr)
        {
            throw std::invalid_argument(
                "output_candidate_indices cannot be null when repairs are enabled");
        }

        float maximum_radius = 0.0f;
        for (std::uint32_t instance = 0; instance < instance_count; ++instance)
        {
            const float radius = instance_radii[instance];
            if (!std::isfinite(radius) || radius <= 0.0f)
            {
                throw std::invalid_argument(
                    "instance radii must be finite and positive");
            }
            maximum_radius = std::max(maximum_radius, radius);
        }
        const auto validate_instances = [instance_count](
            const std::uint32_t* instances,
            std::uint32_t count,
            const char* label) {
            for (std::uint32_t index = 0; index < count; ++index)
            {
                if (instances[index] >= instance_count)
                {
                    throw std::invalid_argument(
                        std::string(label) + " contains an invalid instance index");
                }
            }
        };
        validate_instances(base_instances, base_count, "base_instances");
        validate_instances(candidate_instances, candidate_count, "candidate_instances");
        validate_instances(audit_instances, audit_count, "audit_instances");

        const PointGrid base_grid(
            base_positions, base_instances, base_count, maximum_radius);
        profile_mark(
            output_profile != nullptr
                ? &output_profile->acceleration_structure_ms
                : nullptr);
        std::vector<std::uint32_t> deficits(audit_count, 0);
        parallel_for_indices(audit_count, [&](std::uint32_t audit) {
            const std::uint32_t instance = audit_instances[audit];
            const float radius = instance_radii[instance];
            const float* query_position =
                audit_positions + static_cast<std::size_t>(audit) * 3;
            const float* query_normal =
                audit_normals + static_cast<std::size_t>(audit) * 3;
            std::vector<WeightedProbe> candidates;
            base_grid.visit_nearby(
                query_position,
                instance,
                [&](std::uint32_t probe) {
                    const float weight = gather_weight(
                        query_position,
                        query_normal,
                        base_positions + static_cast<std::size_t>(probe) * 3,
                        base_normals + static_cast<std::size_t>(probe) * 3,
                        radius,
                        options->normal_cosine_threshold,
                        options->weight_epsilon);
                    if (weight > 0.0f)
                    {
                        candidates.push_back(WeightedProbe{weight, probe});
                    }
                });
            std::vector<float> weights = select_gather_weights(candidates);
            const GatherStats stats = summarize_weights(weights);
            output_counts_before[audit] = stats.count;
            output_weight_sums_before[audit] = stats.weight_sum;
            output_ess_before[audit] = stats.ess;
            deficits[audit] = stats.count < options->min_gather_count
                ? options->min_gather_count - stats.count
                : 0;
        });
        profile_mark(
            output_profile != nullptr ? &output_profile->base_gather_ms : nullptr);

        std::vector<std::uint32_t> coverage_candidates;
        std::vector<std::size_t> coverage_offsets;
        std::vector<std::uint32_t> coverage_audits;
        if (candidate_count > 0 && repair_limit > 0)
        {
            const PointGrid candidate_grid(
                candidate_positions,
                candidate_instances,
                candidate_count,
                maximum_radius);
            auto thread_pairs = parallel_collect_indices<std::uint64_t>(
                audit_count,
                [&](std::uint32_t audit, std::vector<std::uint64_t>& pairs) {
                if (deficits[audit] == 0)
                {
                    return;
                }
                const std::uint32_t instance = audit_instances[audit];
                const float radius = instance_radii[instance];
                const float* query_position =
                    audit_positions + static_cast<std::size_t>(audit) * 3;
                const float* query_normal =
                    audit_normals + static_cast<std::size_t>(audit) * 3;
                candidate_grid.visit_nearby(
                    query_position,
                    instance,
                    [&](std::uint32_t candidate) {
                        const float weight = gather_weight(
                            query_position,
                            query_normal,
                            candidate_positions +
                                static_cast<std::size_t>(candidate) * 3,
                            candidate_normals +
                                static_cast<std::size_t>(candidate) * 3,
                            radius,
                            options->normal_cosine_threshold,
                            options->weight_epsilon);
                        if (weight > 0.0f)
                        {
                            pairs.push_back(
                                (static_cast<std::uint64_t>(candidate) << 32) |
                                static_cast<std::uint64_t>(audit));
                        }
                    });
                });
            std::size_t pair_count = 0;
            for (const auto& pairs : thread_pairs)
            {
                pair_count += pairs.size();
            }
            std::vector<std::uint64_t> pairs;
            pairs.reserve(pair_count);
            for (auto& local_pairs : thread_pairs)
            {
                pairs.insert(
                    pairs.end(), local_pairs.begin(), local_pairs.end());
                std::vector<std::uint64_t>().swap(local_pairs);
            }
            std::sort(pairs.begin(), pairs.end());
            coverage_candidates.reserve(std::min<std::size_t>(
                pairs.size(), candidate_count));
            coverage_offsets.reserve(coverage_candidates.capacity() + 1);
            coverage_audits.reserve(pairs.size());
            std::uint32_t previous_candidate =
                std::numeric_limits<std::uint32_t>::max();
            for (const std::uint64_t pair : pairs)
            {
                const std::uint32_t candidate =
                    static_cast<std::uint32_t>(pair >> 32);
                if (coverage_candidates.empty() ||
                    candidate != previous_candidate)
                {
                    coverage_candidates.push_back(candidate);
                    coverage_offsets.push_back(coverage_audits.size());
                    previous_candidate = candidate;
                }
                coverage_audits.push_back(static_cast<std::uint32_t>(pair));
            }
            coverage_offsets.push_back(coverage_audits.size());
            if (output_profile != nullptr)
            {
                output_profile->coverage_pair_count = pair_count;
            }
        }
        profile_mark(
            output_profile != nullptr
                ? &output_profile->coverage_build_ms
                : nullptr);

        const auto score_candidate = [
            &coverage_offsets, &coverage_audits, &deficits](
            std::uint32_t coverage_slot) {
            std::uint64_t score = 0;
            for (std::size_t offset = coverage_offsets[coverage_slot];
                 offset < coverage_offsets[coverage_slot + 1];
                 ++offset)
            {
                const std::uint32_t audit = coverage_audits[offset];
                const std::uint32_t deficit = deficits[audit];
                if (deficit > 0)
                {
                    score += static_cast<std::uint64_t>(2 * deficit - 1);
                }
            }
            return score;
        };
        std::priority_queue<
            RepairHeapEntry,
            std::vector<RepairHeapEntry>,
            RepairHeapCompare>
            heap;
        std::vector<std::uint64_t> initial_scores(
            coverage_candidates.size(), 0);
        parallel_for_indices(
            static_cast<std::uint32_t>(coverage_candidates.size()),
            [&](std::uint32_t slot) {
                initial_scores[slot] = score_candidate(slot);
            });
        for (std::uint32_t slot = 0;
             slot < static_cast<std::uint32_t>(coverage_candidates.size());
             ++slot)
        {
            const std::uint64_t score = initial_scores[slot];
            if (score > 0)
            {
                heap.push(RepairHeapEntry{
                    score, coverage_candidates[slot], slot});
            }
        }
        profile_mark(
            output_profile != nullptr ? &output_profile->heap_build_ms : nullptr);

        std::vector<bool> selected(coverage_candidates.size(), false);
        std::uint32_t selected_count = 0;
        while (selected_count < repair_limit && !heap.empty())
        {
            const RepairHeapEntry entry = heap.top();
            heap.pop();
            if (selected[entry.coverage_slot])
            {
                continue;
            }
            const std::uint64_t current_score = score_candidate(
                entry.coverage_slot);
            if (current_score == 0)
            {
                continue;
            }
            if (current_score != entry.score)
            {
                heap.push(RepairHeapEntry{
                    current_score, entry.candidate, entry.coverage_slot});
                continue;
            }
            selected[entry.coverage_slot] = true;
            output_candidate_indices[selected_count++] = entry.candidate;
            for (std::size_t offset = coverage_offsets[entry.coverage_slot];
                 offset < coverage_offsets[entry.coverage_slot + 1];
                 ++offset)
            {
                const std::uint32_t audit = coverage_audits[offset];
                if (deficits[audit] > 0)
                {
                    --deficits[audit];
                }
            }
        }
        *output_repair_count = selected_count;
        profile_mark(
            output_profile != nullptr
                ? &output_profile->greedy_select_ms
                : nullptr);

        std::copy_n(output_counts_before, audit_count, output_counts_after);
        std::copy_n(
            output_weight_sums_before, audit_count, output_weight_sums_after);
        std::copy_n(output_ess_before, audit_count, output_ess_after);
        std::vector<std::uint32_t> affected_audits;
        std::vector<float> final_positions;
        std::vector<float> final_normals;
        std::vector<std::uint32_t> final_instances;
        if (selected_count > 0)
        {
            final_positions.reserve(
                static_cast<std::size_t>(base_count + selected_count) * 3);
            final_normals.reserve(
                static_cast<std::size_t>(base_count + selected_count) * 3);
            final_instances.reserve(base_count + selected_count);
            final_positions.insert(
                final_positions.end(),
                base_positions,
                base_positions + base_count * 3);
            final_normals.insert(
                final_normals.end(),
                base_normals,
                base_normals + base_count * 3);
            final_instances.insert(
                final_instances.end(),
                base_instances,
                base_instances + base_count);
            for (std::uint32_t repair = 0; repair < selected_count; ++repair)
            {
                const std::uint32_t candidate = output_candidate_indices[repair];
                final_positions.insert(
                    final_positions.end(),
                    candidate_positions + static_cast<std::size_t>(candidate) * 3,
                    candidate_positions +
                        static_cast<std::size_t>(candidate + 1) * 3);
                final_normals.insert(
                    final_normals.end(),
                    candidate_normals + static_cast<std::size_t>(candidate) * 3,
                    candidate_normals +
                        static_cast<std::size_t>(candidate + 1) * 3);
                final_instances.push_back(candidate_instances[candidate]);
            }

            const PointGrid audit_grid(
                audit_positions, audit_instances, audit_count, maximum_radius);
            auto thread_audits = parallel_collect_indices<std::uint32_t>(
                selected_count,
                [&](std::uint32_t repair,
                    std::vector<std::uint32_t>& local_audits) {
                    const std::uint32_t candidate =
                        output_candidate_indices[repair];
                    const float* probe_position = candidate_positions +
                        static_cast<std::size_t>(candidate) * 3;
                    const float* probe_normal = candidate_normals +
                        static_cast<std::size_t>(candidate) * 3;
                    const std::uint32_t instance =
                        candidate_instances[candidate];
                    const float radius = instance_radii[instance];
                    audit_grid.visit_nearby(
                        probe_position,
                        instance,
                        [&](std::uint32_t audit) {
                            const float weight = gather_weight(
                                audit_positions +
                                    static_cast<std::size_t>(audit) * 3,
                                audit_normals +
                                    static_cast<std::size_t>(audit) * 3,
                                probe_position,
                                probe_normal,
                                radius,
                                options->normal_cosine_threshold,
                                options->weight_epsilon);
                            if (weight > 0.0f)
                            {
                                local_audits.push_back(audit);
                            }
                        });
                });
            std::size_t affected_count = 0;
            for (const auto& local_audits : thread_audits)
            {
                affected_count += local_audits.size();
            }
            affected_audits.reserve(affected_count);
            for (auto& local_audits : thread_audits)
            {
                affected_audits.insert(
                    affected_audits.end(),
                    local_audits.begin(),
                    local_audits.end());
            }
            std::sort(affected_audits.begin(), affected_audits.end());
            affected_audits.erase(
                std::unique(affected_audits.begin(), affected_audits.end()),
                affected_audits.end());
        }
        if (output_profile != nullptr)
        {
            output_profile->affected_audit_count =
                static_cast<std::uint32_t>(affected_audits.size());
        }
        profile_mark(
            output_profile != nullptr
                ? &output_profile->affected_audits_ms
                : nullptr);

        if (selected_count > 0)
        {
            const PointGrid final_grid(
                final_positions.data(),
                final_instances.data(),
                base_count + selected_count,
                maximum_radius);
            parallel_for_indices(
                static_cast<std::uint32_t>(affected_audits.size()),
                [&](std::uint32_t affected_index) {
                const std::uint32_t audit = affected_audits[affected_index];
                const std::uint32_t instance = audit_instances[audit];
                const float radius = instance_radii[instance];
                const float* query_position =
                    audit_positions + static_cast<std::size_t>(audit) * 3;
                const float* query_normal =
                    audit_normals + static_cast<std::size_t>(audit) * 3;
                std::vector<WeightedProbe> candidates;
                final_grid.visit_nearby(
                    query_position,
                    instance,
                    [&](std::uint32_t probe) {
                        const float weight = gather_weight(
                            query_position,
                            query_normal,
                            final_positions.data() +
                                static_cast<std::size_t>(probe) * 3,
                            final_normals.data() +
                                static_cast<std::size_t>(probe) * 3,
                            radius,
                            options->normal_cosine_threshold,
                            options->weight_epsilon);
                        if (weight > 0.0f)
                        {
                            candidates.push_back(WeightedProbe{weight, probe});
                        }
                    });
                std::vector<float> weights = select_gather_weights(candidates);
                const GatherStats stats = summarize_weights(weights);
                output_counts_after[audit] = stats.count;
                output_weight_sums_after[audit] = stats.weight_sum;
                output_ess_after[audit] = stats.ess;
            });
        }
        profile_mark(
            output_profile != nullptr ? &output_profile->final_gather_ms : nullptr);
        if (output_profile != nullptr)
        {
            output_profile->total_ms =
                std::chrono::duration<double, std::milli>(
                    RepairClock::now() - profile_origin).count();
        }
        set_error(error_message, error_message_capacity, "");
        return 1;
    }
    catch (const std::exception& exception)
    {
        set_error(error_message, error_message_capacity, exception.what());
        return 0;
    }
    catch (...)
    {
        set_error(error_message, error_message_capacity, "unknown C++ exception");
        return 0;
    }
}

extern "C" int surface_probe_estimate_support(
    const float* reference_positions,
    const float* reference_normals,
    const std::uint32_t* reference_instances,
    const float* reference_area_weights,
    std::uint32_t reference_count,
    const float* query_positions,
    const float* query_normals,
    const std::uint32_t* query_instances,
    std::uint32_t query_count,
    const float* instance_radii,
    std::uint32_t instance_count,
    const SurfaceProbeSupportOptions* options,
    float* output_support_f,
    float* output_density_m,
    char* error_message,
    std::size_t error_message_capacity)
{
    try
    {
        if (reference_positions == nullptr || reference_normals == nullptr ||
            reference_instances == nullptr || reference_area_weights == nullptr ||
            query_positions == nullptr || query_normals == nullptr ||
            query_instances == nullptr || instance_radii == nullptr ||
            options == nullptr || output_support_f == nullptr ||
            output_density_m == nullptr)
        {
            throw std::invalid_argument(
                "surface probe support estimation received a null pointer");
        }
        if (reference_count == 0 || query_count == 0 || instance_count == 0)
        {
            throw std::invalid_argument(
                "support reference, query, and instance counts must be positive");
        }
        if (!std::isfinite(options->normal_cosine_threshold) ||
            options->normal_cosine_threshold < -1.0f ||
            options->normal_cosine_threshold > 1.0f ||
            !std::isfinite(options->weight_epsilon) ||
            options->weight_epsilon < 0.0f ||
            !std::isfinite(options->max_density_multiplier) ||
            options->max_density_multiplier < 1.0f)
        {
            throw std::invalid_argument("invalid support estimation options");
        }

        float maximum_radius = 0.0f;
        for (std::uint32_t instance = 0; instance < instance_count; ++instance)
        {
            const float radius = instance_radii[instance];
            if (!std::isfinite(radius) || radius <= 0.0f)
            {
                throw std::invalid_argument(
                    "instance radii must be finite and positive");
            }
            maximum_radius = std::max(maximum_radius, radius);
        }
        for (std::uint32_t reference = 0; reference < reference_count; ++reference)
        {
            if (reference_instances[reference] >= instance_count ||
                !std::isfinite(reference_area_weights[reference]) ||
                reference_area_weights[reference] < 0.0f)
            {
                throw std::invalid_argument(
                    "invalid support reference instance or area weight");
            }
        }
        for (std::uint32_t query = 0; query < query_count; ++query)
        {
            if (query_instances[query] >= instance_count)
            {
                throw std::invalid_argument(
                    "query_instances contains an invalid instance index");
            }
        }

        const PointGrid reference_grid(
            reference_positions,
            reference_instances,
            reference_count,
            maximum_radius);
        const auto estimate = [&](std::uint32_t query) {
            const std::uint32_t instance = query_instances[query];
            const float radius = instance_radii[instance];
            const float* query_position =
                query_positions + static_cast<std::size_t>(query) * 3;
            const float* query_normal =
                query_normals + static_cast<std::size_t>(query) * 3;
            double compatible_support = 0.0;
            reference_grid.visit_nearby(
                query_position,
                instance,
                [&](std::uint32_t reference) {
                    const float kernel_weight = gather_weight(
                        query_position,
                        query_normal,
                        reference_positions +
                            static_cast<std::size_t>(reference) * 3,
                        reference_normals +
                            static_cast<std::size_t>(reference) * 3,
                        radius,
                        options->normal_cosine_threshold,
                        options->weight_epsilon);
                    compatible_support += static_cast<double>(kernel_weight) *
                        reference_area_weights[reference];
                });
            constexpr double kPi = 3.14159265358979323846;
            const double flat_support =
                kPi * static_cast<double>(radius) * radius / 3.0;
            const float support_f = static_cast<float>(std::clamp(
                compatible_support / std::max(flat_support, 1.0e-30),
                0.0,
                1.0));
            const float density_m = std::min(
                options->max_density_multiplier,
                1.0f / std::max(
                    support_f,
                    1.0f / options->max_density_multiplier));
            output_support_f[query] = support_f;
            output_density_m[query] = std::max(density_m, 1.0f);
        };
        parallel_for_indices(query_count, estimate);
        set_error(error_message, error_message_capacity, "");
        return 1;
    }
    catch (const std::exception& exception)
    {
        set_error(error_message, error_message_capacity, exception.what());
        return 0;
    }
    catch (...)
    {
        set_error(error_message, error_message_capacity, "unknown C++ exception");
        return 0;
    }
}

extern "C" int surface_probe_build_point_octree(
    const float* positions,
    std::uint32_t point_count,
    std::uint32_t leaf_capacity,
    std::uint32_t max_depth,
    SurfaceProbePointOctreeResult* output,
    SurfaceProbePointOctreeProfile* output_profile,
    char* error_message,
    std::size_t error_message_capacity)
{
    try
    {
        using ProfileClock = std::chrono::steady_clock;
        ProfileClock::time_point profile_origin;
        ProfileClock::time_point phase_start;
        if (output == nullptr)
        {
            throw std::invalid_argument(
                "point octree output must not be null");
        }
        *output = {};
        if (output_profile != nullptr)
        {
            *output_profile = {};
            profile_origin = ProfileClock::now();
            phase_start = profile_origin;
        }
        if (positions == nullptr || point_count == 0 || leaf_capacity == 0 ||
            max_depth == 0)
        {
            throw std::invalid_argument(
                "point octree requires positions and positive counts/options");
        }

        std::array<float, 3> minimum{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
        std::array<float, 3> maximum{
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
        for (std::uint32_t point = 0; point < point_count; ++point)
        {
            for (std::uint32_t axis = 0; axis < 3; ++axis)
            {
                const float value =
                    positions[static_cast<std::size_t>(point) * 3 + axis];
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument(
                        "point octree positions must be finite");
                }
                minimum[axis] = std::min(minimum[axis], value);
                maximum[axis] = std::max(maximum[axis], value);
            }
        }
        std::array<double, 3> root_center{};
        double maximum_range = 0.0;
        for (std::uint32_t axis = 0; axis < 3; ++axis)
        {
            root_center[axis] =
                (static_cast<double>(minimum[axis]) + maximum[axis]) * 0.5;
            maximum_range = std::max(
                maximum_range,
                static_cast<double>(maximum[axis]) - minimum[axis]);
        }
        const double root_extent =
            std::max(maximum_range * 0.5, 1.0e-5) * 1.0001;
        if (output_profile != nullptr)
        {
            const auto now = ProfileClock::now();
            output_profile->bounds_ms =
                std::chrono::duration<double, std::milli>(
                    now - phase_start).count();
            phase_start = now;
        }

        std::vector<std::uint32_t> indices(point_count);
        std::iota(indices.begin(), indices.end(), 0u);
        std::vector<std::uint32_t> scratch(point_count);
        std::vector<std::uint8_t> octant_codes(point_count);
        if (output_profile != nullptr)
        {
            const auto now = ProfileClock::now();
            output_profile->index_setup_ms =
                std::chrono::duration<double, std::milli>(
                    now - phase_start).count();
            phase_start = now;
        }

        PointOctreeBuildNode root;
        root.begin = 0;
        root.end = point_count;
        const std::uint32_t worker_count = parallel_worker_count(point_count);
        build_point_octree_node(
            positions,
            indices,
            scratch,
            octant_codes,
            root,
            root_center,
            root_extent,
            0,
            leaf_capacity,
            max_depth,
            worker_count);
        if (output_profile != nullptr)
        {
            const auto now = ProfileClock::now();
            output_profile->partition_ms =
                std::chrono::duration<double, std::milli>(
                    now - phase_start).count();
            phase_start = now;
        }

        std::vector<std::array<std::uint32_t, 4>> flat_nodes(1);
        std::vector<std::uint32_t> probe_order;
        probe_order.reserve(point_count);
        flatten_point_octree(root, indices, 0, flat_nodes, probe_order);
        if (probe_order.size() != point_count)
        {
            throw std::runtime_error(
                "point octree flatten produced an invalid probe count");
        }
        if (output_profile != nullptr)
        {
            const auto now = ProfileClock::now();
            output_profile->flatten_ms =
                std::chrono::duration<double, std::milli>(
                    now - phase_start).count();
            phase_start = now;
        }

        auto output_nodes = std::make_unique<std::uint32_t[]>(
            flat_nodes.size() * 4);
        auto output_order = std::make_unique<std::uint32_t[]>(point_count);
        std::memcpy(
            output_nodes.get(),
            flat_nodes.data(),
            flat_nodes.size() * sizeof(flat_nodes[0]));
        std::memcpy(
            output_order.get(),
            probe_order.data(),
            probe_order.size() * sizeof(probe_order[0]));
        output->nodes = output_nodes.release();
        output->node_count = static_cast<std::uint32_t>(flat_nodes.size());
        output->probe_order = output_order.release();
        output->probe_count = point_count;
        std::copy(root_center.begin(), root_center.end(), output->root_center);
        output->root_extent = root_extent;
        if (output_profile != nullptr)
        {
            const auto now = ProfileClock::now();
            output_profile->output_copy_ms =
                std::chrono::duration<double, std::milli>(
                    now - phase_start).count();
            output_profile->total_ms =
                std::chrono::duration<double, std::milli>(
                    now - profile_origin).count();
            output_profile->worker_count = worker_count;
            output_profile->node_count = output->node_count;
        }
        set_error(error_message, error_message_capacity, "");
        return 1;
    }
    catch (const std::exception& exception)
    {
        if (output != nullptr)
        {
            delete[] output->nodes;
            delete[] output->probe_order;
            *output = {};
        }
        set_error(error_message, error_message_capacity, exception.what());
        return 0;
    }
    catch (...)
    {
        if (output != nullptr)
        {
            delete[] output->nodes;
            delete[] output->probe_order;
            *output = {};
        }
        set_error(error_message, error_message_capacity, "unknown C++ exception");
        return 0;
    }
}

extern "C" void surface_probe_free_point_octree(
    SurfaceProbePointOctreeResult* result)
{
    if (result == nullptr)
    {
        return;
    }
    delete[] result->nodes;
    delete[] result->probe_order;
    *result = {};
}

extern "C" const char* surface_probe_wse_version()
{
    return kVersion;
}
