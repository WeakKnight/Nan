#include "tinybvh_visibility.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#define NO_DOUBLE_PRECISION_SUPPORT
#define NO_CUSTOM_GEOMETRY
#define NO_VOXEL_SUPPORT
#define WATERTIGHT_TRITEST
#if !defined(VBAKE_TINYBVH_AVX2)
#define TINYBVH_NO_SIMD
#endif
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

namespace {

constexpr int kShCoefficientCount = 16;
constexpr int kRayBatchSize = 64;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kFarDistance = 1.0e30f;

struct RayProjection {
    tinybvh::bvhvec3 direction;
    tinybvh::bvhvec3 reciprocal_direction;
    std::array<float, kShCoefficientCount> weighted_sh;
};

std::uint32_t reverse_bits(std::uint32_t value)
{
    value = (value << 16) | (value >> 16);
    value = ((value & 0x55555555u) << 1) | ((value & 0xaaaaaaaau) >> 1);
    value = ((value & 0x33333333u) << 2) | ((value & 0xccccccccu) >> 2);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value & 0xf0f0f0f0u) >> 4);
    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
    return value;
}

tinybvh::bvhvec3 reciprocal(const tinybvh::bvhvec3& value)
{
    return tinybvh::tinybvh_rcp(value);
}

void evaluate_sh16(const tinybvh::bvhvec3& direction, float weight, std::array<float, 16>& sh)
{
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    sh[0] = weight * 0.282094791774f;
    sh[1] = weight * (-0.488602511903f * y);
    sh[2] = weight * (0.488602511903f * z);
    sh[3] = weight * (-0.488602511903f * x);
    sh[4] = weight * (1.09254843059f * x * y);
    sh[5] = weight * (-1.09254843059f * y * z);
    sh[6] = weight * (0.315391565253f * (-1.0f + 3.0f * z2));
    sh[7] = weight * (-1.09254843059f * x * z);
    sh[8] = weight * (0.546274215296f * (x2 - y2));
    sh[9] = weight * (-0.590043589927f * (3.0f * x2 * y - y2 * y));
    sh[10] = weight * (2.89061144264f * x * y * z);
    sh[11] = weight * (-0.457045799464f * y * (-1.0f + 5.0f * z2));
    sh[12] = weight * (0.37317633259f * z * (-3.0f + 5.0f * z2));
    sh[13] = weight * (-0.457045799464f * x * (-1.0f + 5.0f * z2));
    sh[14] = weight * (1.44530572132f * (x2 - y2) * z);
    sh[15] = weight * (-0.590043589927f * (x2 * x - 3.0f * x * y2));
}

std::vector<RayProjection> make_ray_projections(int ray_count)
{
    std::vector<RayProjection> rays(static_cast<std::size_t>(ray_count));
    const float sample_weight = (4.0f * kPi) / static_cast<float>(ray_count);
    for (int ray_index = 0; ray_index < ray_count; ++ray_index) {
        const float u0 = static_cast<float>(ray_index) / static_cast<float>(ray_count);
        const float u1 = static_cast<float>(reverse_bits(static_cast<std::uint32_t>(ray_index)))
            * 2.3283064365386963e-10f;
        const float phi = u0 * (2.0f * kPi);
        const float cos_theta = 1.0f - 2.0f * u1;
        const float sin_theta = std::sqrt((std::max)(1.0f - cos_theta * cos_theta, 0.0f));
        RayProjection& ray = rays[static_cast<std::size_t>(ray_index)];
        ray.direction = tinybvh::bvhvec3(
            std::cos(phi) * sin_theta,
            cos_theta,
            std::sin(phi) * sin_theta);
        ray.reciprocal_direction = reciprocal(ray.direction);
        evaluate_sh16(ray.direction, sample_weight, ray.weighted_sh);
    }
    return rays;
}

tinybvh::bvhvec3 normalized_sample_normal(const float* normal)
{
    const float length_squared = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if (std::isfinite(length_squared) && length_squared > 1.0e-16f) {
        const float inverse_length = 1.0f / std::sqrt(length_squared);
        return tinybvh::bvhvec3(
            normal[0] * inverse_length,
            normal[1] * inverse_length,
            normal[2] * inverse_length);
    }
    return tinybvh::bvhvec3(0.0f, 0.0f, 1.0f);
}

float nudge_component(float origin, float direction)
{
    if (direction > 0.0f) {
        return std::nextafter(origin, std::numeric_limits<float>::infinity());
    }
    if (direction < 0.0f) {
        return std::nextafter(origin, -std::numeric_limits<float>::infinity());
    }
    return origin;
}

tinybvh::bvhvec3 nudge_ray_origin(
    const tinybvh::bvhvec3& origin,
    const tinybvh::bvhvec3& direction)
{
    // TinyBVH has no ray t-min. Move to the next representable point so t == 0
    // surface hits are rejected like PMR's Python and hardware ray queries.
    return tinybvh::bvhvec3(
        nudge_component(origin.x, direction.x),
        nudge_component(origin.y, direction.y),
        nudge_component(origin.z, direction.z));
}

template <typename BvhType>
void build_bvh(BvhType& bvh, const tinybvh::bvhvec4* vertices, std::uint32_t triangle_count)
{
    bvh.Build(vertices, triangle_count);
}

template <typename BvhType>
void trace_samples(
    const BvhType& bvh,
    const float* sample_positions,
    const float* sample_normals,
    int sample_count,
    const std::vector<RayProjection>& rays,
    float max_distance,
    float self_bias,
    int requested_thread_count,
    float* out_sample_sh16,
    TinyBvhTraceTimings& timings)
{
    const unsigned int hardware_threads = (std::max)(1u, std::thread::hardware_concurrency());
    const int desired_threads = requested_thread_count > 0
        ? requested_thread_count
        : static_cast<int>(hardware_threads);
    const int worker_count = (std::max)(1, (std::min)(sample_count, desired_threads));
    timings.thread_count = worker_count;
    std::atomic<std::uint64_t> visible_ray_count{0};

    auto worker = [&](int begin, int end) {
        std::uint64_t local_visible_count = 0;
        for (int sample_index = begin; sample_index < end; ++sample_index) {
            const float* position = sample_positions + static_cast<std::size_t>(sample_index) * 3;
            const float* source_normal = sample_normals + static_cast<std::size_t>(sample_index) * 3;
            const float source_normal_length_squared = source_normal[0] * source_normal[0]
                + source_normal[1] * source_normal[1]
                + source_normal[2] * source_normal[2];
            const bool has_surface_normal =
                std::isfinite(source_normal_length_squared) && source_normal_length_squared > 1.0e-16f;
            const tinybvh::bvhvec3 normal = normalized_sample_normal(source_normal);
            const tinybvh::bvhvec3 origin(
                position[0] + normal.x * self_bias,
                position[1] + normal.y * self_bias,
                position[2] + normal.z * self_bias);
            std::array<float, kShCoefficientCount> accumulated{};

            for (int ray_begin = 0; ray_begin < static_cast<int>(rays.size()); ray_begin += kRayBatchSize) {
                const int ray_end = (std::min)(ray_begin + kRayBatchSize, static_cast<int>(rays.size()));
                std::array<float, kShCoefficientCount> batch{};
                for (int ray_index = ray_begin; ray_index < ray_end; ++ray_index) {
                    const RayProjection& projection = rays[static_cast<std::size_t>(ray_index)];
                    tinybvh::Ray ray;
                    ray.O = has_surface_normal ? origin : nudge_ray_origin(origin, projection.direction);
                    ray.mask = RAY_MASK_INTERSECT_ALL;
                    ray.D = projection.direction;
                    ray.instIdx = 0;
                    ray.rD = projection.reciprocal_direction;
                    ray.hit.t = max_distance;
                    if (bvh.IsOccluded(ray)) {
                        continue;
                    }
                    ++local_visible_count;
                    for (int coefficient = 0; coefficient < kShCoefficientCount; ++coefficient) {
                        batch[coefficient] += projection.weighted_sh[coefficient];
                    }
                }
                for (int coefficient = 0; coefficient < kShCoefficientCount; ++coefficient) {
                    accumulated[coefficient] += batch[coefficient];
                }
            }
            std::memcpy(
                out_sample_sh16 + static_cast<std::size_t>(sample_index) * kShCoefficientCount,
                accumulated.data(),
                sizeof(float) * kShCoefficientCount);
        }
        visible_ray_count.fetch_add(local_visible_count, std::memory_order_relaxed);
    };

    const auto trace_begin = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count - 1));
    for (int worker_index = 1; worker_index < worker_count; ++worker_index) {
        const int begin = sample_count * worker_index / worker_count;
        const int end = sample_count * (worker_index + 1) / worker_count;
        workers.emplace_back(worker, begin, end);
    }
    worker(0, sample_count / worker_count);
    for (std::thread& thread : workers) {
        thread.join();
    }
    const auto trace_end = std::chrono::steady_clock::now();
    timings.trace_milliseconds = std::chrono::duration<double, std::milli>(trace_end - trace_begin).count();
    timings.visible_ray_count = visible_ray_count.load(std::memory_order_relaxed);
}

template <typename BvhType>
void build_and_trace(
    const std::vector<tinybvh::bvhvec4>& vertices,
    int triangle_count,
    const float* sample_positions,
    const float* sample_normals,
    int sample_count,
    const std::vector<RayProjection>& rays,
    float max_distance,
    float self_bias,
    int thread_count,
    float* out_sample_sh16,
    TinyBvhTraceTimings& timings)
{
    BvhType bvh;
    const auto build_begin = std::chrono::steady_clock::now();
    build_bvh(bvh, vertices.data(), static_cast<std::uint32_t>(triangle_count));
    const auto build_end = std::chrono::steady_clock::now();
    timings.build_milliseconds = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    trace_samples(
        bvh,
        sample_positions,
        sample_normals,
        sample_count,
        rays,
        max_distance,
        self_bias,
        thread_count,
        out_sample_sh16,
        timings);
}

} // namespace

bool tinybvh_has_bvh8_cpu()
{
#if defined(VBAKE_TINYBVH_AVX2) && defined(BVH_USEAVX2)
    return true;
#else
    return false;
#endif
}

bool trace_pmr_visibility_sh_tinybvh(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    int sample_count,
    const float* sample_positions,
    const float* sample_normals,
    int ray_count,
    float max_distance,
    float self_bias,
    int thread_count,
    int layout,
    float* out_sample_sh16,
    TinyBvhTraceTimings& timings,
    std::string& error)
{
    if (vertex_count <= 0 || triangle_count <= 0 || sample_count <= 0 || ray_count <= 0) {
        error = "TinyBVH vertex, triangle, sample, and ray counts must be positive";
        return false;
    }
    if (!positions || !indices || !sample_positions || !sample_normals || !out_sample_sh16) {
        error = "null TinyBVH visibility input/output pointer";
        return false;
    }
    if (layout != 0 && layout != 1) {
        error = "TinyBVH layout must be 0 (BVH) or 1 (BVH8_CPU)";
        return false;
    }
    if (thread_count < 0) {
        error = "TinyBVH thread_count must be non-negative";
        return false;
    }

    std::vector<tinybvh::bvhvec4> indexed_vertices(static_cast<std::size_t>(vertex_count));
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
        const float* position = positions + static_cast<std::size_t>(vertex) * 3;
        if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2])) {
            error = "TinyBVH geometry contains a non-finite vertex position";
            return false;
        }
        indexed_vertices[static_cast<std::size_t>(vertex)] =
            tinybvh::bvhvec4(position[0], position[1], position[2], 0.0f);
    }
    for (int index = 0; index < triangle_count * 3; ++index) {
        if (indices[index] >= static_cast<unsigned int>(vertex_count)) {
            error = "TinyBVH triangle vertex index out of range";
            return false;
        }
    }
    std::vector<tinybvh::bvhvec4> triangle_vertices;
    triangle_vertices.reserve(static_cast<std::size_t>(triangle_count) * 3);
    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        const tinybvh::bvhvec4& a = indexed_vertices[indices[static_cast<std::size_t>(triangle) * 3 + 0]];
        const tinybvh::bvhvec4& b = indexed_vertices[indices[static_cast<std::size_t>(triangle) * 3 + 1]];
        const tinybvh::bvhvec4& c = indexed_vertices[indices[static_cast<std::size_t>(triangle) * 3 + 2]];
        const double abx = static_cast<double>(b.x) - a.x;
        const double aby = static_cast<double>(b.y) - a.y;
        const double abz = static_cast<double>(b.z) - a.z;
        const double acx = static_cast<double>(c.x) - a.x;
        const double acy = static_cast<double>(c.y) - a.y;
        const double acz = static_cast<double>(c.z) - a.z;
        const double cross_x = aby * acz - abz * acy;
        const double cross_y = abz * acx - abx * acz;
        const double cross_z = abx * acy - aby * acx;
        const double cross_length_squared =
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
        if (!std::isfinite(cross_length_squared) || cross_length_squared <= 1.0e-20) {
            continue;
        }
        triangle_vertices.push_back(a);
        triangle_vertices.push_back(b);
        triangle_vertices.push_back(c);
    }
    const int trace_triangle_count = static_cast<int>(triangle_vertices.size() / 3);
    if (trace_triangle_count == 0) {
        error = "TinyBVH geometry contains no non-degenerate triangles";
        return false;
    }
    for (int sample = 0; sample < sample_count * 3; ++sample) {
        if (!std::isfinite(sample_positions[sample]) || !std::isfinite(sample_normals[sample])) {
            error = "TinyBVH sample positions and normals must be finite";
            return false;
        }
    }

    if (!std::isfinite(max_distance) || max_distance <= 0.0f) {
        max_distance = kFarDistance;
    }
    if (!std::isfinite(self_bias)) {
        error = "TinyBVH self_bias must be finite";
        return false;
    }
    self_bias = (std::max)(0.0f, self_bias);
    timings = TinyBvhTraceTimings{};
    const std::vector<RayProjection> rays = make_ray_projections(ray_count);

    try {
        if (layout == 0) {
            build_and_trace<tinybvh::BVH>(
                triangle_vertices,
                trace_triangle_count,
                sample_positions,
                sample_normals,
                sample_count,
                rays,
                max_distance,
                self_bias,
                thread_count,
                out_sample_sh16,
                timings);
            return true;
        }
#if defined(VBAKE_TINYBVH_AVX2) && defined(BVH_USEAVX2)
        build_and_trace<tinybvh::BVH8_CPU>(
            triangle_vertices,
            trace_triangle_count,
            sample_positions,
            sample_normals,
            sample_count,
            rays,
            max_distance,
            self_bias,
            thread_count,
            out_sample_sh16,
            timings);
        return true;
#else
        error = "TinyBVH BVH8_CPU was not compiled; use the BVH layout";
        return false;
#endif
    } catch (const std::exception& exception) {
        error = std::string("TinyBVH failed: ") + exception.what();
        return false;
    } catch (...) {
        error = "TinyBVH failed with an unknown native exception";
        return false;
    }
}
