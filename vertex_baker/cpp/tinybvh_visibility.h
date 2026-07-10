#pragma once

#include <cstdint>
#include <string>

struct TinyBvhTraceTimings {
    double build_milliseconds = 0.0;
    double trace_milliseconds = 0.0;
    std::uint64_t visible_ray_count = 0;
    int thread_count = 0;
};

bool tinybvh_has_bvh8_cpu();

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
    std::string& error);
