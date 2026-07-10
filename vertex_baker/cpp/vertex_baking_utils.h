#pragma once

#ifdef _WIN32
#define VBAKE_API __declspec(dllexport)
#else
#define VBAKE_API __attribute__((visibility("default")))
#endif

extern "C" {

enum VBakeStatus {
    VBAKE_SUCCESS = 0,
    VBAKE_INVALID_ARGUMENT = 1,
    VBAKE_DEGENERATE_SYSTEM = 2,
    VBAKE_NUMERICAL_FAILURE = 3,
};

enum VBakeTinyBvhLayout {
    VBAKE_TINYBVH_BVH = 0,
    VBAKE_TINYBVH_BVH8_CPU = 1,
};

VBAKE_API const char* vbake_last_error();

VBAKE_API int vbake_least_squares(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_values,
    int channels,
    float regularization_weight,
    float* out_vertex_values);

VBAKE_API int vbake_visibility_least_squares(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const float* normals,
    const float* tangents,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_raw_cones,
    float regularization_weight,
    float* out_vertex_cones,
    float* out_encoded_texcoord2);

VBAKE_API int vbake_pmr_visibility_sh_least_squares(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    const float* triangle_areas,
    int samples_per_triangle,
    const float* sample_barycentrics,
    const float* sample_sh16,
    float edge_regularization,
    float* out_vertex_sh16);

VBAKE_API int vbake_pmr_sh_to_cones(
    int vertex_count,
    const float* vertex_sh16,
    const float* fallback_normals,
    float* out_vertex_cones);

VBAKE_API int vbake_tinybvh_has_bvh8_cpu();

VBAKE_API int vbake_pmr_visibility_sh_tinybvh(
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
    double* out_build_milliseconds,
    double* out_trace_milliseconds,
    unsigned long long* out_visible_ray_count,
    int* out_thread_count);

}
