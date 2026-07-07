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

}
