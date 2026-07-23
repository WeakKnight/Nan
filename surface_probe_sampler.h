#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#if defined(SURFACE_PROBE_SAMPLER_BUILD)
#define SURFACE_PROBE_SAMPLER_API __declspec(dllexport)
#else
#define SURFACE_PROBE_SAMPLER_API __declspec(dllimport)
#endif
#else
#define SURFACE_PROBE_SAMPLER_API __attribute__((visibility("default")))
#endif

extern "C"
{
struct SurfaceProbeWSEOptions
{
    float surface_area;
    float normal_cosine_threshold;
    float plane_distance_scale;
};

struct SurfaceProbeAdaptiveWSEProfile
{
    double setup_ms;
    double stage1_partition_ms;
    double stage1_eliminate_wall_ms;
    double stage1_pack_cpu_ms;
    double stage1_grid_cpu_ms;
    double stage1_weights_cpu_ms;
    double stage1_heap_cpu_ms;
    double stage2_partition_ms;
    double stage2_eliminate_wall_ms;
    double stage2_pack_cpu_ms;
    double stage2_grid_cpu_ms;
    double stage2_weights_cpu_ms;
    double stage2_heap_cpu_ms;
    double final_pack_ms;
    double final_grid_ms;
    double final_weights_ms;
    double final_heap_ms;
    double total_ms;
    std::uint32_t stage1_input_count;
    std::uint32_t stage1_output_count;
    std::uint32_t stage2_output_count;
    std::uint32_t final_output_count;
    std::uint32_t stage1_partition_count;
    std::uint32_t stage2_partition_count;
    std::uint32_t parallel_path;
};

struct SurfaceProbeRepairOptions
{
    std::uint32_t min_gather_count;
    std::uint32_t max_repair_count;
    float normal_cosine_threshold;
    float weight_epsilon;
};

struct SurfaceProbeRepairProfile
{
    double acceleration_structure_ms;
    double base_gather_ms;
    double coverage_build_ms;
    double heap_build_ms;
    double greedy_select_ms;
    double affected_audits_ms;
    double final_gather_ms;
    double total_ms;
    std::uint64_t coverage_pair_count;
    std::uint32_t affected_audit_count;
    std::uint32_t worker_count;
};

struct SurfaceProbeCandidateFilterProfile
{
    double audit_partition_ms;
    double audit_deduplicate_ms;
    double repair_partition_ms;
    double repair_exclude_ms;
    double compact_ms;
    double total_ms;
    std::uint32_t audit_output_count;
    std::uint32_t repair_output_count;
    std::uint32_t shard_count;
    std::uint32_t worker_count;
};

struct SurfaceProbeSupportOptions
{
    float normal_cosine_threshold;
    float weight_epsilon;
    float max_density_multiplier;
};

struct SurfaceProbePointOctreeProfile
{
    double bounds_ms;
    double index_setup_ms;
    double partition_ms;
    double flatten_ms;
    double output_copy_ms;
    double total_ms;
    std::uint32_t worker_count;
    std::uint32_t node_count;
};

struct SurfaceProbePointOctreeResult
{
    std::uint32_t* nodes;
    std::uint32_t node_count;
    std::uint32_t* probe_order;
    std::uint32_t probe_count;
    double root_center[3];
    double root_extent;
};

SURFACE_PROBE_SAMPLER_API int surface_probe_wse_eliminate(
    const float* positions,
    const float* normals,
    std::uint32_t input_count,
    std::uint32_t output_count,
    const SurfaceProbeWSEOptions* options,
    std::uint32_t* output_indices,
    float* output_poisson_radius,
    char* error_message,
    std::size_t error_message_capacity);

SURFACE_PROBE_SAMPLER_API int surface_probe_wse_eliminate_adaptive(
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
    std::size_t error_message_capacity);

SURFACE_PROBE_SAMPLER_API int surface_probe_deficit_repair(
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
    std::size_t error_message_capacity);

SURFACE_PROBE_SAMPLER_API int surface_probe_filter_audit_repair_candidates(
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
    std::size_t error_message_capacity);

SURFACE_PROBE_SAMPLER_API int surface_probe_estimate_support(
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
    std::size_t error_message_capacity);

SURFACE_PROBE_SAMPLER_API int surface_probe_build_point_octree(
    const float* positions,
    std::uint32_t point_count,
    std::uint32_t leaf_capacity,
    std::uint32_t max_depth,
    SurfaceProbePointOctreeResult* output,
    SurfaceProbePointOctreeProfile* output_profile,
    char* error_message,
    std::size_t error_message_capacity);

SURFACE_PROBE_SAMPLER_API void surface_probe_free_point_octree(
    SurfaceProbePointOctreeResult* result);

SURFACE_PROBE_SAMPLER_API const char* surface_probe_wse_version();
}
