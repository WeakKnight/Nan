#include "surface_probe_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace
{
bool require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
    }
    return condition;
}
}

int main()
{
    constexpr std::uint32_t resolution = 100;
    constexpr std::uint32_t input_count = resolution * resolution;
    constexpr std::uint32_t output_count = 2000;
    std::vector<float> positions(static_cast<std::size_t>(input_count) * 3);
    std::vector<float> normals(static_cast<std::size_t>(input_count) * 3);
    std::vector<float> densities(input_count);
    for (std::uint32_t z = 0; z < resolution; ++z)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const std::uint32_t index = z * resolution + x;
            positions[index * 3] = static_cast<float>(x) / (resolution - 1);
            positions[index * 3 + 1] = 0.0f;
            positions[index * 3 + 2] = static_cast<float>(z) / (resolution - 1);
            normals[index * 3] = 0.0f;
            normals[index * 3 + 1] = 1.0f;
            normals[index * 3 + 2] = 0.0f;
            densities[index] = x < resolution / 2 ? 1.6f : 0.4f;
        }
    }

    const SurfaceProbeWSEOptions options{1.0f, 0.5f, 0.35f};
    std::vector<std::uint32_t> first(output_count);
    std::vector<std::uint32_t> second(output_count);
    float first_radius = 0.0f;
    float second_radius = 0.0f;
    SurfaceProbeAdaptiveWSEProfile adaptive_profile{};
    char error[2048]{};
    const int first_ok = surface_probe_wse_eliminate_adaptive(
        positions.data(),
        normals.data(),
        densities.data(),
        densities.data(),
        input_count,
        output_count,
        &options,
        first.data(),
        &first_radius,
        &adaptive_profile,
        error,
        sizeof(error));
    if (!require(first_ok != 0, error))
    {
        return 1;
    }
    const int second_ok = surface_probe_wse_eliminate_adaptive(
        positions.data(),
        normals.data(),
        densities.data(),
        densities.data(),
        input_count,
        output_count,
        &options,
        second.data(),
        &second_radius,
        nullptr,
        error,
        sizeof(error));

    bool passed = true;
    passed &= require(second_ok != 0, error);
    passed &= require(first == second, "parallel adaptive WSE is not deterministic");
    passed &= require(first.size() == output_count, "incorrect output count");
    passed &= require(
        std::set<std::uint32_t>(first.begin(), first.end()).size() == output_count,
        "output contains duplicate indices");
    passed &= require(first_radius > 0.0f, "Poisson radius is not positive");
    passed &= require(
        adaptive_profile.total_ms > 0.0 && adaptive_profile.parallel_path == 1 &&
            adaptive_profile.stage1_partition_count == 64 &&
            adaptive_profile.stage2_partition_count == 8,
        "adaptive WSE sampled profiling was not populated");
    passed &= require(
        std::abs(first_radius - second_radius) < 1.0e-7f,
        "Poisson radius changed between deterministic runs");

    const std::uint32_t high_density_count = static_cast<std::uint32_t>(
        std::count_if(first.begin(), first.end(), [](std::uint32_t index) {
            return index % resolution < resolution / 2;
        }));
    passed &= require(
        high_density_count >= 1550 && high_density_count <= 1700,
        "parallel partitioning does not preserve adaptive target density");

    const std::vector<float> filter_positions{
        0.0f, 0.0f, 0.0f,
        0.1f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.2f, 0.0f, 0.0f,
        1.1f, 0.0f, 0.0f};
    const std::vector<float> filter_normals{
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    const std::uint32_t filter_base_selected = 0;
    std::uint32_t audit_indices[5]{};
    std::uint32_t repair_filter_indices[5]{};
    std::uint32_t audit_filter_count = 0;
    std::uint32_t repair_filter_count = 0;
    SurfaceProbeCandidateFilterProfile filter_profile{};
    const int filter_ok = surface_probe_filter_audit_repair_candidates(
        filter_positions.data(),
        filter_normals.data(),
        5,
        filter_positions.data(),
        filter_normals.data(),
        5,
        &filter_base_selected,
        1,
        1.0f,
        0.5f,
        audit_indices,
        &audit_filter_count,
        repair_filter_indices,
        &repair_filter_count,
        &filter_profile,
        error,
        sizeof(error));
    passed &= require(filter_ok != 0, error);
    passed &= require(
        audit_filter_count == 3 && audit_indices[0] == 0 &&
            audit_indices[1] == 3 && audit_indices[2] == 4,
        "audit cell/normal deduplication changed semantics");
    passed &= require(
        repair_filter_count == 3 && repair_filter_indices[0] == 1 &&
            repair_filter_indices[1] == 3 && repair_filter_indices[2] == 4,
        "repair exact duplicate exclusion changed semantics");
    passed &= require(
        filter_profile.audit_output_count == 3 &&
            filter_profile.repair_output_count == 3,
        "candidate filter profile counts are incorrect");

    const std::vector<float> repair_base_positions{
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f};
    const std::vector<float> repair_base_normals{
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    const std::vector<std::uint32_t> repair_base_instances{0, 0};
    const std::vector<float> repair_candidate_positions{
        0.1f, 0.0f, 0.0f,
        0.2f, 0.0f, 0.0f,
        0.9f, 0.0f, 0.0f};
    const std::vector<float> repair_candidate_normals{
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    const std::vector<std::uint32_t> repair_candidate_instances{0, 0, 0};
    const float repair_radius = 0.5f;
    const SurfaceProbeRepairOptions repair_options{3, 3, 0.5f, 1.0e-6f};
    std::vector<std::uint32_t> repair_indices(3);
    std::uint32_t repair_count = 0;
    std::uint32_t counts_before[2]{};
    std::uint32_t counts_after[2]{};
    float weight_sums_before[2]{};
    float weight_sums_after[2]{};
    float ess_before[2]{};
    float ess_after[2]{};
    SurfaceProbeRepairProfile repair_profile{};
    const int repair_ok = surface_probe_deficit_repair(
        repair_base_positions.data(),
        repair_base_normals.data(),
        repair_base_instances.data(),
        2,
        repair_candidate_positions.data(),
        repair_candidate_normals.data(),
        repair_candidate_instances.data(),
        3,
        repair_base_positions.data(),
        repair_base_normals.data(),
        repair_base_instances.data(),
        2,
        &repair_radius,
        1,
        &repair_options,
        repair_indices.data(),
        &repair_count,
        counts_before,
        counts_after,
        weight_sums_before,
        weight_sums_after,
        ess_before,
        ess_after,
        &repair_profile,
        error,
        sizeof(error));
    passed &= require(repair_ok != 0, error);
    passed &= require(repair_count == 3, "repair did not select expected sites");
    passed &= require(
        counts_after[0] == 3 && counts_after[1] == 2,
        "incremental final gather statistics are incorrect");
    passed &= require(
        repair_profile.coverage_pair_count > 0,
        "repair coverage CSR was not populated");
    passed &= require(
        repair_profile.affected_audit_count == 2,
        "repair affected-audit discovery is incorrect");
    passed &= require(
        repair_profile.worker_count > 0 && repair_profile.total_ms >= 0.0,
        "repair profiling was not populated");

    const std::vector<float> octree_positions{
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         0.0f,  0.0f,  0.0f};
    SurfaceProbePointOctreeResult octree{};
    SurfaceProbePointOctreeProfile octree_profile{};
    const int octree_ok = surface_probe_build_point_octree(
        octree_positions.data(),
        9,
        1,
        8,
        &octree,
        &octree_profile,
        error,
        sizeof(error));
    passed &= require(octree_ok != 0, error);
    passed &= require(
        octree.node_count > 1 && octree.probe_count == 9,
        "point octree returned invalid counts");
    if (octree_ok != 0)
    {
        std::set<std::uint32_t> octree_order(
            octree.probe_order, octree.probe_order + octree.probe_count);
        passed &= require(
            octree_order.size() == 9 && *octree_order.begin() == 0 &&
                *octree_order.rbegin() == 8,
            "point octree probe order is not a permutation");
        passed &= require(
            octree.nodes[1] != 0 && octree.nodes[0] == 1,
            "point octree root children are not compact");
        passed &= require(
            octree_profile.total_ms > 0.0 &&
                octree_profile.node_count == octree.node_count,
            "point octree profiling was not populated");
    }
    surface_probe_free_point_octree(&octree);
    passed &= require(
        octree.nodes == nullptr && octree.probe_order == nullptr,
        "point octree buffers were not released");
    if (passed)
    {
        std::cout << "parallel adaptive WSE: " << input_count << " -> "
                  << output_count << ", high-density samples "
                  << high_density_count << '\n';
    }
    return passed ? 0 : 1;
}
