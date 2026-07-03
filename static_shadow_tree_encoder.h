#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#if defined(STATIC_SHADOW_TREE_ENCODER_BUILD)
#define SST_API __declspec(dllexport)
#else
#define SST_API __declspec(dllimport)
#endif
#else
#define SST_API __attribute__((visibility("default")))
#endif

extern "C" {

struct SSTEncoderOptions
{
    uint32_t width;
    uint32_t height;
    uint32_t tile_size;
    uint32_t min_leaf_size;
    float plane_error_threshold;
    float constant_epsilon;
    uint32_t use_dual_layer;
    uint32_t has_second_depth;
    uint32_t has_dual_depth_slack;
    float dual_depth_slack;
    uint32_t dual_conservative;
    uint32_t has_dual_max_leak;
    float dual_max_leak;
    float dual_visibility_tolerance;
    float shadow_bias;
    uint32_t plane_quantization_search_radius;
    uint32_t has_forced_leaf_error_cap;
    float forced_leaf_error_cap;
    uint32_t forced_split_bias_fit;
    uint32_t thread_count;
};

struct SSTEncoderStats
{
    uint32_t width;
    uint32_t height;
    uint32_t tile_grid_x;
    uint32_t tile_grid_y;
    uint32_t tile_size;
    uint32_t min_leaf_size;
    uint32_t max_tree_depth;
    uint32_t max_traversal_steps;
    uint32_t branch_10bit_start_level;
    uint32_t tile_count;
    uint32_t node_count;
    uint32_t branch_node_count;
    uint32_t branch_13bit_node_count;
    uint32_t branch_10bit_node_count;
    uint32_t plane_node_count;
    uint32_t uniform_plane_node_count;
    uint32_t compact_branch_words;
    uint32_t compact_30bit_plane_words;
    uint32_t compact_62bit_plane_words;
    uint32_t compact_node_words;
    uint32_t compact_tile_root_bytes;
    uint32_t compact_branch_offset_overflow_count;
    uint32_t fixed64_branch_offset_overflow_count;
    uint32_t max_compact_branch_offset;
    uint32_t max_fixed64_branch_offset;
    uint32_t compact_branch_13bit_max_offset;
    float compact_branch_13bit_capacity_percent;
    uint32_t compact_branch_10bit_max_offset;
    float compact_branch_10bit_capacity_percent;
    uint32_t forced_leaf_node_count;
    uint64_t forced_leaf_pixel_count;
    float forced_leaf_max_error;
    double forced_leaf_error_sum;
    uint64_t original_bytes;
    uint64_t encoded_bytes;
    uint64_t packed_encoded_bytes;
    uint64_t fixed64_encoded_bytes;
    uint64_t decompressed_depth_bytes;
    uint64_t packed_decompressed_working_set_bytes;
    float compression_ratio;
    float packed_compression_ratio;
    float fixed64_compression_ratio;
    float packed_decompressed_working_set_ratio;
    uint32_t packed_decode_valid;
    float max_error;
    float mean_error;
    float rmse_error;
};

struct SSTEncoderOutput
{
    uint8_t* nodes;
    size_t nodes_size;
    uint32_t* fixed64_nodes;
    size_t fixed64_word_count;
    uint32_t* compact_words;
    size_t compact_word_count;
    uint32_t* compact_roots;
    size_t compact_root_count;
    uint32_t* tile_roots;
    size_t tile_root_count;
    SSTEncoderStats stats;
    char* error_message;
};

SST_API int sst_encode(
    const float* depth,
    const float* second_depth,
    const SSTEncoderOptions* options,
    SSTEncoderOutput* output);

SST_API void sst_free_output(SSTEncoderOutput* output);

SST_API const char* sst_version();

}
