from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np


SST_NODE_FLAG_PLANE = 1 << 0
SST_NODE_FLAG_CONSTANT = 1 << 1
SST_INVALID_ROOT = 0xFFFFFFFF


@dataclass
class SparseShadowTreeStats:
    width: int
    height: int
    dual_layer: bool
    dual_second_hit_percent: float
    dual_raw_gap_mean_percent: float
    dual_raw_gap_p95_percent: float
    dual_raw_gap_max_percent: float
    dual_capped_gap_mean_percent: float
    dual_capped_gap_p95_percent: float
    dual_capped_gap_max_percent: float
    dual_slack_clamped_percent: float
    max_tree_depth: int
    max_traversal_steps: int
    branch_10bit_start_level: int
    tile_count: int
    node_count: int
    branch_node_count: int
    branch_13bit_node_count: int
    branch_10bit_node_count: int
    fixed64_branch_offset_overflow_count: int
    compact_branch_offset_overflow_count: int
    max_fixed64_branch_offset: int
    max_compact_branch_offset: int
    compact_branch_13bit_max_offset: int
    compact_branch_13bit_capacity_percent: float
    compact_branch_10bit_max_offset: int
    compact_branch_10bit_capacity_percent: float
    plane_node_count: int
    uniform_plane_node_count: int
    compact_branch_words: int
    compact_30bit_plane_words: int
    compact_62bit_plane_words: int
    compact_node_words: int
    compact_tile_root_bytes: int
    forced_leaf_node_count: int
    forced_leaf_pixel_percent: float
    forced_leaf_max_error_percent: float
    forced_leaf_mean_error_percent: float
    original_bytes: int
    encoded_bytes: int
    compression_ratio: float
    packed_encoded_bytes: int
    packed_compression_ratio: float
    decompressed_depth_bytes: int
    packed_decompressed_working_set_bytes: int
    packed_decompressed_working_set_ratio: float
    packed_decode_valid: bool
    packed_morton_decode_valid: bool
    packed_morton_max_delta_percent: float
    compact_fixed64_decode_valid: bool
    compact_fixed64_max_delta_percent: float
    packed_max_error_percent: float
    packed_mean_error_percent: float
    packed_rmse_error_percent: float
    packed_abs_error_percentiles: dict
    packed_abs_error_bias_cdf: tuple[dict, ...]
    packed_signed_error_bias_histogram: tuple[dict, ...]
    packed_leak_pixel_percent: float
    shadow_bias: float
    packed_leak_over_half_bias_percent: float
    packed_leak_over_full_bias_percent: float
    packed_conservative_over_half_bias_percent: float
    packed_conservative_over_full_bias_percent: float
    visibility_probe_offsets_in_bias: tuple[float, ...]
    packed_visibility_mismatch_percent: float
    packed_false_lit_percent: float
    packed_false_shadow_percent: float
    packed_visibility_probe_details: tuple[dict, ...]
    packed_depth_pcf3_mae_percent: float
    packed_depth_pcf3_max_error_percent: float
    packed_hard_vs_depth_pcf3_mae_percent: float
    packed_hard_vs_depth_pcf3_max_error_percent: float
    packed_depth_pcf3_probe_details: tuple[dict, ...]
    fixed64_encoded_bytes: int
    fixed64_compression_ratio: float
    fixed64_max_error: float
    fixed64_mean_error: float
    fixed64_rmse_error: float
    fixed64_max_error_percent: float
    fixed64_mean_error_percent: float
    fixed64_rmse_error_percent: float
    fixed64_abs_error_percentiles: dict
    fixed64_abs_error_bias_cdf: tuple[dict, ...]
    fixed64_signed_error_bias_histogram: tuple[dict, ...]
    fixed64_lossy_pixel_percent: float
    fixed64_leak_pixel_percent: float
    fixed64_max_leak_error_percent: float
    fixed64_mean_leak_error_percent: float
    fixed64_leak_over_half_bias_percent: float
    fixed64_leak_over_full_bias_percent: float
    fixed64_conservative_over_half_bias_percent: float
    fixed64_conservative_over_full_bias_percent: float
    fixed64_visibility_mismatch_percent: float
    fixed64_false_lit_percent: float
    fixed64_false_shadow_percent: float
    fixed64_visibility_probe_details: tuple[dict, ...]
    fixed64_conservative_pixel_percent: float
    fixed64_dual_interval_violation_percent: float
    fixed64_max_dual_interval_violation_percent: float
    fixed64_mean_dual_interval_violation_percent: float
    max_error: float
    mean_error: float
    rmse_error: float
    max_error_percent: float
    mean_error_percent: float
    rmse_error_percent: float
    lossy_pixel_percent: float
    leak_pixel_percent: float
    max_leak_error_percent: float
    mean_leak_error_percent: float
    conservative_pixel_percent: float
    mean_conservative_error_percent: float
    dual_interval_violation_percent: float
    max_dual_interval_violation_percent: float
    mean_dual_interval_violation_percent: float
    tile_diagnostics: dict


@dataclass
class SparseShadowTreeData:
    nodes: bytes
    packed_nodes: bytes
    compact_words: np.ndarray
    compact_tile_roots: np.ndarray
    fixed64_nodes: np.ndarray
    tile_roots: np.ndarray
    tile_grid: tuple[int, int]
    tile_size: int
    min_leaf_size: int
    stats: SparseShadowTreeStats


class SparseShadowTreeEncoder:
    NODE_STRUCT = struct.Struct("IIIIffff")

    def __init__(
        self,
        tile_size: int = 128,
        min_leaf_size: int = 1,
        plane_error_threshold: float = 0.0015,
        constant_epsilon: float = 0.0005,
        use_dual_layer: bool = True,
        dual_depth_slack: float | None = 0.0015,
        dual_conservative: bool = False,
        dual_max_leak: float | None = None,
        dual_visibility_tolerance: float = 0.0,
        shadow_bias: float = 0.0015,
        plane_quantization_search_radius: int = 0,
        forced_leaf_error_cap: float | None = None,
        forced_split_bias_fit: bool = False,
    ):
        self.tile_size = tile_size
        self.min_leaf_size = min_leaf_size
        self.plane_error_threshold = plane_error_threshold
        self.constant_epsilon = constant_epsilon
        self.use_dual_layer = use_dual_layer
        self.dual_depth_slack = dual_depth_slack
        self.dual_conservative = dual_conservative
        self.dual_max_leak = None if dual_max_leak is None else max(0.0, float(dual_max_leak))
        self.dual_max_leak_guard = min(1e-6, 0.1 * self.dual_max_leak) if self.dual_max_leak is not None else 0.0
        self.dual_visibility_tolerance = max(0.0, float(dual_visibility_tolerance))
        self.shadow_bias = max(0.0, float(shadow_bias))
        self.plane_quantization_search_radius = max(0, int(plane_quantization_search_radius))
        self.forced_leaf_error_cap = None if forced_leaf_error_cap is None else max(0.0, float(forced_leaf_error_cap))
        self.forced_split_bias_fit = bool(forced_split_bias_fit)
        self._nodes: list[tuple[int, int, int, int, float, float, float, float]] = []
        self._node_levels: list[int] = []
        self._upper_depth: np.ndarray | None = None
        self._max_tree_depth = self._compute_max_tree_depth()
        self._branch_10bit_start_level = self._compute_branch_10bit_start_level()

    @property
    def max_traversal_steps(self) -> int:
        return self._max_tree_depth + 1

    @property
    def branch_10bit_start_level(self) -> int:
        return self._branch_10bit_start_level

    def encode(self, depth: np.ndarray, second_depth: np.ndarray | None = None) -> SparseShadowTreeData:
        depth_2d = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_2d.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth_2d.shape}")
        second_2d = None
        raw_upper = depth_2d
        if self.use_dual_layer and second_depth is not None:
            second_2d = np.asarray(second_depth, dtype=np.float32).squeeze()
            if second_2d.shape != depth_2d.shape:
                raise ValueError(f"Second depth shape {second_2d.shape} does not match {depth_2d.shape}")
            raw_upper = np.maximum(second_2d, depth_2d)
            upper = raw_upper
            if self.dual_depth_slack is not None:
                upper = np.minimum(upper, depth_2d + max(0.0, float(self.dual_depth_slack)))
            self._upper_depth = np.maximum(upper, depth_2d)
        else:
            self._upper_depth = depth_2d

        height, width = depth_2d.shape
        tile_grid_x = math.ceil(width / self.tile_size)
        tile_grid_y = math.ceil(height / self.tile_size)
        tile_roots = np.full(tile_grid_x * tile_grid_y, SST_INVALID_ROOT, dtype=np.uint32)

        self._nodes = []
        self._node_levels = []
        self._forced_leaf_node_count = 0
        self._forced_leaf_pixel_count = 0
        self._forced_leaf_error_sum = 0.0
        self._forced_leaf_max_error = 0.0
        for tile_y in range(tile_grid_y):
            for tile_x in range(tile_grid_x):
                x0 = tile_x * self.tile_size
                y0 = tile_y * self.tile_size
                x1 = min(x0 + self.tile_size, width)
                y1 = min(y0 + self.tile_size, height)
                tile_roots[tile_y * tile_grid_x + tile_x] = self._encode_region(depth_2d, x0, y0, x1, y1)

        tile_roots = self._reorder_nodes_level_order(tile_roots)
        self._branch_10bit_start_level = self._find_safe_branch_10bit_start_level()
        encoded = b"".join(self.NODE_STRUCT.pack(*node) for node in self._nodes)
        compact_words, compact_roots, compact_overflow_count, max_compact_offset = self._pack_compact_nodes(tile_roots)
        packed_nodes = compact_words.tobytes()
        fixed64_nodes = self._pack_fixed64_nodes()
        decoded = self.decode(depth_2d.shape, encoded, tile_roots, (tile_grid_x, tile_grid_y))
        decoded_packed = self.decode_compact(depth_2d.shape, compact_words, compact_roots, (tile_grid_x, tile_grid_y))
        decoded_packed_morton = self.decode_compact_morton(
            depth_2d.shape,
            compact_words,
            compact_roots,
            (tile_grid_x, tile_grid_y),
        )
        decoded_fixed64 = self.decode_fixed64(depth_2d.shape, fixed64_nodes, tile_roots, (tile_grid_x, tile_grid_y))
        error = decoded - depth_2d
        abs_error = np.abs(error)
        leak_mask = error > 1e-6
        leak_error = np.maximum(error, 0.0)
        conservative_error = np.maximum(depth_2d - decoded, 0.0)
        upper = self._upper_depth if self._upper_depth is not None else depth_2d
        interval_violation = np.maximum(np.maximum(depth_2d - decoded, 0.0), np.maximum(decoded - upper, 0.0))
        packed_error = decoded_packed - depth_2d
        packed_abs_error = np.abs(packed_error)
        packed_leak_error = np.maximum(packed_error, 0.0)
        packed_conservative_error = np.maximum(-packed_error, 0.0)
        packed_morton_delta = np.abs(decoded_packed_morton - decoded_packed)
        visibility_probe_offsets = (0.0, 0.5, 1.0, 2.0)
        packed_visibility = self._compute_visibility_mismatch_stats(
            depth_2d,
            decoded_packed,
            visibility_probe_offsets,
        )
        packed_pcf3_visibility = self._compute_pcf3_visibility_delta_stats(
            depth_2d,
            decoded_packed,
            visibility_probe_offsets,
        )
        fixed64_error = decoded_fixed64 - depth_2d
        compact_fixed64_delta = np.abs(decoded_packed - decoded_fixed64)
        fixed64_abs_error = np.abs(fixed64_error)
        fixed64_leak_error = np.maximum(fixed64_error, 0.0)
        fixed64_conservative_error = np.maximum(-fixed64_error, 0.0)
        fixed64_visibility = self._compute_visibility_mismatch_stats(
            depth_2d,
            decoded_fixed64,
            visibility_probe_offsets,
        )
        fixed64_interval_violation = np.maximum(
            np.maximum(depth_2d - decoded_fixed64, 0.0),
            np.maximum(decoded_fixed64 - upper, 0.0),
        )
        half_bias = self.shadow_bias * 0.5
        branch_count = sum(1 for node in self._nodes if (node[0] & SST_NODE_FLAG_PLANE) == 0)
        branch_13bit_count = sum(
            1
            for index, node in enumerate(self._nodes)
            if (node[0] & SST_NODE_FLAG_PLANE) == 0 and self._branch_offset_bits_for_level(self._node_levels[index]) == 13
        )
        branch_10bit_count = branch_count - branch_13bit_count
        plane_count = len(self._nodes) - branch_count
        uniform_count = sum(1 for node in self._nodes if (node[0] & SST_NODE_FLAG_CONSTANT) != 0)
        plane62_count = plane_count - uniform_count
        compact_branch_words = branch_count * 2
        compact_30bit_plane_words = uniform_count
        compact_62bit_plane_words = plane62_count * 2
        compact_node_words = int(compact_words.size)
        compact_tile_root_bytes = int(compact_roots.nbytes)
        branch_offset_diagnostics = self._compute_compact_branch_offset_diagnostics()
        fixed64_overflow_count, max_fixed64_offset = self._count_fixed64_branch_offset_overflow()
        original_bytes = width * height * 4
        encoded_bytes = len(encoded) + tile_roots.nbytes
        packed_encoded_bytes = int(compact_words.nbytes + compact_roots.nbytes)
        fixed64_encoded_bytes = int(fixed64_nodes.nbytes + tile_roots.nbytes)
        decompressed_depth_bytes = original_bytes
        packed_decompressed_working_set_bytes = packed_encoded_bytes + decompressed_depth_bytes
        pixel_count = max(width * height, 1)
        dual_stats = self._compute_dual_layer_utilization(depth_2d, second_2d, raw_upper, upper)
        tile_diagnostics = self._compute_tile_diagnostics(
            depth_2d.shape,
            packed_error,
            compact_words,
            compact_roots,
            (tile_grid_x, tile_grid_y),
        )
        stats = SparseShadowTreeStats(
            width=width,
            height=height,
            dual_layer=bool(self.use_dual_layer and second_depth is not None),
            dual_second_hit_percent=dual_stats["second_hit_percent"],
            dual_raw_gap_mean_percent=dual_stats["raw_gap_mean_percent"],
            dual_raw_gap_p95_percent=dual_stats["raw_gap_p95_percent"],
            dual_raw_gap_max_percent=dual_stats["raw_gap_max_percent"],
            dual_capped_gap_mean_percent=dual_stats["capped_gap_mean_percent"],
            dual_capped_gap_p95_percent=dual_stats["capped_gap_p95_percent"],
            dual_capped_gap_max_percent=dual_stats["capped_gap_max_percent"],
            dual_slack_clamped_percent=dual_stats["slack_clamped_percent"],
            max_tree_depth=self._max_tree_depth,
            max_traversal_steps=self.max_traversal_steps,
            branch_10bit_start_level=self.branch_10bit_start_level,
            tile_count=tile_grid_x * tile_grid_y,
            node_count=len(self._nodes),
            branch_node_count=branch_count,
            branch_13bit_node_count=branch_13bit_count,
            branch_10bit_node_count=branch_10bit_count,
            fixed64_branch_offset_overflow_count=fixed64_overflow_count,
            compact_branch_offset_overflow_count=compact_overflow_count,
            max_fixed64_branch_offset=max_fixed64_offset,
            max_compact_branch_offset=max_compact_offset,
            compact_branch_13bit_max_offset=branch_offset_diagnostics["13bit_max_offset"],
            compact_branch_13bit_capacity_percent=branch_offset_diagnostics["13bit_capacity_percent"],
            compact_branch_10bit_max_offset=branch_offset_diagnostics["10bit_max_offset"],
            compact_branch_10bit_capacity_percent=branch_offset_diagnostics["10bit_capacity_percent"],
            plane_node_count=plane_count,
            uniform_plane_node_count=uniform_count,
            compact_branch_words=compact_branch_words,
            compact_30bit_plane_words=compact_30bit_plane_words,
            compact_62bit_plane_words=compact_62bit_plane_words,
            compact_node_words=compact_node_words,
            compact_tile_root_bytes=compact_tile_root_bytes,
            forced_leaf_node_count=int(self._forced_leaf_node_count),
            forced_leaf_pixel_percent=float(self._forced_leaf_pixel_count * 100.0 / float(pixel_count)),
            forced_leaf_max_error_percent=float(self._forced_leaf_max_error * 100.0),
            forced_leaf_mean_error_percent=float(
                (self._forced_leaf_error_sum / float(max(self._forced_leaf_pixel_count, 1))) * 100.0
            ),
            original_bytes=original_bytes,
            encoded_bytes=encoded_bytes,
            compression_ratio=(original_bytes / encoded_bytes) if encoded_bytes > 0 else 0.0,
            packed_encoded_bytes=packed_encoded_bytes,
            packed_compression_ratio=(original_bytes / packed_encoded_bytes) if packed_encoded_bytes > 0 else 0.0,
            decompressed_depth_bytes=decompressed_depth_bytes,
            packed_decompressed_working_set_bytes=packed_decompressed_working_set_bytes,
            packed_decompressed_working_set_ratio=(
                original_bytes / packed_decompressed_working_set_bytes
                if packed_decompressed_working_set_bytes > 0
                else 0.0
            ),
            packed_decode_valid=compact_overflow_count == 0,
            packed_morton_decode_valid=bool(np.max(packed_morton_delta) <= 1e-7) if packed_morton_delta.size else True,
            packed_morton_max_delta_percent=float(np.max(packed_morton_delta) * 100.0) if packed_morton_delta.size else 0.0,
            compact_fixed64_decode_valid=bool(np.max(compact_fixed64_delta) <= 1e-7) if compact_fixed64_delta.size else True,
            compact_fixed64_max_delta_percent=float(np.max(compact_fixed64_delta) * 100.0) if compact_fixed64_delta.size else 0.0,
            packed_max_error_percent=float(np.max(packed_abs_error) * 100.0) if packed_error.size else 0.0,
            packed_mean_error_percent=float(np.mean(packed_abs_error) * 100.0) if packed_error.size else 0.0,
            packed_rmse_error_percent=float(np.sqrt(np.mean(packed_error * packed_error)) * 100.0) if packed_error.size else 0.0,
            packed_abs_error_percentiles=self._compute_error_percentiles(packed_abs_error),
            packed_abs_error_bias_cdf=self._compute_abs_error_bias_cdf(packed_abs_error),
            packed_signed_error_bias_histogram=self._compute_signed_error_bias_histogram(packed_error),
            packed_leak_pixel_percent=float(np.mean(packed_error > 1e-6) * 100.0) if packed_error.size else 0.0,
            shadow_bias=self.shadow_bias,
            packed_leak_over_half_bias_percent=float(np.mean(packed_leak_error > half_bias) * 100.0) if packed_error.size else 0.0,
            packed_leak_over_full_bias_percent=float(np.mean(packed_leak_error > self.shadow_bias) * 100.0) if packed_error.size else 0.0,
            packed_conservative_over_half_bias_percent=float(np.mean(packed_conservative_error > half_bias) * 100.0) if packed_error.size else 0.0,
            packed_conservative_over_full_bias_percent=float(np.mean(packed_conservative_error > self.shadow_bias) * 100.0) if packed_error.size else 0.0,
            visibility_probe_offsets_in_bias=visibility_probe_offsets,
            packed_visibility_mismatch_percent=packed_visibility["mismatch_percent"],
            packed_false_lit_percent=packed_visibility["false_lit_percent"],
            packed_false_shadow_percent=packed_visibility["false_shadow_percent"],
            packed_visibility_probe_details=packed_visibility["probe_details"],
            packed_depth_pcf3_mae_percent=packed_pcf3_visibility["pcf3_mae_percent"],
            packed_depth_pcf3_max_error_percent=packed_pcf3_visibility["pcf3_max_error_percent"],
            packed_hard_vs_depth_pcf3_mae_percent=packed_pcf3_visibility["hard_vs_pcf3_mae_percent"],
            packed_hard_vs_depth_pcf3_max_error_percent=packed_pcf3_visibility["hard_vs_pcf3_max_error_percent"],
            packed_depth_pcf3_probe_details=packed_pcf3_visibility["probe_details"],
            fixed64_encoded_bytes=fixed64_encoded_bytes,
            fixed64_compression_ratio=(original_bytes / fixed64_encoded_bytes) if fixed64_encoded_bytes > 0 else 0.0,
            fixed64_max_error=float(np.max(fixed64_abs_error)) if fixed64_error.size else 0.0,
            fixed64_mean_error=float(np.mean(fixed64_abs_error)) if fixed64_error.size else 0.0,
            fixed64_rmse_error=float(np.sqrt(np.mean(fixed64_error * fixed64_error))) if fixed64_error.size else 0.0,
            fixed64_max_error_percent=float(np.max(fixed64_abs_error) * 100.0) if fixed64_error.size else 0.0,
            fixed64_mean_error_percent=float(np.mean(fixed64_abs_error) * 100.0) if fixed64_error.size else 0.0,
            fixed64_rmse_error_percent=float(np.sqrt(np.mean(fixed64_error * fixed64_error)) * 100.0) if fixed64_error.size else 0.0,
            fixed64_abs_error_percentiles=self._compute_error_percentiles(fixed64_abs_error),
            fixed64_abs_error_bias_cdf=self._compute_abs_error_bias_cdf(fixed64_abs_error),
            fixed64_signed_error_bias_histogram=self._compute_signed_error_bias_histogram(fixed64_error),
            fixed64_lossy_pixel_percent=float(np.mean(fixed64_abs_error > 1e-6) * 100.0) if fixed64_error.size else 0.0,
            fixed64_leak_pixel_percent=float(np.mean(fixed64_error > 1e-6) * 100.0) if fixed64_error.size else 0.0,
            fixed64_max_leak_error_percent=float(np.max(fixed64_leak_error) * 100.0) if fixed64_error.size else 0.0,
            fixed64_mean_leak_error_percent=float(np.mean(fixed64_leak_error) * 100.0) if fixed64_error.size else 0.0,
            fixed64_leak_over_half_bias_percent=float(np.mean(fixed64_leak_error > half_bias) * 100.0) if fixed64_error.size else 0.0,
            fixed64_leak_over_full_bias_percent=float(np.mean(fixed64_leak_error > self.shadow_bias) * 100.0) if fixed64_error.size else 0.0,
            fixed64_conservative_over_half_bias_percent=float(np.mean(fixed64_conservative_error > half_bias) * 100.0) if fixed64_error.size else 0.0,
            fixed64_conservative_over_full_bias_percent=float(np.mean(fixed64_conservative_error > self.shadow_bias) * 100.0) if fixed64_error.size else 0.0,
            fixed64_visibility_mismatch_percent=fixed64_visibility["mismatch_percent"],
            fixed64_false_lit_percent=fixed64_visibility["false_lit_percent"],
            fixed64_false_shadow_percent=fixed64_visibility["false_shadow_percent"],
            fixed64_visibility_probe_details=fixed64_visibility["probe_details"],
            fixed64_conservative_pixel_percent=float(np.mean(fixed64_error < -1e-6) * 100.0) if fixed64_error.size else 0.0,
            fixed64_dual_interval_violation_percent=float(np.mean(fixed64_interval_violation > 1e-6) * 100.0) if fixed64_error.size else 0.0,
            fixed64_max_dual_interval_violation_percent=float(np.max(fixed64_interval_violation) * 100.0) if fixed64_error.size else 0.0,
            fixed64_mean_dual_interval_violation_percent=float(np.mean(fixed64_interval_violation) * 100.0) if fixed64_error.size else 0.0,
            max_error=float(np.max(abs_error)) if error.size else 0.0,
            mean_error=float(np.mean(abs_error)) if error.size else 0.0,
            rmse_error=float(np.sqrt(np.mean(error * error))) if error.size else 0.0,
            max_error_percent=float(np.max(abs_error) * 100.0) if error.size else 0.0,
            mean_error_percent=float(np.mean(abs_error) * 100.0) if error.size else 0.0,
            rmse_error_percent=float(np.sqrt(np.mean(error * error)) * 100.0) if error.size else 0.0,
            lossy_pixel_percent=float(np.mean(abs_error > 1e-6) * 100.0) if error.size else 0.0,
            leak_pixel_percent=float(np.mean(leak_mask) * 100.0) if error.size else 0.0,
            max_leak_error_percent=float(np.max(leak_error) * 100.0) if error.size else 0.0,
            mean_leak_error_percent=float(np.mean(leak_error) * 100.0) if error.size else 0.0,
            conservative_pixel_percent=float(np.mean(error < -1e-6) * 100.0) if error.size else 0.0,
            mean_conservative_error_percent=float(np.mean(conservative_error) * 100.0) if error.size else 0.0,
            dual_interval_violation_percent=float(np.mean(interval_violation > 1e-6) * 100.0) if error.size else 0.0,
            max_dual_interval_violation_percent=float(np.max(interval_violation) * 100.0) if error.size else 0.0,
            mean_dual_interval_violation_percent=float(np.mean(interval_violation) * 100.0) if error.size else 0.0,
            tile_diagnostics=tile_diagnostics,
        )
        return SparseShadowTreeData(
            nodes=encoded,
            packed_nodes=packed_nodes,
            compact_words=compact_words,
            compact_tile_roots=compact_roots,
            fixed64_nodes=fixed64_nodes,
            tile_roots=tile_roots,
            tile_grid=(tile_grid_x, tile_grid_y),
            tile_size=self.tile_size,
            min_leaf_size=self.min_leaf_size,
            stats=stats,
        )

    def _compute_max_tree_depth(self) -> int:
        if self.tile_size <= 1 or self.min_leaf_size >= self.tile_size:
            return 0
        return int(math.ceil(math.log2(float(self.tile_size) / float(max(self.min_leaf_size, 1)))))

    def _compute_branch_10bit_start_level(self) -> int:
        for level in range(self._max_tree_depth + 1):
            child_depth = max(self._max_tree_depth - level - 1, 0)
            child_subtree_nodes = (4 ** (child_depth + 1) - 1) // 3
            max_child_offset = 1 + 3 * child_subtree_nodes
            if max_child_offset <= 0x3FF:
                return level
        return self._max_tree_depth + 1

    def _branch_offset_bits_for_level(self, level: int) -> int:
        return 10 if level >= self._branch_10bit_start_level else 13

    def _compute_node_word_offsets(self) -> list[int]:
        node_word_offsets: list[int] = []
        word_count = 0
        for flags, *_ in self._nodes:
            node_word_offsets.append(word_count)
            word_count += 1 if (flags & SST_NODE_FLAG_CONSTANT) else 2
        return node_word_offsets

    def _compute_dual_layer_utilization(
        self,
        depth: np.ndarray,
        second_depth: np.ndarray | None,
        raw_upper: np.ndarray,
        capped_upper: np.ndarray,
    ) -> dict[str, float]:
        if second_depth is None or depth.size == 0:
            return {
                "second_hit_percent": 0.0,
                "raw_gap_mean_percent": 0.0,
                "raw_gap_p95_percent": 0.0,
                "raw_gap_max_percent": 0.0,
                "capped_gap_mean_percent": 0.0,
                "capped_gap_p95_percent": 0.0,
                "capped_gap_max_percent": 0.0,
                "slack_clamped_percent": 0.0,
            }

        raw_gap = np.maximum(np.asarray(raw_upper, dtype=np.float32) - depth, 0.0)
        capped_gap = np.maximum(np.asarray(capped_upper, dtype=np.float32) - depth, 0.0)
        second = np.asarray(second_depth, dtype=np.float32)
        second_hit = (second < 1.0 - 1e-6) & (raw_gap > 1e-7)
        slack_clamped = raw_gap > capped_gap + 1e-7

        def percentile_percent(values: np.ndarray, percentile: float) -> float:
            return float(np.percentile(values, percentile) * 100.0) if values.size else 0.0

        return {
            "second_hit_percent": float(np.mean(second_hit) * 100.0),
            "raw_gap_mean_percent": float(np.mean(raw_gap) * 100.0),
            "raw_gap_p95_percent": percentile_percent(raw_gap, 95.0),
            "raw_gap_max_percent": float(np.max(raw_gap) * 100.0) if raw_gap.size else 0.0,
            "capped_gap_mean_percent": float(np.mean(capped_gap) * 100.0),
            "capped_gap_p95_percent": percentile_percent(capped_gap, 95.0),
            "capped_gap_max_percent": float(np.max(capped_gap) * 100.0) if capped_gap.size else 0.0,
            "slack_clamped_percent": float(np.mean(slack_clamped) * 100.0),
        }

    def _compute_tile_diagnostics(
        self,
        shape: tuple[int, int],
        packed_error: np.ndarray,
        compact_words: np.ndarray,
        compact_roots: np.ndarray,
        tile_grid: tuple[int, int],
    ) -> dict:
        height, width = shape
        tile_grid_x, tile_grid_y = tile_grid
        roots = [int(root) for root in np.asarray(compact_roots, dtype=np.uint32).reshape(-1)]
        valid_roots = sorted(root for root in roots if root != SST_INVALID_ROOT)
        word_count = int(np.asarray(compact_words, dtype=np.uint32).size)
        next_root_by_root = {}
        for index, root in enumerate(valid_roots):
            next_root_by_root[root] = valid_roots[index + 1] if index + 1 < len(valid_roots) else word_count

        records: list[dict] = []
        for tile_y in range(tile_grid_y):
            for tile_x in range(tile_grid_x):
                tile_index = tile_y * tile_grid_x + tile_x
                if tile_index >= len(roots) or roots[tile_index] == SST_INVALID_ROOT:
                    continue
                x0 = tile_x * self.tile_size
                y0 = tile_y * self.tile_size
                x1 = min(x0 + self.tile_size, width)
                y1 = min(y0 + self.tile_size, height)
                tile_error = packed_error[y0:y1, x0:x1]
                if tile_error.size == 0:
                    continue

                abs_error = np.abs(tile_error)
                leak_error = np.maximum(tile_error, 0.0)
                compact_word_start = roots[tile_index]
                compact_word_end = next_root_by_root.get(compact_word_start, compact_word_start)
                compact_bytes = max(0, compact_word_end - compact_word_start) * 4 + 4
                original_bytes = (x1 - x0) * (y1 - y0) * 4
                record = {
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "pixel_count": int(tile_error.size),
                    "compact_bytes": int(compact_bytes),
                    "original_bytes": int(original_bytes),
                    "compression_ratio": float(original_bytes / compact_bytes) if compact_bytes > 0 else 0.0,
                    "mean_error_percent": float(np.mean(abs_error) * 100.0),
                    "rmse_error_percent": float(np.sqrt(np.mean(tile_error * tile_error)) * 100.0),
                    "max_error_percent": float(np.max(abs_error) * 100.0),
                    "leak_over_full_bias_percent": float(np.mean(leak_error > self.shadow_bias) * 100.0),
                }
                records.append(record)

        if not records:
            return {
                "tile_count": 0,
                "top_max_error_tiles": [],
                "top_leak_over_bias_tiles": [],
                "lowest_compression_tiles": [],
                "compression_ratio_percentiles": {},
                "max_error_percentiles": {},
                "mean_error_percentiles": {},
                "leak_over_full_bias_percentiles": {},
            }

        def percentiles(key: str) -> dict:
            values = np.array([record[key] for record in records], dtype=np.float32)
            return {
                "min": float(np.min(values)),
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }

        return {
            "tile_count": len(records),
            "top_max_error_tiles": sorted(records, key=lambda record: record["max_error_percent"], reverse=True)[:8],
            "top_leak_over_bias_tiles": sorted(records, key=lambda record: record["leak_over_full_bias_percent"], reverse=True)[:8],
            "lowest_compression_tiles": sorted(records, key=lambda record: record["compression_ratio"])[:8],
            "compression_ratio_percentiles": percentiles("compression_ratio"),
            "max_error_percentiles": percentiles("max_error_percent"),
            "mean_error_percentiles": percentiles("mean_error_percent"),
            "leak_over_full_bias_percentiles": percentiles("leak_over_full_bias_percent"),
        }

    def _compute_visibility_mismatch_stats(
        self,
        reference_depth: np.ndarray,
        encoded_depth: np.ndarray,
        offsets_in_bias: tuple[float, ...],
    ) -> dict[str, float]:
        if reference_depth.size == 0 or not offsets_in_bias:
            return {
                "mismatch_percent": 0.0,
                "false_lit_percent": 0.0,
                "false_shadow_percent": 0.0,
                "probe_details": tuple(),
            }

        bias = max(self.shadow_bias, 0.0)
        mismatch_count = 0
        false_lit_count = 0
        false_shadow_count = 0
        total_count = 0
        probe_details: list[dict] = []

        for offset in offsets_in_bias:
            receiver_depth = reference_depth + bias * float(offset)
            reference_visible = receiver_depth <= reference_depth + bias
            encoded_visible = receiver_depth <= encoded_depth + bias
            mismatch = reference_visible != encoded_visible
            probe_mismatch_count = int(np.count_nonzero(mismatch))
            probe_false_lit_count = int(np.count_nonzero(mismatch & encoded_visible))
            probe_false_shadow_count = int(np.count_nonzero(mismatch & ~encoded_visible))
            probe_total_count = int(reference_depth.size)
            mismatch_count += probe_mismatch_count
            false_lit_count += probe_false_lit_count
            false_shadow_count += probe_false_shadow_count
            total_count += probe_total_count
            probe_scale = 100.0 / float(max(probe_total_count, 1))
            probe_details.append(
                {
                    "offset_in_bias": float(offset),
                    "mismatch_percent": float(probe_mismatch_count * probe_scale),
                    "false_lit_percent": float(probe_false_lit_count * probe_scale),
                    "false_shadow_percent": float(probe_false_shadow_count * probe_scale),
                    "reference_visible_percent": float(np.mean(reference_visible) * 100.0),
                    "encoded_visible_percent": float(np.mean(encoded_visible) * 100.0),
                }
            )

        if total_count == 0:
            return {
                "mismatch_percent": 0.0,
                "false_lit_percent": 0.0,
                "false_shadow_percent": 0.0,
                "probe_details": tuple(),
            }

        scale = 100.0 / float(total_count)
        return {
            "mismatch_percent": float(mismatch_count * scale),
            "false_lit_percent": float(false_lit_count * scale),
            "false_shadow_percent": float(false_shadow_count * scale),
            "probe_details": tuple(probe_details),
        }

    def _compute_pcf3_visibility_delta_stats(
        self,
        reference_depth: np.ndarray,
        encoded_depth: np.ndarray,
        offsets_in_bias: tuple[float, ...],
    ) -> dict[str, float]:
        if reference_depth.size == 0 or not offsets_in_bias:
            return {
                "pcf3_mae_percent": 0.0,
                "pcf3_max_error_percent": 0.0,
                "hard_vs_pcf3_mae_percent": 0.0,
                "hard_vs_pcf3_max_error_percent": 0.0,
                "probe_details": tuple(),
            }

        bias = max(self.shadow_bias, 0.0)
        pcf3_abs_sum = 0.0
        hard_abs_sum = 0.0
        pcf3_max_error = 0.0
        hard_max_error = 0.0
        total_count = 0
        probe_details: list[dict] = []

        for offset in offsets_in_bias:
            receiver_depth = reference_depth + bias * float(offset)
            reference_pcf3 = self._compute_pcf3_visibility(reference_depth, receiver_depth)
            encoded_pcf3 = self._compute_pcf3_visibility(encoded_depth, receiver_depth)
            encoded_hard = (receiver_depth <= encoded_depth + bias).astype(np.float32)

            pcf3_abs_delta = np.abs(encoded_pcf3 - reference_pcf3)
            hard_abs_delta = np.abs(encoded_hard - reference_pcf3)
            probe_count = int(reference_depth.size)
            probe_pcf3_max = float(np.max(pcf3_abs_delta)) if pcf3_abs_delta.size else 0.0
            probe_hard_max = float(np.max(hard_abs_delta)) if hard_abs_delta.size else 0.0

            pcf3_abs_sum += float(np.sum(pcf3_abs_delta))
            hard_abs_sum += float(np.sum(hard_abs_delta))
            pcf3_max_error = max(pcf3_max_error, probe_pcf3_max)
            hard_max_error = max(hard_max_error, probe_hard_max)
            total_count += probe_count
            probe_details.append(
                {
                    "offset_in_bias": float(offset),
                    "pcf3_mae_percent": float(np.mean(pcf3_abs_delta) * 100.0),
                    "pcf3_max_error_percent": probe_pcf3_max * 100.0,
                    "hard_vs_pcf3_mae_percent": float(np.mean(hard_abs_delta) * 100.0),
                    "hard_vs_pcf3_max_error_percent": probe_hard_max * 100.0,
                    "reference_pcf3_visible_percent": float(np.mean(reference_pcf3) * 100.0),
                    "encoded_pcf3_visible_percent": float(np.mean(encoded_pcf3) * 100.0),
                    "encoded_hard_visible_percent": float(np.mean(encoded_hard) * 100.0),
                }
            )

        if total_count == 0:
            return {
                "pcf3_mae_percent": 0.0,
                "pcf3_max_error_percent": 0.0,
                "hard_vs_pcf3_mae_percent": 0.0,
                "hard_vs_pcf3_max_error_percent": 0.0,
                "probe_details": tuple(),
            }

        scale = 100.0 / float(total_count)
        return {
            "pcf3_mae_percent": float(pcf3_abs_sum * scale),
            "pcf3_max_error_percent": float(pcf3_max_error * 100.0),
            "hard_vs_pcf3_mae_percent": float(hard_abs_sum * scale),
            "hard_vs_pcf3_max_error_percent": float(hard_max_error * 100.0),
            "probe_details": tuple(probe_details),
        }

    def _compute_pcf3_visibility(self, depth: np.ndarray, receiver_depth: np.ndarray) -> np.ndarray:
        padded = np.pad(depth, ((1, 1), (1, 1)), mode="edge")
        visibility = np.zeros(depth.shape, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                neighbor_depth = padded[dy : dy + depth.shape[0], dx : dx + depth.shape[1]]
                visibility += (receiver_depth <= neighbor_depth + self.shadow_bias).astype(np.float32)
        return visibility * (1.0 / 9.0)

    def _compute_error_percentiles(self, abs_error: np.ndarray) -> dict[str, float]:
        if abs_error.size == 0:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}
        return {
            "p50": float(np.percentile(abs_error, 50.0) * 100.0),
            "p90": float(np.percentile(abs_error, 90.0) * 100.0),
            "p95": float(np.percentile(abs_error, 95.0) * 100.0),
            "p99": float(np.percentile(abs_error, 99.0) * 100.0),
            "p999": float(np.percentile(abs_error, 99.9) * 100.0),
        }

    def _compute_abs_error_bias_cdf(
        self,
        abs_error: np.ndarray,
        thresholds_in_bias: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    ) -> tuple[dict, ...]:
        if abs_error.size == 0:
            return tuple(
                {"threshold_in_bias": float(threshold), "within_percent": 100.0}
                for threshold in thresholds_in_bias
            )

        bias = max(float(self.shadow_bias), 1e-12)
        return tuple(
            {
                "threshold_in_bias": float(threshold),
                "within_percent": float(np.mean(abs_error <= bias * float(threshold)) * 100.0),
            }
            for threshold in thresholds_in_bias
        )

    def _compute_signed_error_bias_histogram(
        self,
        error: np.ndarray,
        edges_in_bias: tuple[float, ...] = (-math.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, math.inf),
    ) -> tuple[dict, ...]:
        if len(edges_in_bias) < 2:
            return tuple()

        if error.size == 0:
            return tuple(
                {
                    "min_in_bias": float(edges_in_bias[i]),
                    "max_in_bias": float(edges_in_bias[i + 1]),
                    "percent": 0.0,
                }
                for i in range(len(edges_in_bias) - 1)
            )

        bias = max(float(self.shadow_bias), 1e-12)
        normalized_error = error / bias
        buckets: list[dict] = []
        for i in range(len(edges_in_bias) - 1):
            lower = float(edges_in_bias[i])
            upper = float(edges_in_bias[i + 1])
            if math.isinf(lower):
                mask = normalized_error < upper
            elif math.isinf(upper):
                mask = normalized_error >= lower
            else:
                mask = (normalized_error >= lower) & (normalized_error < upper)
            buckets.append(
                {
                    "min_in_bias": lower,
                    "max_in_bias": upper,
                    "percent": float(np.mean(mask) * 100.0),
                }
            )
        return tuple(buckets)

    def _find_safe_branch_10bit_start_level(self) -> int:
        node_word_offsets = self._compute_node_word_offsets()
        for start_level in range(self._max_tree_depth + 2):
            has_overflow = False
            for node_index, node in enumerate(self._nodes):
                flags, child_base, *_ = node
                if flags & SST_NODE_FLAG_PLANE:
                    continue
                bits = 10 if self._node_levels[node_index] >= start_level else 13
                max_value = (1 << bits) - 1
                for child in range(4):
                    child_index = child_base + child
                    node_relative_offset = child_index - node_index
                    word_relative_offset = node_word_offsets[child_index] - node_word_offsets[node_index]
                    if node_relative_offset > max_value or word_relative_offset > max_value:
                        has_overflow = True
                        break
                if has_overflow:
                    break
            if not has_overflow:
                return start_level
        return self._max_tree_depth + 1

    def _compute_compact_branch_offset_diagnostics(self) -> dict[str, float | int]:
        node_word_offsets = self._compute_node_word_offsets()
        max_13bit_offset = 0
        max_10bit_offset = 0
        for node_index, node in enumerate(self._nodes):
            flags, child_base, *_ = node
            if flags & SST_NODE_FLAG_PLANE:
                continue
            bits = self._branch_offset_bits_for_level(self._node_levels[node_index])
            word_offset = node_word_offsets[node_index]
            for child in range(4):
                child_index = child_base + child
                relative_offset = node_word_offsets[child_index] - word_offset
                if bits == 10:
                    max_10bit_offset = max(max_10bit_offset, relative_offset)
                else:
                    max_13bit_offset = max(max_13bit_offset, relative_offset)

        return {
            "13bit_max_offset": int(max_13bit_offset),
            "13bit_capacity_percent": float(max_13bit_offset * 100.0 / float((1 << 13) - 1)),
            "10bit_max_offset": int(max_10bit_offset),
            "10bit_capacity_percent": float(max_10bit_offset * 100.0 / float((1 << 10) - 1)),
        }

    def decode(
        self,
        shape: tuple[int, int],
        nodes: bytes,
        tile_roots: np.ndarray,
        tile_grid: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        node_count = len(nodes) // self.NODE_STRUCT.size
        unpacked = [self.NODE_STRUCT.unpack_from(nodes, i * self.NODE_STRUCT.size) for i in range(node_count)]
        output = np.ones((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                tile_x = min(x // self.tile_size, tile_grid[0] - 1)
                tile_y = min(y // self.tile_size, tile_grid[1] - 1)
                node_index = int(tile_roots[tile_y * tile_grid[0] + tile_x])
                if node_index == SST_INVALID_ROOT:
                    continue

                tile_origin_x = tile_x * self.tile_size
                tile_origin_y = tile_y * self.tile_size
                tile_extent_x = max(min(self.tile_size, width - tile_origin_x), 1)
                tile_extent_y = max(min(self.tile_size, height - tile_origin_y), 1)
                local_x = ((x + 0.5) - tile_origin_x) / tile_extent_x
                local_y = ((y + 0.5) - tile_origin_y) / tile_extent_y

                for _level in range(self.max_traversal_steps):
                    if node_index >= node_count:
                        break
                    flags, child_base, _, _, a, b, c, _ = unpacked[node_index]
                    if flags & SST_NODE_FLAG_PLANE:
                        output[y, x] = np.float32(np.clip(a * (local_x - 0.5) + b * (local_y - 0.5) + c, 0.0, 1.0))
                        break

                    quadrant = (1 if local_x >= 0.5 else 0) + (2 if local_y >= 0.5 else 0)
                    node_index = child_base + quadrant
                    local_x = local_x * 2.0 - (1.0 if quadrant & 1 else 0.0)
                    local_y = local_y * 2.0 - (1.0 if quadrant & 2 else 0.0)

        return output

    def decode_fixed64(
        self,
        shape: tuple[int, int],
        fixed64_nodes: np.ndarray,
        tile_roots: np.ndarray,
        tile_grid: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        nodes = np.asarray(fixed64_nodes, dtype=np.uint32).reshape(-1, 2)
        output = np.ones((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                tile_x = min(x // self.tile_size, tile_grid[0] - 1)
                tile_y = min(y // self.tile_size, tile_grid[1] - 1)
                node_index = int(tile_roots[tile_y * tile_grid[0] + tile_x])
                if node_index == SST_INVALID_ROOT:
                    continue

                tile_origin_x = tile_x * self.tile_size
                tile_origin_y = tile_y * self.tile_size
                tile_extent_x = max(min(self.tile_size, width - tile_origin_x), 1)
                tile_extent_y = max(min(self.tile_size, height - tile_origin_y), 1)
                local_x = ((x + 0.5) - tile_origin_x) / tile_extent_x
                local_y = ((y + 0.5) - tile_origin_y) / tile_extent_y

                for level in range(self.max_traversal_steps):
                    if node_index >= nodes.shape[0]:
                        break
                    word0 = int(nodes[node_index, 0])
                    word1 = int(nodes[node_index, 1])
                    control = word0 & 0x3
                    if control == 1:
                        output[y, x] = np.float32(self._decode_unorm((word0 >> 2) & 0x3FFFFFFF, 0x3FFFFFFF))
                        break
                    if control == 2:
                        qz = (word0 >> 2) & 0x3FFFFFFF
                        qx = word1 & 0xFFFF
                        qy = (word1 >> 16) & 0xFFFF
                        a = self._decode_snorm(qx, 0xFFFF, 2.0)
                        b = self._decode_snorm(qy, 0xFFFF, 2.0)
                        c = self._decode_unorm(qz, 0x3FFFFFFF)
                        output[y, x] = np.float32(np.clip(a * (local_x - 0.5) + b * (local_y - 0.5) + c, 0.0, 1.0))
                        break

                    bits = self._branch_offset_bits_for_level(level)
                    mask = (1 << bits) - 1
                    payload = (word0 >> 2) | (word1 << 30)
                    offsets = [
                        (payload >> (bits * child)) & mask
                        for child in range(4)
                    ]
                    quadrant = (1 if local_x >= 0.5 else 0) + (2 if local_y >= 0.5 else 0)
                    relative_offset = offsets[quadrant]
                    if relative_offset == 0:
                        break
                    node_index += relative_offset
                    local_x = local_x * 2.0 - (1.0 if quadrant & 1 else 0.0)
                    local_y = local_y * 2.0 - (1.0 if quadrant & 2 else 0.0)

        return output

    def decode_compact(
        self,
        shape: tuple[int, int],
        compact_words: np.ndarray,
        compact_roots: np.ndarray,
        tile_grid: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        words = np.asarray(compact_words, dtype=np.uint32).reshape(-1)
        output = np.ones((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                tile_x = min(x // self.tile_size, tile_grid[0] - 1)
                tile_y = min(y // self.tile_size, tile_grid[1] - 1)
                node_word_offset = int(compact_roots[tile_y * tile_grid[0] + tile_x])
                if node_word_offset == SST_INVALID_ROOT:
                    continue

                tile_origin_x = tile_x * self.tile_size
                tile_origin_y = tile_y * self.tile_size
                tile_extent_x = max(min(self.tile_size, width - tile_origin_x), 1)
                tile_extent_y = max(min(self.tile_size, height - tile_origin_y), 1)
                local_x = ((x + 0.5) - tile_origin_x) / tile_extent_x
                local_y = ((y + 0.5) - tile_origin_y) / tile_extent_y

                for level in range(self.max_traversal_steps):
                    if node_word_offset >= words.shape[0]:
                        break
                    word0 = int(words[node_word_offset])
                    control = word0 & 0x3
                    if control == 1:
                        output[y, x] = np.float32(self._decode_unorm((word0 >> 2) & 0x3FFFFFFF, 0x3FFFFFFF))
                        break
                    if control == 2:
                        if node_word_offset + 1 >= words.shape[0]:
                            break
                        word1 = int(words[node_word_offset + 1])
                        qz = (word0 >> 2) & 0x3FFFFFFF
                        qx = word1 & 0xFFFF
                        qy = (word1 >> 16) & 0xFFFF
                        a = self._decode_snorm(qx, 0xFFFF, 2.0)
                        b = self._decode_snorm(qy, 0xFFFF, 2.0)
                        c = self._decode_unorm(qz, 0x3FFFFFFF)
                        output[y, x] = np.float32(np.clip(a * (local_x - 0.5) + b * (local_y - 0.5) + c, 0.0, 1.0))
                        break

                    if node_word_offset + 1 >= words.shape[0]:
                        break
                    word1 = int(words[node_word_offset + 1])
                    bits = self._branch_offset_bits_for_level(level)
                    mask = (1 << bits) - 1
                    payload = (word0 >> 2) | (word1 << 30)
                    quadrant = (1 if local_x >= 0.5 else 0) + (2 if local_y >= 0.5 else 0)
                    relative_offset = (payload >> (bits * quadrant)) & mask
                    if relative_offset == 0:
                        break
                    node_word_offset += relative_offset
                    local_x = local_x * 2.0 - (1.0 if quadrant & 1 else 0.0)
                    local_y = local_y * 2.0 - (1.0 if quadrant & 2 else 0.0)

        return output

    def decode_compact_morton(
        self,
        shape: tuple[int, int],
        compact_words: np.ndarray,
        compact_roots: np.ndarray,
        tile_grid: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        words = np.asarray(compact_words, dtype=np.uint32).reshape(-1)
        output = np.ones((height, width), dtype=np.float32)
        tree_bits = int(math.log2(self.tile_size)) if self.tile_size > 0 and (self.tile_size & (self.tile_size - 1)) == 0 else 0

        for y in range(height):
            for x in range(width):
                tile_x = min(x // self.tile_size, tile_grid[0] - 1)
                tile_y = min(y // self.tile_size, tile_grid[1] - 1)
                node_word_offset = int(compact_roots[tile_y * tile_grid[0] + tile_x])
                if node_word_offset == SST_INVALID_ROOT:
                    continue

                tile_origin_x = tile_x * self.tile_size
                tile_origin_y = tile_y * self.tile_size
                tile_extent_x = max(min(self.tile_size, width - tile_origin_x), 1)
                tile_extent_y = max(min(self.tile_size, height - tile_origin_y), 1)
                local_x = ((x + 0.5) - tile_origin_x) / tile_extent_x
                local_y = ((y + 0.5) - tile_origin_y) / tile_extent_y

                for level in range(self.max_traversal_steps):
                    if node_word_offset >= words.shape[0]:
                        break
                    word0 = int(words[node_word_offset])
                    control = word0 & 0x3
                    if control == 1:
                        output[y, x] = np.float32(self._decode_unorm((word0 >> 2) & 0x3FFFFFFF, 0x3FFFFFFF))
                        break
                    if control == 2:
                        if node_word_offset + 1 >= words.shape[0]:
                            break
                        word1 = int(words[node_word_offset + 1])
                        qz = (word0 >> 2) & 0x3FFFFFFF
                        qx = word1 & 0xFFFF
                        qy = (word1 >> 16) & 0xFFFF
                        a = self._decode_snorm(qx, 0xFFFF, 2.0)
                        b = self._decode_snorm(qy, 0xFFFF, 2.0)
                        c = self._decode_unorm(qz, 0x3FFFFFFF)
                        output[y, x] = np.float32(np.clip(a * (local_x - 0.5) + b * (local_y - 0.5) + c, 0.0, 1.0))
                        break

                    if node_word_offset + 1 >= words.shape[0]:
                        break
                    word1 = int(words[node_word_offset + 1])
                    bits = self._branch_offset_bits_for_level(level)
                    mask = (1 << bits) - 1
                    payload = (word0 >> 2) | (word1 << 30)
                    quadrant = self._decode_tile_morton_quadrant(
                        x,
                        y,
                        tile_origin_x,
                        tile_origin_y,
                        tile_extent_x,
                        tile_extent_y,
                        local_x,
                        local_y,
                        tree_bits,
                        level,
                    )
                    relative_offset = (payload >> (bits * quadrant)) & mask
                    if relative_offset == 0:
                        break
                    node_word_offset += relative_offset
                    local_x = local_x * 2.0 - (1.0 if quadrant & 1 else 0.0)
                    local_y = local_y * 2.0 - (1.0 if quadrant & 2 else 0.0)

        return output

    def _decode_tile_morton_quadrant(
        self,
        pixel_x: int,
        pixel_y: int,
        tile_origin_x: int,
        tile_origin_y: int,
        tile_extent_x: int,
        tile_extent_y: int,
        local_x: float,
        local_y: float,
        tree_bits: int,
        level: int,
    ) -> int:
        fallback_quadrant = (1 if local_x >= 0.5 else 0) + (2 if local_y >= 0.5 else 0)
        if tree_bits == 0 or tile_extent_x != self.tile_size or tile_extent_y != self.tile_size or level >= tree_bits:
            return fallback_quadrant

        bit_index = tree_bits - 1 - level
        local_pixel_x = min(pixel_x - tile_origin_x, self.tile_size - 1)
        local_pixel_y = min(pixel_y - tile_origin_y, self.tile_size - 1)
        return ((local_pixel_x >> bit_index) & 1) | (((local_pixel_y >> bit_index) & 1) << 1)

    def _encode_region(self, depth: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
        node_index = self._append_node(level=0)
        self._encode_region_at(node_index, depth, x0, y0, x1, y1, level=0)
        return node_index

    def _append_node(self, level: int) -> int:
        node_index = len(self._nodes)
        self._nodes.append((0, 0, 0, 0, 0.0, 0.0, 1.0, 0.0))
        self._node_levels.append(level)
        return node_index

    def _reorder_nodes_level_order(self, tile_roots: np.ndarray) -> np.ndarray:
        if not self._nodes:
            return tile_roots

        queued: set[int] = set()
        order: list[int] = []
        for root in tile_roots:
            root_index = int(root)
            if root_index == SST_INVALID_ROOT or root_index in queued:
                continue

            queue = [root_index]
            queued.add(root_index)
            cursor = 0
            while cursor < len(queue):
                old_index = queue[cursor]
                cursor += 1
                order.append(old_index)

                flags, child_base, *_ = self._nodes[old_index]
                if flags & SST_NODE_FLAG_PLANE:
                    continue
                for child in range(4):
                    child_index = child_base + child
                    if child_index < len(self._nodes) and child_index not in queued:
                        queued.add(child_index)
                        queue.append(child_index)

        if len(order) != len(self._nodes):
            for old_index in range(len(self._nodes)):
                if old_index not in queued:
                    order.append(old_index)

        old_to_new = {old_index: new_index for new_index, old_index in enumerate(order)}
        reordered_nodes: list[tuple[int, int, int, int, float, float, float, float]] = []
        reordered_levels: list[int] = []
        for old_index in order:
            flags, child_base, meta, reserved, a, b, c, max_error = self._nodes[old_index]
            new_child_base = 0
            if (flags & SST_NODE_FLAG_PLANE) == 0:
                new_child_base = old_to_new[child_base]
            reordered_nodes.append((flags, new_child_base, meta, reserved, a, b, c, max_error))
            reordered_levels.append(self._node_levels[old_index])

        reordered_roots = np.full(tile_roots.shape, SST_INVALID_ROOT, dtype=np.uint32)
        for index, root in enumerate(tile_roots):
            root_index = int(root)
            if root_index != SST_INVALID_ROOT:
                reordered_roots[index] = np.uint32(old_to_new[root_index])

        self._nodes = reordered_nodes
        self._node_levels = reordered_levels
        return reordered_roots

    def _encode_region_at(self, node_index: int, depth: np.ndarray, x0: int, y0: int, x1: int, y1: int, level: int):
        region = depth[y0:y1, x0:x1]
        width = x1 - x0
        height = y1 - y0

        flags, a, b, c, max_error = self._fit_plane(depth, x0, y0, x1, y1)
        reached_min_leaf = width <= self.min_leaf_size or height <= self.min_leaf_size
        can_split_further = width > 1 and height > 1
        forced_leaf_blocked = (
            reached_min_leaf and
            can_split_further and
            self.forced_leaf_error_cap is not None and
            max_error > self.forced_leaf_error_cap
        )
        if max_error <= self.plane_error_threshold or (reached_min_leaf and not forced_leaf_blocked):
            if max_error > self.plane_error_threshold:
                pixel_count = max(width * height, 0)
                self._forced_leaf_node_count += 1
                self._forced_leaf_pixel_count += pixel_count
                self._forced_leaf_error_sum += float(max_error) * float(pixel_count)
                self._forced_leaf_max_error = max(self._forced_leaf_max_error, float(max_error))
            meta = (x0 & 0xFFFF) | ((y0 & 0xFFFF) << 16)
            self._nodes[node_index] = (flags, 0, meta, 0, a, b, c, max_error)
            return

        if float(region.max() - region.min()) <= self.constant_epsilon:
            value = float(region.min())
            meta = (x0 & 0xFFFF) | ((y0 & 0xFFFF) << 16)
            self._nodes[node_index] = (
                SST_NODE_FLAG_PLANE | SST_NODE_FLAG_CONSTANT,
                0,
                meta,
                0,
                0.0,
                0.0,
                value,
                float(region.max() - region.min()),
            )
            return

        mid_x = x0 + max(width // 2, 1)
        mid_y = y0 + max(height // 2, 1)
        child_base = len(self._nodes)
        for _ in range(4):
            self._append_node(level=level + 1)
        previous_visibility_tolerance = self.dual_visibility_tolerance
        if forced_leaf_blocked and self.forced_split_bias_fit:
            self.dual_visibility_tolerance = 0.0
        try:
            self._encode_region_at(child_base + 0, depth, x0, y0, mid_x, mid_y, level + 1)
            self._encode_region_at(child_base + 1, depth, mid_x, y0, x1, mid_y, level + 1)
            self._encode_region_at(child_base + 2, depth, x0, mid_y, mid_x, y1, level + 1)
            self._encode_region_at(child_base + 3, depth, mid_x, mid_y, x1, y1, level + 1)
        finally:
            self.dual_visibility_tolerance = previous_visibility_tolerance
        meta = (x0 & 0xFFFF) | ((y0 & 0xFFFF) << 16)
        self._nodes[node_index] = (0, child_base, meta, 0, 0.0, 0.0, 0.0, max_error)

    def _pack_paper_like_nodes(self) -> bytes:
        packed = bytearray()
        for node_index, node in enumerate(self._nodes):
            flags, child_base, _, _, a, b, c, _ = node
            if flags & SST_NODE_FLAG_CONSTANT:
                qz = self._quantize_unorm(c, 30)
                packed.extend(((qz << 2) | 0b01).to_bytes(4, "little", signed=False))
            elif flags & SST_NODE_FLAG_PLANE:
                qz = self._quantize_unorm(c, 30)
                qx = self._quantize_snorm(a, 16, value_range=2.0)
                qy = self._quantize_snorm(b, 16, value_range=2.0)
                word0 = (qz << 2) | 0b10
                word1 = qx | (qy << 16)
                packed.extend(word0.to_bytes(4, "little", signed=False))
                packed.extend(word1.to_bytes(4, "little", signed=False))
            else:
                bits = self._branch_offset_bits_for_level(self._node_levels[node_index])
                max_value = (1 << bits) - 1
                payload = 0
                for child in range(4):
                    relative_offset = max(0, min((child_base + child) - node_index, max_value))
                    payload |= relative_offset << (bits * child)
                packed.extend(((payload << 2) | 0b00).to_bytes(8, "little", signed=False))
        return bytes(packed)

    def _pack_compact_nodes(self, tile_roots: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        node_word_offsets = self._compute_node_word_offsets()
        word_count = 0
        if node_word_offsets:
            last_flags = self._nodes[-1][0]
            word_count = node_word_offsets[-1] + (1 if (last_flags & SST_NODE_FLAG_CONSTANT) else 2)

        words = np.zeros(max(word_count, 1), dtype=np.uint32)
        compact_roots = np.full(tile_roots.shape, SST_INVALID_ROOT, dtype=np.uint32)
        for i, root in enumerate(tile_roots):
            root_index = int(root)
            if root_index != SST_INVALID_ROOT and root_index < len(node_word_offsets):
                compact_roots[i] = np.uint32(node_word_offsets[root_index])

        overflow_count = 0
        max_branch_offset = 0
        for node_index, node in enumerate(self._nodes):
            flags, child_base, _, _, a, b, c, _ = node
            word_offset = node_word_offsets[node_index]
            if flags & SST_NODE_FLAG_CONSTANT:
                qz = self._quantize_unorm(c, 30)
                words[word_offset] = np.uint32((qz << 2) | 0b01)
            elif flags & SST_NODE_FLAG_PLANE:
                qz = self._quantize_unorm(c, 30)
                qx = self._quantize_snorm(a, 16, value_range=2.0)
                qy = self._quantize_snorm(b, 16, value_range=2.0)
                words[word_offset] = np.uint32((qz << 2) | 0b10)
                words[word_offset + 1] = np.uint32(qx | (qy << 16))
            else:
                bits = self._branch_offset_bits_for_level(self._node_levels[node_index])
                max_value = (1 << bits) - 1
                payload = 0
                for child in range(4):
                    child_index = child_base + child
                    relative_offset = node_word_offsets[child_index] - word_offset
                    max_branch_offset = max(max_branch_offset, relative_offset)
                    if relative_offset > max_value:
                        overflow_count += 1
                    payload |= max(0, min(relative_offset, max_value)) << (bits * child)
                packed = payload << 2
                words[word_offset] = np.uint32(packed & 0xFFFFFFFF)
                words[word_offset + 1] = np.uint32((packed >> 32) & 0xFFFFFFFF)

        return words, compact_roots, overflow_count, max_branch_offset

    def _pack_fixed64_nodes(self) -> np.ndarray:
        packed = np.zeros((len(self._nodes), 2), dtype=np.uint32)
        for node_index, node in enumerate(self._nodes):
            flags, child_base, _, _, a, b, c, _ = node
            if flags & SST_NODE_FLAG_CONSTANT:
                qz = self._quantize_unorm(c, 30)
                packed[node_index, 0] = np.uint32((qz << 2) | 0b01)
                packed[node_index, 1] = np.uint32(0)
            elif flags & SST_NODE_FLAG_PLANE:
                qz = self._quantize_unorm(c, 30)
                qx = self._quantize_snorm(a, 16, value_range=2.0)
                qy = self._quantize_snorm(b, 16, value_range=2.0)
                word0 = (qz << 2) | 0b10
                word1 = qx | (qy << 16)
                packed[node_index, 0] = np.uint32(word0)
                packed[node_index, 1] = np.uint32(word1)
            else:
                bits = self._branch_offset_bits_for_level(self._node_levels[node_index])
                max_value = (1 << bits) - 1
                payload = 0
                for child in range(4):
                    relative_offset = max(0, min((child_base + child) - node_index, max_value))
                    payload |= relative_offset << (bits * child)
                packed_bits = payload << 2
                packed[node_index, 0] = np.uint32(packed_bits & 0xFFFFFFFF)
                packed[node_index, 1] = np.uint32((packed_bits >> 32) & 0xFFFFFFFF)
        return packed

    def _count_fixed64_branch_offset_overflow(self) -> tuple[int, int]:
        overflow_count = 0
        max_branch_offset = 0
        for node_index, node in enumerate(self._nodes):
            flags, child_base, *_ = node
            if flags & SST_NODE_FLAG_PLANE:
                continue
            bits = self._branch_offset_bits_for_level(self._node_levels[node_index])
            max_value = (1 << bits) - 1
            for child in range(4):
                relative_offset = (child_base + child) - node_index
                max_branch_offset = max(max_branch_offset, relative_offset)
                if relative_offset > max_value:
                    overflow_count += 1
        return overflow_count, max_branch_offset

    @staticmethod
    def _quantize_unorm(value: float, bits: int) -> int:
        max_value = (1 << bits) - 1
        return int(round(max(0.0, min(1.0, float(value))) * max_value))

    @staticmethod
    def _quantize_snorm(value: float, bits: int, value_range: float) -> int:
        max_value = (1 << bits) - 1
        normalized = max(-1.0, min(1.0, float(value) / value_range)) * 0.5 + 0.5
        return int(round(normalized * max_value))

    @staticmethod
    def _decode_unorm(value: int, max_value: int) -> float:
        return float(value) / float(max_value)

    @staticmethod
    def _decode_snorm(value: int, max_value: int, value_range: float) -> float:
        return ((float(value) / float(max_value)) * 2.0 - 1.0) * value_range

    def _fit_plane(self, depth: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        region = depth[y0:y1, x0:x1]
        upper_region = self._upper_depth[y0:y1, x0:x1] if self._upper_depth is not None else region
        constant_value, constant_error = self._fit_constant_depth(region, upper_region)
        if constant_error <= self.plane_error_threshold:
            return (
                SST_NODE_FLAG_PLANE | SST_NODE_FLAG_CONSTANT,
                0.0,
                0.0,
                constant_value,
                constant_error,
            )

        ys, xs = np.mgrid[y0:y1, x0:x1]
        local_x = ((xs.astype(np.float32) + 0.5 - float(x0)) / float(max(x1 - x0, 1))).reshape(-1) - 0.5
        local_y = ((ys.astype(np.float32) + 0.5 - float(y0)) / float(max(y1 - y0, 1))).reshape(-1) - 0.5
        z = region.reshape(-1)
        upper_z = upper_region.reshape(-1)
        design = np.stack([local_x, local_y, np.ones_like(local_x)], axis=1)
        coeff, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
        predicted = design @ coeff

        if self.use_dual_layer and self._upper_depth is not None and self.dual_conservative:
            conservative_shift = min(float(np.min(z - predicted)), 0.0)
            coeff[2] += conservative_shift
            predicted += conservative_shift
            max_error = float(np.max(np.maximum(z - predicted, predicted - z)))
        elif self.use_dual_layer and self._upper_depth is not None and self.dual_max_leak is not None:
            shift, max_error = self._compute_dual_biased_shift_and_error(predicted, z, upper_z)
            coeff[2] += shift
        elif self.use_dual_layer and self._upper_depth is not None:
            min_shift = float(np.max(z - predicted))
            max_shift = float(np.min(upper_z - predicted))
            if min_shift <= max_shift:
                shift = min(max(0.0, min_shift), max_shift)
                max_error = 0.0
            else:
                shift = 0.5 * (min_shift + max_shift)
                shifted = predicted + shift
                max_error = float(np.max(np.maximum(z - shifted, shifted - upper_z)))
            coeff[2] += shift
        else:
            # Bias the plane toward the light so encoded depths do not sit behind
            # the original first-hit depth, which would create light leaks.
            conservative_shift = min(float(np.min(z - predicted)), 0.0)
            coeff[2] += conservative_shift
            predicted += conservative_shift
            max_error = float(np.max(z - predicted))
        coeff, max_error = self._quantize_adjust_plane(coeff, design, z, upper_z)
        return (
            SST_NODE_FLAG_PLANE,
            float(coeff[0]),
            float(coeff[1]),
            float(coeff[2]),
            max_error,
        )

    def _fit_constant_depth(self, lower_region: np.ndarray, upper_region: np.ndarray) -> tuple[float, float]:
        lower_z = lower_region.reshape(-1)
        upper_z = upper_region.reshape(-1)
        max_unorm = (1 << 30) - 1

        if self.use_dual_layer and self._upper_depth is not None and self.dual_conservative:
            q_value = int(math.floor(max(0.0, min(1.0, float(np.min(lower_z)))) * max_unorm))
            value = self._decode_unorm(q_value, max_unorm)
            conservative_error = np.maximum(lower_z - value, 0.0)
            leak_error = np.maximum(value - lower_z, 0.0)
            return value, float(np.max(np.maximum(conservative_error, leak_error)))

        if self.use_dual_layer and self._upper_depth is not None:
            if self.dual_max_leak is not None:
                upper_limit = np.minimum(
                    upper_z,
                    lower_z + max(0.0, float(self.dual_max_leak or 0.0) - self.dual_max_leak_guard),
                )
                lower_limit = lower_z - max(0.0, self.dual_visibility_tolerance - self.dual_max_leak_guard)
            else:
                lower_limit = lower_z
                upper_limit = upper_z

            interval_min = float(np.max(lower_limit))
            interval_max = float(np.min(upper_limit))
            if interval_min <= interval_max:
                preferred_value = interval_min
                if self.dual_visibility_tolerance > 0.0:
                    preferred_value = min(max(float(np.max(lower_z)), interval_min), interval_max)
                q_min = int(math.ceil(max(0.0, min(1.0, interval_min)) * max_unorm))
                q_max = int(math.floor(max(0.0, min(1.0, interval_max)) * max_unorm))
                preferred_q = self._quantize_unorm(preferred_value, 30)
                q_value = min(max(preferred_q, q_min), q_max) if q_min <= q_max else self._quantize_unorm(preferred_value, 30)
            elif self.dual_max_leak is not None:
                lower_shift = interval_min
                upper_shift = interval_max
                q_value = self._quantize_unorm(min(max(0.0, lower_shift), upper_shift), 30)
            else:
                q_value = self._quantize_unorm(0.5 * (interval_min + interval_max), 30)
            value = self._decode_unorm(q_value, max_unorm)

            if self.dual_max_leak is not None:
                practical_error = np.maximum(np.maximum(lower_limit - value, 0.0), np.maximum(value - upper_limit, 0.0))
                if float(np.max(practical_error)) > 1e-7:
                    return value, self.plane_error_threshold + float(np.max(practical_error))
                if self.dual_visibility_tolerance <= 0.0:
                    return value, float(np.max(np.maximum(lower_z - value, 0.0)))
                return value, 0.0

            interval_violation = np.maximum(np.maximum(lower_z - value, 0.0), np.maximum(value - upper_z, 0.0))
            return value, float(np.max(interval_violation))

        q_value = int(math.floor(max(0.0, min(1.0, float(np.min(lower_z)))) * max_unorm))
        value = self._decode_unorm(q_value, max_unorm)
        conservative_error = np.maximum(lower_z - value, 0.0)
        leak_error = np.maximum(value - lower_z, 0.0)
        return value, float(np.max(np.maximum(conservative_error, leak_error)))

    def _quantize_adjust_plane(
        self,
        coeff: np.ndarray,
        design: np.ndarray,
        lower_z: np.ndarray,
        upper_z: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        qa = self._quantize_snorm(float(coeff[0]), 16, value_range=2.0)
        qb = self._quantize_snorm(float(coeff[1]), 16, value_range=2.0)
        max_snorm = (1 << 16) - 1
        radius = self.plane_quantization_search_radius

        best_coeff, best_error = self._evaluate_quantized_plane_slopes(
            qa,
            qb,
            float(coeff[2]),
            design,
            lower_z,
            upper_z,
        )
        for delta_a in range(-radius, radius + 1):
            candidate_qa = min(max(qa + delta_a, 0), max_snorm)
            for delta_b in range(-radius, radius + 1):
                if delta_a == 0 and delta_b == 0:
                    continue
                candidate_qb = min(max(qb + delta_b, 0), max_snorm)
                candidate_coeff, candidate_error = self._evaluate_quantized_plane_slopes(
                    candidate_qa,
                    candidate_qb,
                    float(coeff[2]),
                    design,
                    lower_z,
                    upper_z,
                )
                if candidate_error < best_error:
                    best_coeff = candidate_coeff
                    best_error = candidate_error

        return best_coeff, best_error

    def _evaluate_quantized_plane_slopes(
        self,
        qa: int,
        qb: int,
        c: float,
        design: np.ndarray,
        lower_z: np.ndarray,
        upper_z: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        max_snorm = (1 << 16) - 1
        quantized = np.array(
            [
                self._decode_snorm(qa, max_snorm, 2.0),
                self._decode_snorm(qb, max_snorm, 2.0),
                self._decode_unorm(self._quantize_unorm(c, 30), 0x3FFFFFFF),
            ],
            dtype=np.float32,
        )
        predicted = design @ quantized

        if self.use_dual_layer and self._upper_depth is not None and self.dual_conservative:
            shift = min(float(np.min(lower_z - predicted)), 0.0)
        elif self.use_dual_layer and self._upper_depth is not None and self.dual_max_leak is not None:
            shift, _ = self._compute_dual_biased_shift_and_error(predicted, lower_z, upper_z)
        elif self.use_dual_layer and self._upper_depth is not None:
            min_shift = float(np.max(lower_z - predicted))
            max_shift = float(np.min(upper_z - predicted))
            if min_shift <= max_shift:
                shift = min(max(0.0, min_shift), max_shift)
            else:
                shift = 0.5 * (min_shift + max_shift)
        else:
            shift = min(float(np.min(lower_z - predicted)), 0.0)

        quantized[2] += shift
        quantized[2] = self._decode_unorm(self._quantize_unorm(float(quantized[2]), 30), 0x3FFFFFFF)
        predicted = design @ quantized

        if self.use_dual_layer and self._upper_depth is not None and self.dual_conservative:
            conservative_error = np.maximum(lower_z - predicted, 0.0)
            leak_error = np.maximum(predicted - lower_z, 0.0)
            max_error = float(np.max(np.maximum(conservative_error, leak_error)))
        elif self.use_dual_layer and self._upper_depth is not None and self.dual_max_leak is not None:
            _, max_error = self._compute_dual_biased_shift_and_error(predicted, lower_z, upper_z, allow_shift=False)
        elif self.use_dual_layer and self._upper_depth is not None:
            interval_violation = np.maximum(np.maximum(lower_z - predicted, 0.0), np.maximum(predicted - upper_z, 0.0))
            max_error = float(np.max(interval_violation))
        else:
            conservative_error = np.maximum(lower_z - predicted, 0.0)
            leak_error = np.maximum(predicted - lower_z, 0.0)
            max_error = float(np.max(np.maximum(conservative_error, leak_error)))

        return quantized, max_error

    def _compute_dual_biased_shift_and_error(
        self,
        predicted: np.ndarray,
        lower_z: np.ndarray,
        upper_z: np.ndarray,
        allow_shift: bool = True,
    ) -> tuple[float, float]:
        upper_limit = np.minimum(
            upper_z,
            lower_z + max(0.0, float(self.dual_max_leak or 0.0) - self.dual_max_leak_guard),
        )
        lower_limit = lower_z - max(0.0, self.dual_visibility_tolerance - self.dual_max_leak_guard)
        if allow_shift:
            lower_shift = float(np.max(lower_limit - predicted))
            upper_shift = float(np.min(upper_limit - predicted))
            if lower_shift <= upper_shift:
                if self.dual_visibility_tolerance > 0.0:
                    preferred_shift = float(np.max(lower_z - predicted))
                    shift = min(max(preferred_shift, lower_shift), upper_shift)
                else:
                    shift = min(max(0.0, lower_shift), upper_shift)
            else:
                shift = upper_shift
        else:
            shift = 0.0

        shifted = predicted + shift
        practical_conservative_error = np.maximum(lower_limit - shifted, 0.0)
        practical_leak_error = np.maximum(shifted - upper_limit, 0.0)
        practical_error = max(float(np.max(practical_leak_error)), float(np.max(practical_conservative_error)))
        if practical_error > 1e-7:
            max_error = self.plane_error_threshold + practical_error
        elif self.dual_visibility_tolerance <= 0.0:
            max_error = float(np.max(lower_z - shifted))
        else:
            max_error = 0.0
        return shift, max_error

    def _quantize_decode_plane(self, coeff: np.ndarray) -> np.ndarray:
        qa = self._quantize_snorm(float(coeff[0]), 16, value_range=2.0)
        qb = self._quantize_snorm(float(coeff[1]), 16, value_range=2.0)
        qc = self._quantize_unorm(float(coeff[2]), 30)
        return np.array(
            [
                self._decode_snorm(qa, 0xFFFF, 2.0),
                self._decode_snorm(qb, 0xFFFF, 2.0),
                self._decode_unorm(qc, 0x3FFFFFFF),
            ],
            dtype=np.float32,
        )
