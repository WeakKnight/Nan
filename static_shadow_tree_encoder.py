from __future__ import annotations

import ctypes
from dataclasses import fields
from pathlib import Path
import platform
import subprocess
from typing import Literal

import numpy as np

from sparse_shadow_tree import SparseShadowTreeData, SparseShadowTreeEncoder, SparseShadowTreeStats


PROJECT_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_DIR / "sst_encoder_build"


class CppSSTEncoderUnavailable(RuntimeError):
    pass


class _COptions(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("tile_size", ctypes.c_uint32),
        ("min_leaf_size", ctypes.c_uint32),
        ("plane_error_threshold", ctypes.c_float),
        ("constant_epsilon", ctypes.c_float),
        ("use_dual_layer", ctypes.c_uint32),
        ("has_second_depth", ctypes.c_uint32),
        ("has_dual_depth_slack", ctypes.c_uint32),
        ("dual_depth_slack", ctypes.c_float),
        ("dual_conservative", ctypes.c_uint32),
        ("has_dual_max_leak", ctypes.c_uint32),
        ("dual_max_leak", ctypes.c_float),
        ("dual_visibility_tolerance", ctypes.c_float),
        ("shadow_bias", ctypes.c_float),
        ("plane_quantization_search_radius", ctypes.c_uint32),
        ("has_forced_leaf_error_cap", ctypes.c_uint32),
        ("forced_leaf_error_cap", ctypes.c_float),
        ("forced_split_bias_fit", ctypes.c_uint32),
        ("thread_count", ctypes.c_uint32),
    ]


class _CStats(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("tile_grid_x", ctypes.c_uint32),
        ("tile_grid_y", ctypes.c_uint32),
        ("tile_size", ctypes.c_uint32),
        ("min_leaf_size", ctypes.c_uint32),
        ("max_tree_depth", ctypes.c_uint32),
        ("max_traversal_steps", ctypes.c_uint32),
        ("branch_10bit_start_level", ctypes.c_uint32),
        ("tile_count", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
        ("branch_node_count", ctypes.c_uint32),
        ("branch_13bit_node_count", ctypes.c_uint32),
        ("branch_10bit_node_count", ctypes.c_uint32),
        ("plane_node_count", ctypes.c_uint32),
        ("uniform_plane_node_count", ctypes.c_uint32),
        ("compact_branch_words", ctypes.c_uint32),
        ("compact_30bit_plane_words", ctypes.c_uint32),
        ("compact_62bit_plane_words", ctypes.c_uint32),
        ("compact_node_words", ctypes.c_uint32),
        ("compact_tile_root_bytes", ctypes.c_uint32),
        ("compact_branch_offset_overflow_count", ctypes.c_uint32),
        ("fixed64_branch_offset_overflow_count", ctypes.c_uint32),
        ("max_compact_branch_offset", ctypes.c_uint32),
        ("max_fixed64_branch_offset", ctypes.c_uint32),
        ("compact_branch_13bit_max_offset", ctypes.c_uint32),
        ("compact_branch_13bit_capacity_percent", ctypes.c_float),
        ("compact_branch_10bit_max_offset", ctypes.c_uint32),
        ("compact_branch_10bit_capacity_percent", ctypes.c_float),
        ("forced_leaf_node_count", ctypes.c_uint32),
        ("forced_leaf_pixel_count", ctypes.c_uint64),
        ("forced_leaf_max_error", ctypes.c_float),
        ("forced_leaf_error_sum", ctypes.c_double),
        ("original_bytes", ctypes.c_uint64),
        ("encoded_bytes", ctypes.c_uint64),
        ("packed_encoded_bytes", ctypes.c_uint64),
        ("fixed64_encoded_bytes", ctypes.c_uint64),
        ("decompressed_depth_bytes", ctypes.c_uint64),
        ("packed_decompressed_working_set_bytes", ctypes.c_uint64),
        ("compression_ratio", ctypes.c_float),
        ("packed_compression_ratio", ctypes.c_float),
        ("fixed64_compression_ratio", ctypes.c_float),
        ("packed_decompressed_working_set_ratio", ctypes.c_float),
        ("packed_decode_valid", ctypes.c_uint32),
        ("max_error", ctypes.c_float),
        ("mean_error", ctypes.c_float),
        ("rmse_error", ctypes.c_float),
    ]


class _COutput(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(ctypes.c_uint8)),
        ("nodes_size", ctypes.c_size_t),
        ("fixed64_nodes", ctypes.POINTER(ctypes.c_uint32)),
        ("fixed64_word_count", ctypes.c_size_t),
        ("compact_words", ctypes.POINTER(ctypes.c_uint32)),
        ("compact_word_count", ctypes.c_size_t),
        ("compact_roots", ctypes.POINTER(ctypes.c_uint32)),
        ("compact_root_count", ctypes.c_size_t),
        ("tile_roots", ctypes.POINTER(ctypes.c_uint32)),
        ("tile_root_count", ctypes.c_size_t),
        ("stats", _CStats),
        ("error_message", ctypes.c_char_p),
    ]


_loaded_library: ctypes.CDLL | None = None
_fallback_warning_printed = False


def _run_cmake(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CppSSTEncoderUnavailable("cmake was not found on PATH") from exc

    if result.returncode != 0:
        details = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise CppSSTEncoderUnavailable(f"CMake command failed: {' '.join(command)}\n{details}")


def build_cpp_encoder() -> Path:
    configure = ["cmake", "-S", str(PROJECT_DIR), "-B", str(BUILD_DIR)]
    if platform.system() == "Windows" and not (BUILD_DIR / "CMakeCache.txt").exists():
        configure.extend(["-A", "x64"])
    _run_cmake(configure)
    _run_cmake(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--parallel"])

    names = (
        ["static_shadow_tree_encoder.dll"]
        if platform.system() == "Windows"
        else ["libstatic_shadow_tree_encoder.dylib", "libstatic_shadow_tree_encoder.so"]
    )
    for name in names:
        matches = sorted(BUILD_DIR.rglob(name), key=lambda path: len(path.parts))
        if matches:
            return matches[0]
    raise CppSSTEncoderUnavailable(f"CMake build succeeded but no encoder library was found in {BUILD_DIR}")


def load_cpp_encoder_library() -> ctypes.CDLL:
    global _loaded_library
    if _loaded_library is not None:
        return _loaded_library

    library_path = build_cpp_encoder()
    library = ctypes.CDLL(str(library_path))
    library.sst_encode.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(_COptions),
        ctypes.POINTER(_COutput),
    ]
    library.sst_encode.restype = ctypes.c_int
    library.sst_free_output.argtypes = [ctypes.POINTER(_COutput)]
    library.sst_free_output.restype = None
    library.sst_version.argtypes = []
    library.sst_version.restype = ctypes.c_char_p
    _loaded_library = library
    return library


def _array_from_pointer(pointer, count: int, dtype) -> np.ndarray:
    if not pointer or count <= 0:
        return np.zeros((0,), dtype=dtype)
    array = np.ctypeslib.as_array(pointer, shape=(int(count),))
    return np.asarray(array, dtype=dtype).copy()


def _default_stats_values() -> dict:
    values: dict = {}
    for field in fields(SparseShadowTreeStats):
        name = field.name
        if name == "tile_diagnostics" or name.endswith("percentiles"):
            values[name] = {}
        elif (
            name.endswith("details")
            or name.endswith("cdf")
            or name.endswith("histogram")
            or name == "visibility_probe_offsets_in_bias"
        ):
            values[name] = tuple()
        elif name.endswith("valid") or name == "dual_layer":
            values[name] = False
        elif (
            "percent" in name
            or "ratio" in name
            or "error" in name
            or name in {"shadow_bias", "max_error", "mean_error", "rmse_error"}
        ):
            values[name] = 0.0
        else:
            values[name] = 0
    return values


def _stats_from_cpp(cstats: _CStats, use_dual_layer: bool, shadow_bias: float) -> SparseShadowTreeStats:
    values = _default_stats_values()
    values["tile_diagnostics"] = {"backend": "cpp", "full_error_stats": False}
    forced_leaf_mean = 0.0
    if cstats.forced_leaf_pixel_count:
        forced_leaf_mean = float(cstats.forced_leaf_error_sum / float(cstats.forced_leaf_pixel_count) * 100.0)
    pixel_count = max(int(cstats.width) * int(cstats.height), 1)
    forced_leaf_percent = float(cstats.forced_leaf_pixel_count * 100.0 / float(pixel_count))

    values.update(
        width=int(cstats.width),
        height=int(cstats.height),
        dual_layer=bool(use_dual_layer),
        max_tree_depth=int(cstats.max_tree_depth),
        max_traversal_steps=int(cstats.max_traversal_steps),
        branch_10bit_start_level=int(cstats.branch_10bit_start_level),
        tile_count=int(cstats.tile_count),
        node_count=int(cstats.node_count),
        branch_node_count=int(cstats.branch_node_count),
        branch_13bit_node_count=int(cstats.branch_13bit_node_count),
        branch_10bit_node_count=int(cstats.branch_10bit_node_count),
        fixed64_branch_offset_overflow_count=int(cstats.fixed64_branch_offset_overflow_count),
        compact_branch_offset_overflow_count=int(cstats.compact_branch_offset_overflow_count),
        max_fixed64_branch_offset=int(cstats.max_fixed64_branch_offset),
        max_compact_branch_offset=int(cstats.max_compact_branch_offset),
        compact_branch_13bit_max_offset=int(cstats.compact_branch_13bit_max_offset),
        compact_branch_13bit_capacity_percent=float(cstats.compact_branch_13bit_capacity_percent),
        compact_branch_10bit_max_offset=int(cstats.compact_branch_10bit_max_offset),
        compact_branch_10bit_capacity_percent=float(cstats.compact_branch_10bit_capacity_percent),
        plane_node_count=int(cstats.plane_node_count),
        uniform_plane_node_count=int(cstats.uniform_plane_node_count),
        compact_branch_words=int(cstats.compact_branch_words),
        compact_30bit_plane_words=int(cstats.compact_30bit_plane_words),
        compact_62bit_plane_words=int(cstats.compact_62bit_plane_words),
        compact_node_words=int(cstats.compact_node_words),
        compact_tile_root_bytes=int(cstats.compact_tile_root_bytes),
        forced_leaf_node_count=int(cstats.forced_leaf_node_count),
        forced_leaf_pixel_percent=forced_leaf_percent,
        forced_leaf_max_error_percent=float(cstats.forced_leaf_max_error * 100.0),
        forced_leaf_mean_error_percent=forced_leaf_mean,
        original_bytes=int(cstats.original_bytes),
        encoded_bytes=int(cstats.encoded_bytes),
        compression_ratio=float(cstats.compression_ratio),
        packed_encoded_bytes=int(cstats.packed_encoded_bytes),
        packed_compression_ratio=float(cstats.packed_compression_ratio),
        decompressed_depth_bytes=int(cstats.decompressed_depth_bytes),
        packed_decompressed_working_set_bytes=int(cstats.packed_decompressed_working_set_bytes),
        packed_decompressed_working_set_ratio=float(cstats.packed_decompressed_working_set_ratio),
        packed_decode_valid=bool(cstats.packed_decode_valid),
        packed_morton_decode_valid=bool(cstats.packed_decode_valid),
        compact_fixed64_decode_valid=bool(cstats.packed_decode_valid),
        fixed64_encoded_bytes=int(cstats.fixed64_encoded_bytes),
        fixed64_compression_ratio=float(cstats.fixed64_compression_ratio),
        shadow_bias=float(shadow_bias),
        max_error=float(cstats.max_error),
        mean_error=float(cstats.mean_error),
        rmse_error=float(cstats.rmse_error),
        max_error_percent=float(cstats.max_error * 100.0),
        mean_error_percent=float(cstats.mean_error * 100.0),
        rmse_error_percent=float(cstats.rmse_error * 100.0),
        packed_max_error_percent=float(cstats.max_error * 100.0),
        fixed64_max_error=float(cstats.max_error),
        fixed64_max_error_percent=float(cstats.max_error * 100.0),
    )
    return SparseShadowTreeStats(**values)


class CppSparseShadowTreeEncoder:
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
        thread_count: int = 0,
    ):
        self.tile_size = max(1, int(tile_size))
        self.min_leaf_size = max(1, int(min_leaf_size))
        self.plane_error_threshold = float(plane_error_threshold)
        self.constant_epsilon = float(constant_epsilon)
        self.use_dual_layer = bool(use_dual_layer)
        self.dual_depth_slack = None if dual_depth_slack is None else float(dual_depth_slack)
        self.dual_conservative = bool(dual_conservative)
        self.dual_max_leak = None if dual_max_leak is None else max(0.0, float(dual_max_leak))
        self.dual_visibility_tolerance = max(0.0, float(dual_visibility_tolerance))
        self.shadow_bias = max(0.0, float(shadow_bias))
        self.plane_quantization_search_radius = max(0, int(plane_quantization_search_radius))
        self.forced_leaf_error_cap = None if forced_leaf_error_cap is None else max(0.0, float(forced_leaf_error_cap))
        self.forced_split_bias_fit = bool(forced_split_bias_fit)
        self.thread_count = max(0, int(thread_count))
        self._max_tree_depth = self._compute_max_tree_depth()
        self._branch_10bit_start_level = self._compute_branch_10bit_start_level()

    @property
    def max_traversal_steps(self) -> int:
        return self._max_tree_depth + 1

    @property
    def branch_10bit_start_level(self) -> int:
        return self._branch_10bit_start_level

    def _compute_max_tree_depth(self) -> int:
        if self.tile_size <= 1 or self.min_leaf_size >= self.tile_size:
            return 0
        import math

        return int(math.ceil(math.log2(float(self.tile_size) / float(max(self.min_leaf_size, 1)))))

    def _compute_branch_10bit_start_level(self) -> int:
        for level in range(self._max_tree_depth + 1):
            child_depth = max(self._max_tree_depth - level - 1, 0)
            child_subtree_nodes = (4 ** (child_depth + 1) - 1) // 3
            max_child_offset = 1 + 3 * child_subtree_nodes
            if max_child_offset <= 0x3FF:
                return level
        return self._max_tree_depth + 1

    def encode(self, depth: np.ndarray, second_depth: np.ndarray | None = None) -> SparseShadowTreeData:
        depth_2d = np.ascontiguousarray(np.asarray(depth, dtype=np.float32).squeeze())
        if depth_2d.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth_2d.shape}")

        second_2d = None
        if self.use_dual_layer and second_depth is not None:
            second_2d = np.ascontiguousarray(np.asarray(second_depth, dtype=np.float32).squeeze())
            if second_2d.shape != depth_2d.shape:
                raise ValueError(f"Second depth shape {second_2d.shape} does not match {depth_2d.shape}")

        height, width = depth_2d.shape
        options = _COptions(
            width=int(width),
            height=int(height),
            tile_size=self.tile_size,
            min_leaf_size=self.min_leaf_size,
            plane_error_threshold=self.plane_error_threshold,
            constant_epsilon=self.constant_epsilon,
            use_dual_layer=1 if self.use_dual_layer else 0,
            has_second_depth=1 if second_2d is not None else 0,
            has_dual_depth_slack=1 if self.dual_depth_slack is not None else 0,
            dual_depth_slack=0.0 if self.dual_depth_slack is None else self.dual_depth_slack,
            dual_conservative=1 if self.dual_conservative else 0,
            has_dual_max_leak=1 if self.dual_max_leak is not None else 0,
            dual_max_leak=0.0 if self.dual_max_leak is None else self.dual_max_leak,
            dual_visibility_tolerance=self.dual_visibility_tolerance,
            shadow_bias=self.shadow_bias,
            plane_quantization_search_radius=self.plane_quantization_search_radius,
            has_forced_leaf_error_cap=1 if self.forced_leaf_error_cap is not None else 0,
            forced_leaf_error_cap=0.0 if self.forced_leaf_error_cap is None else self.forced_leaf_error_cap,
            forced_split_bias_fit=1 if self.forced_split_bias_fit else 0,
            thread_count=self.thread_count,
        )

        library = load_cpp_encoder_library()
        output = _COutput()
        depth_ptr = depth_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        second_ptr = None if second_2d is None else second_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        try:
            ok = library.sst_encode(depth_ptr, second_ptr, ctypes.byref(options), ctypes.byref(output))
            if not ok:
                message = output.error_message.decode("utf-8", errors="replace") if output.error_message else "unknown C++ SST encode failure"
                raise RuntimeError(message)

            nodes = ctypes.string_at(output.nodes, output.nodes_size) if output.nodes and output.nodes_size else b""
            compact_words = _array_from_pointer(output.compact_words, output.compact_word_count, np.uint32)
            compact_roots = _array_from_pointer(output.compact_roots, output.compact_root_count, np.uint32)
            tile_roots = _array_from_pointer(output.tile_roots, output.tile_root_count, np.uint32)
            fixed64_words = _array_from_pointer(output.fixed64_nodes, output.fixed64_word_count, np.uint32)
            fixed64_nodes = fixed64_words.reshape((-1, 2)).copy() if fixed64_words.size else np.zeros((0, 2), dtype=np.uint32)
            stats = _stats_from_cpp(output.stats, self.use_dual_layer and second_2d is not None, self.shadow_bias)
            self._branch_10bit_start_level = stats.branch_10bit_start_level
            return SparseShadowTreeData(
                nodes=nodes,
                packed_nodes=compact_words.tobytes(),
                compact_words=compact_words,
                compact_tile_roots=compact_roots,
                fixed64_nodes=fixed64_nodes,
                tile_roots=tile_roots,
                tile_grid=(int(output.stats.tile_grid_x), int(output.stats.tile_grid_y)),
                tile_size=self.tile_size,
                min_leaf_size=self.min_leaf_size,
                stats=stats,
            )
        finally:
            library.sst_free_output(ctypes.byref(output))


class AutoSparseShadowTreeEncoder:
    def __init__(self, *args, force_cpp: bool = False, **kwargs):
        self._force_cpp = force_cpp
        self._cpp = CppSparseShadowTreeEncoder(*args, **kwargs)
        self._python = SparseShadowTreeEncoder(*args, **kwargs)

    @property
    def tile_size(self) -> int:
        return self._cpp.tile_size

    @property
    def min_leaf_size(self) -> int:
        return self._cpp.min_leaf_size

    @property
    def max_traversal_steps(self) -> int:
        return self._cpp.max_traversal_steps

    @property
    def branch_10bit_start_level(self) -> int:
        return self._cpp.branch_10bit_start_level

    def encode(self, depth: np.ndarray, second_depth: np.ndarray | None = None) -> SparseShadowTreeData:
        global _fallback_warning_printed
        try:
            return self._cpp.encode(depth, second_depth)
        except Exception as exc:
            if self._force_cpp:
                raise
            if not _fallback_warning_printed:
                print(f"[SST] C++ encoder unavailable, falling back to Python: {exc}")
                _fallback_warning_printed = True
            return self._python.encode(depth, second_depth)


def create_sparse_shadow_tree_encoder(
    backend: Literal["auto", "cpp", "python"] = "auto",
    **kwargs,
):
    backend = backend.strip().lower()
    if backend == "python":
        return SparseShadowTreeEncoder(**kwargs)
    if backend == "cpp":
        return AutoSparseShadowTreeEncoder(force_cpp=True, **kwargs)
    if backend == "auto":
        return AutoSparseShadowTreeEncoder(force_cpp=False, **kwargs)
    raise ValueError(f"Unsupported SST encoder backend '{backend}'")
