from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "cpp"
BUILD_DIR = ROOT / "build" / "vertex_baking_utils"


class NativeBakeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TinyBvhTraceStats:
    layout: str
    thread_count: int
    build_milliseconds: float
    trace_milliseconds: float
    visible_ray_count: int
    sample_count: int
    ray_count: int

    @property
    def total_ray_count(self) -> int:
        return self.sample_count * self.ray_count

    @property
    def rays_per_second(self) -> float:
        seconds = self.trace_milliseconds * 1e-3
        return self.total_ray_count / seconds if seconds > 0.0 else 0.0

    @property
    def visible_fraction(self) -> float:
        total = self.total_ray_count
        return self.visible_ray_count / total if total > 0 else 0.0


def _library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "vertex_baking_utils.dll"
    if system == "Darwin":
        return "libvertex_baking_utils.dylib"
    return "libvertex_baking_utils.so"


def _library_names() -> list[str]:
    names = [_library_name()]
    if platform.system() == "Windows":
        names.append("libvertex_baking_utils.dll")
    return names


def _library_path() -> Path:
    return BUILD_DIR / _library_name()


def _find_library() -> Path | None:
    for name in _library_names():
        direct = BUILD_DIR / name
        if direct.exists():
            return direct
    for name in _library_names():
        candidates = list(BUILD_DIR.rglob(name))
        if candidates:
            return candidates[0]
    return None


def _native_sources_newer_than(lib_path: Path) -> bool:
    source_paths = [CPP_DIR / "CMakeLists.txt"]
    source_paths.extend(CPP_DIR.glob("*.cpp"))
    source_paths.extend(CPP_DIR.glob("*.h"))
    source_paths.extend((CPP_DIR / "third_party" / "tinybvh").glob("*"))
    lib_mtime = lib_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > lib_mtime for path in source_paths)


def build_native(force: bool = False) -> Path:
    lib_path = _find_library()
    if lib_path is not None and not force and not _native_sources_newer_than(lib_path):
        return lib_path

    cmake = shutil.which("cmake")
    if cmake is None:
        raise NativeBakeError("cmake was not found on PATH; cannot build vertex_baking_utils")

    generator_args = []
    if shutil.which("ninja") is not None:
        generator_args = ["-G", "Ninja"]

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    configure_cmd = [
        cmake,
        "-S",
        str(CPP_DIR),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        *generator_args,
    ]
    subprocess.run(configure_cmd, check=True)
    subprocess.run([cmake, "--build", str(BUILD_DIR), "--config", "Release"], check=True)

    lib_path = _find_library()
    if lib_path is None:
        raise NativeBakeError(f"native library was not produced under: {BUILD_DIR}")
    return lib_path


_LIB = None


def _load_library(auto_build: bool = True):
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = _find_library()
    if lib_path is None:
        if not auto_build:
            raise NativeBakeError(f"native library not found under: {BUILD_DIR}")
        lib_path = build_native()

    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.fspath(lib_path.parent))
        gxx = shutil.which("g++")
        if gxx is not None:
            os.add_dll_directory(os.fspath(Path(gxx).parent))

    lib = ctypes.CDLL(os.fspath(lib_path))
    lib.vbake_last_error.restype = ctypes.c_char_p
    lib.vbake_least_squares.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.vbake_least_squares.restype = ctypes.c_int
    lib.vbake_visibility_least_squares.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.vbake_visibility_least_squares.restype = ctypes.c_int
    lib.vbake_pmr_visibility_sh_least_squares.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.vbake_pmr_visibility_sh_least_squares.restype = ctypes.c_int
    lib.vbake_pmr_sh_to_cones.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.vbake_pmr_sh_to_cones.restype = ctypes.c_int
    lib.vbake_tinybvh_has_bvh8_cpu.argtypes = []
    lib.vbake_tinybvh_has_bvh8_cpu.restype = ctypes.c_int
    lib.vbake_pmr_visibility_sh_tinybvh.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.vbake_pmr_visibility_sh_tinybvh.restype = ctypes.c_int
    _LIB = lib
    return lib


def _as_float32_contiguous(array, shape_tail: tuple[int, ...], name: str) -> npt.NDArray[np.float32]:
    result = np.ascontiguousarray(array, dtype=np.float32)
    if result.ndim != 1 + len(shape_tail) or result.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(str(v) for v in shape_tail)})")
    return result


def _as_uint32_contiguous(array, shape_tail: tuple[int, ...], name: str) -> npt.NDArray[np.uint32]:
    result = np.ascontiguousarray(array, dtype=np.uint32)
    if result.ndim != 1 + len(shape_tail) or result.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(str(v) for v in shape_tail)})")
    return result


def bake_least_squares(
    positions,
    indices,
    sample_triangles,
    sample_barycentrics,
    sample_values,
    *,
    regularization_weight: float = 0.0,
    auto_build: bool = True,
) -> npt.NDArray[np.float32]:
    positions = _as_float32_contiguous(positions, (3,), "positions")
    indices = _as_uint32_contiguous(indices, (3,), "indices")
    sample_triangles = np.ascontiguousarray(sample_triangles, dtype=np.uint32)
    if sample_triangles.ndim != 1:
        raise ValueError("sample_triangles must have shape (N,)")
    sample_barycentrics = _as_float32_contiguous(sample_barycentrics, (3,), "sample_barycentrics")
    sample_values = np.ascontiguousarray(sample_values, dtype=np.float32)
    if sample_values.ndim == 1:
        sample_values = sample_values.reshape(-1, 1)
    if sample_values.ndim != 2:
        raise ValueError("sample_values must have shape (N,) or (N, C)")

    sample_count = int(sample_triangles.shape[0])
    if sample_barycentrics.shape[0] != sample_count or sample_values.shape[0] != sample_count:
        raise ValueError("sample_triangles, sample_barycentrics, and sample_values must have matching lengths")
    if sample_count == 0:
        raise ValueError("at least one sample is required")

    channels = int(sample_values.shape[1])
    out = np.zeros((positions.shape[0], channels), dtype=np.float32)
    lib = _load_library(auto_build=auto_build)
    status = lib.vbake_least_squares(
        ctypes.c_int(int(positions.shape[0])),
        ctypes.c_int(int(indices.shape[0])),
        positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int(sample_count),
        sample_triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sample_barycentrics.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sample_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(channels),
        ctypes.c_float(float(regularization_weight)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        message = lib.vbake_last_error()
        decoded = message.decode("utf-8", errors="replace") if message else "unknown native bake error"
        raise NativeBakeError(f"vbake_least_squares failed with status {status}: {decoded}")
    return out


def bake_visibility_least_squares(
    positions,
    normals,
    tangents,
    indices,
    sample_triangles,
    sample_barycentrics,
    sample_raw_cones,
    *,
    regularization_weight: float = 0.0,
    auto_build: bool = True,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    positions = _as_float32_contiguous(positions, (3,), "positions")
    normals = _as_float32_contiguous(normals, (3,), "normals")
    tangents = _as_float32_contiguous(tangents, (4,), "tangents")
    indices = _as_uint32_contiguous(indices, (3,), "indices")
    sample_triangles = np.ascontiguousarray(sample_triangles, dtype=np.uint32)
    if sample_triangles.ndim != 1:
        raise ValueError("sample_triangles must have shape (N,)")
    sample_barycentrics = _as_float32_contiguous(sample_barycentrics, (3,), "sample_barycentrics")
    sample_raw_cones = _as_float32_contiguous(sample_raw_cones, (5,), "sample_raw_cones")

    vertex_count = int(positions.shape[0])
    if normals.shape[0] != vertex_count or tangents.shape[0] != vertex_count:
        raise ValueError("positions, normals, and tangents must have matching lengths")
    sample_count = int(sample_triangles.shape[0])
    if sample_barycentrics.shape[0] != sample_count or sample_raw_cones.shape[0] != sample_count:
        raise ValueError("sample_triangles, sample_barycentrics, and sample_raw_cones must have matching lengths")
    if sample_count == 0:
        raise ValueError("at least one sample is required")

    out_vertex_cones = np.zeros((vertex_count, 5), dtype=np.float32)
    out_encoded = np.zeros((vertex_count, 4), dtype=np.float32)
    lib = _load_library(auto_build=auto_build)
    status = lib.vbake_visibility_least_squares(
        ctypes.c_int(vertex_count),
        ctypes.c_int(int(indices.shape[0])),
        positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tangents.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int(sample_count),
        sample_triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sample_barycentrics.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sample_raw_cones.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(float(regularization_weight)),
        out_vertex_cones.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_encoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        message = lib.vbake_last_error()
        decoded = message.decode("utf-8", errors="replace") if message else "unknown native visibility bake error"
        raise NativeBakeError(f"vbake_visibility_least_squares failed with status {status}: {decoded}")
    return out_vertex_cones, out_encoded


def bake_pmr_visibility_sh_least_squares(
    positions,
    indices,
    triangle_areas,
    samples_per_triangle: int,
    sample_barycentrics,
    sample_sh16,
    *,
    edge_regularization: float = 0.05,
    auto_build: bool = True,
) -> npt.NDArray[np.float32]:
    positions = _as_float32_contiguous(positions, (3,), "positions")
    indices = _as_uint32_contiguous(indices, (3,), "indices")
    triangle_areas = np.ascontiguousarray(triangle_areas, dtype=np.float32)
    if triangle_areas.ndim != 1 or triangle_areas.shape[0] != indices.shape[0]:
        raise ValueError("triangle_areas must have shape (triangle_count,)")
    samples_per_triangle = int(samples_per_triangle)
    if samples_per_triangle <= 0:
        raise ValueError("samples_per_triangle must be positive")
    sample_barycentrics = _as_float32_contiguous(sample_barycentrics, (3,), "sample_barycentrics")
    sample_sh16 = _as_float32_contiguous(sample_sh16, (16,), "sample_sh16")
    expected_sample_count = int(indices.shape[0]) * samples_per_triangle
    if sample_barycentrics.shape[0] != expected_sample_count or sample_sh16.shape[0] != expected_sample_count:
        raise ValueError(
            "sample_barycentrics and sample_sh16 must contain samples_per_triangle entries for every triangle"
        )

    out = np.zeros((positions.shape[0], 16), dtype=np.float32)
    lib = _load_library(auto_build=auto_build)
    status = lib.vbake_pmr_visibility_sh_least_squares(
        ctypes.c_int(int(positions.shape[0])),
        ctypes.c_int(int(indices.shape[0])),
        positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        triangle_areas.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(samples_per_triangle),
        sample_barycentrics.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sample_sh16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(max(0.0, float(edge_regularization))),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        message = lib.vbake_last_error()
        decoded = message.decode("utf-8", errors="replace") if message else "unknown native PMR bake error"
        raise NativeBakeError(f"vbake_pmr_visibility_sh_least_squares failed with status {status}: {decoded}")
    return out


def pmr_sh_to_cones_native(
    vertex_sh16,
    fallback_normals,
    *,
    auto_build: bool = True,
) -> npt.NDArray[np.float32]:
    vertex_sh16 = _as_float32_contiguous(vertex_sh16, (16,), "vertex_sh16")
    fallback_normals = _as_float32_contiguous(fallback_normals, (3,), "fallback_normals")
    if fallback_normals.shape[0] != vertex_sh16.shape[0]:
        raise ValueError("vertex_sh16 and fallback_normals must have matching lengths")

    out = np.zeros((vertex_sh16.shape[0], 5), dtype=np.float32)
    lib = _load_library(auto_build=auto_build)
    status = lib.vbake_pmr_sh_to_cones(
        ctypes.c_int(int(vertex_sh16.shape[0])),
        vertex_sh16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fallback_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        message = lib.vbake_last_error()
        decoded = message.decode("utf-8", errors="replace") if message else "unknown native PMR cone error"
        raise NativeBakeError(f"vbake_pmr_sh_to_cones failed with status {status}: {decoded}")
    return out


def tinybvh_has_bvh8_cpu(*, auto_build: bool = True) -> bool:
    lib = _load_library(auto_build=auto_build)
    return bool(lib.vbake_tinybvh_has_bvh8_cpu())


def trace_pmr_visibility_sh_tinybvh(
    positions,
    indices,
    sample_positions,
    sample_normals,
    *,
    ray_count: int = 512,
    max_distance: float = 0.5,
    self_bias: float = 0.001,
    thread_count: int = 0,
    layout: str = "auto",
    auto_build: bool = True,
) -> tuple[npt.NDArray[np.float32], TinyBvhTraceStats]:
    positions = _as_float32_contiguous(positions, (3,), "positions")
    indices = _as_uint32_contiguous(indices, (3,), "indices")
    sample_positions = _as_float32_contiguous(sample_positions, (3,), "sample_positions")
    sample_normals = _as_float32_contiguous(sample_normals, (3,), "sample_normals")
    if sample_normals.shape[0] != sample_positions.shape[0]:
        raise ValueError("sample_positions and sample_normals must have matching lengths")
    if positions.shape[0] == 0 or indices.shape[0] == 0:
        raise ValueError("TinyBVH geometry must contain vertices and triangles")
    if sample_positions.shape[0] == 0:
        raise ValueError("at least one visibility sample is required")
    ray_count = int(ray_count)
    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    thread_count = int(thread_count)
    if thread_count < 0:
        raise ValueError("thread_count must be non-negative")

    lib = _load_library(auto_build=auto_build)
    has_bvh8 = bool(lib.vbake_tinybvh_has_bvh8_cpu())
    normalized_layout = str(layout).lower().replace("-", "_")
    if normalized_layout == "auto":
        normalized_layout = "bvh"
    if normalized_layout in ("bvh8_cpu", "bvh8cpu"):
        normalized_layout = "bvh8"
    if normalized_layout not in ("bvh", "bvh8"):
        raise ValueError("layout must be one of: auto, bvh, bvh8")
    if normalized_layout == "bvh8" and not has_bvh8:
        raise NativeBakeError("TinyBVH BVH8_CPU is unavailable in this build; use layout='bvh'")
    layout_index = 1 if normalized_layout == "bvh8" else 0

    sample_count = int(sample_positions.shape[0])
    out = np.zeros((sample_count, 16), dtype=np.float32)
    build_milliseconds = ctypes.c_double(0.0)
    trace_milliseconds = ctypes.c_double(0.0)
    visible_ray_count = ctypes.c_uint64(0)
    actual_thread_count = ctypes.c_int(0)
    status = lib.vbake_pmr_visibility_sh_tinybvh(
        ctypes.c_int(int(positions.shape[0])),
        ctypes.c_int(int(indices.shape[0])),
        positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int(sample_count),
        sample_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sample_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(ray_count),
        ctypes.c_float(float(max_distance)),
        ctypes.c_float(float(self_bias)),
        ctypes.c_int(thread_count),
        ctypes.c_int(layout_index),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(build_milliseconds),
        ctypes.byref(trace_milliseconds),
        ctypes.byref(visible_ray_count),
        ctypes.byref(actual_thread_count),
    )
    if status != 0:
        message = lib.vbake_last_error()
        decoded = message.decode("utf-8", errors="replace") if message else "unknown TinyBVH trace error"
        raise NativeBakeError(f"vbake_pmr_visibility_sh_tinybvh failed with status {status}: {decoded}")
    stats = TinyBvhTraceStats(
        layout=normalized_layout,
        thread_count=int(actual_thread_count.value),
        build_milliseconds=float(build_milliseconds.value),
        trace_milliseconds=float(trace_milliseconds.value),
        visible_ray_count=int(visible_ray_count.value),
        sample_count=sample_count,
        ray_count=ray_count,
    )
    return out, stats
