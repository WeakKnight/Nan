from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import os
import platform
import shutil
import subprocess

import numpy as np
import numpy.typing as npt


PROJECT_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_DIR / "surface_probe_sampler_build"


class CppSurfaceProbeSamplerUnavailable(RuntimeError):
    pass


class CppSurfaceProbeSamplerError(RuntimeError):
    pass


class _WSEOptions(ctypes.Structure):
    _fields_ = [
        ("surface_area", ctypes.c_float),
        ("normal_cosine_threshold", ctypes.c_float),
        ("plane_distance_scale", ctypes.c_float),
    ]


class _AdaptiveWSEProfile(ctypes.Structure):
    _fields_ = [
        ("setup_ms", ctypes.c_double),
        ("stage1_partition_ms", ctypes.c_double),
        ("stage1_eliminate_wall_ms", ctypes.c_double),
        ("stage1_pack_cpu_ms", ctypes.c_double),
        ("stage1_grid_cpu_ms", ctypes.c_double),
        ("stage1_weights_cpu_ms", ctypes.c_double),
        ("stage1_heap_cpu_ms", ctypes.c_double),
        ("stage2_partition_ms", ctypes.c_double),
        ("stage2_eliminate_wall_ms", ctypes.c_double),
        ("stage2_pack_cpu_ms", ctypes.c_double),
        ("stage2_grid_cpu_ms", ctypes.c_double),
        ("stage2_weights_cpu_ms", ctypes.c_double),
        ("stage2_heap_cpu_ms", ctypes.c_double),
        ("final_pack_ms", ctypes.c_double),
        ("final_grid_ms", ctypes.c_double),
        ("final_weights_ms", ctypes.c_double),
        ("final_heap_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("stage1_input_count", ctypes.c_uint32),
        ("stage1_output_count", ctypes.c_uint32),
        ("stage2_output_count", ctypes.c_uint32),
        ("final_output_count", ctypes.c_uint32),
        ("stage1_partition_count", ctypes.c_uint32),
        ("stage2_partition_count", ctypes.c_uint32),
        ("parallel_path", ctypes.c_uint32),
    ]


class _RepairOptions(ctypes.Structure):
    _fields_ = [
        ("min_gather_count", ctypes.c_uint32),
        ("max_repair_count", ctypes.c_uint32),
        ("normal_cosine_threshold", ctypes.c_float),
        ("weight_epsilon", ctypes.c_float),
    ]


class _RepairProfile(ctypes.Structure):
    _fields_ = [
        ("acceleration_structure_ms", ctypes.c_double),
        ("base_gather_ms", ctypes.c_double),
        ("coverage_build_ms", ctypes.c_double),
        ("heap_build_ms", ctypes.c_double),
        ("greedy_select_ms", ctypes.c_double),
        ("affected_audits_ms", ctypes.c_double),
        ("final_gather_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("coverage_pair_count", ctypes.c_uint64),
        ("affected_audit_count", ctypes.c_uint32),
        ("worker_count", ctypes.c_uint32),
    ]


class _CandidateFilterProfile(ctypes.Structure):
    _fields_ = [
        ("audit_partition_ms", ctypes.c_double),
        ("audit_deduplicate_ms", ctypes.c_double),
        ("repair_partition_ms", ctypes.c_double),
        ("repair_exclude_ms", ctypes.c_double),
        ("compact_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("audit_output_count", ctypes.c_uint32),
        ("repair_output_count", ctypes.c_uint32),
        ("shard_count", ctypes.c_uint32),
        ("worker_count", ctypes.c_uint32),
    ]


class _SupportOptions(ctypes.Structure):
    _fields_ = [
        ("normal_cosine_threshold", ctypes.c_float),
        ("weight_epsilon", ctypes.c_float),
        ("max_density_multiplier", ctypes.c_float),
    ]


class _PointOctreeProfile(ctypes.Structure):
    _fields_ = [
        ("bounds_ms", ctypes.c_double),
        ("index_setup_ms", ctypes.c_double),
        ("partition_ms", ctypes.c_double),
        ("flatten_ms", ctypes.c_double),
        ("output_copy_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("worker_count", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
    ]


class _PointOctreeResult(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(ctypes.c_uint32)),
        ("node_count", ctypes.c_uint32),
        ("probe_order", ctypes.POINTER(ctypes.c_uint32)),
        ("probe_count", ctypes.c_uint32),
        ("root_center", ctypes.c_double * 3),
        ("root_extent", ctypes.c_double),
    ]


@dataclass(frozen=True)
class AdaptiveWSEProfile:
    setup_ms: float
    stage1_partition_ms: float
    stage1_eliminate_wall_ms: float
    stage1_pack_cpu_ms: float
    stage1_grid_cpu_ms: float
    stage1_weights_cpu_ms: float
    stage1_heap_cpu_ms: float
    stage2_partition_ms: float
    stage2_eliminate_wall_ms: float
    stage2_pack_cpu_ms: float
    stage2_grid_cpu_ms: float
    stage2_weights_cpu_ms: float
    stage2_heap_cpu_ms: float
    final_pack_ms: float
    final_grid_ms: float
    final_weights_ms: float
    final_heap_ms: float
    total_ms: float
    stage1_input_count: int
    stage1_output_count: int
    stage2_output_count: int
    final_output_count: int
    stage1_partition_count: int
    stage2_partition_count: int
    parallel_path: bool


@dataclass(frozen=True)
class DeficitRepairProfile:
    acceleration_structure_ms: float
    base_gather_ms: float
    coverage_build_ms: float
    heap_build_ms: float
    greedy_select_ms: float
    affected_audits_ms: float
    final_gather_ms: float
    total_ms: float
    coverage_pair_count: int
    affected_audit_count: int
    worker_count: int


@dataclass(frozen=True)
class CandidateFilterProfile:
    audit_partition_ms: float
    audit_deduplicate_ms: float
    repair_partition_ms: float
    repair_exclude_ms: float
    compact_ms: float
    total_ms: float
    audit_output_count: int
    repair_output_count: int
    shard_count: int
    worker_count: int


@dataclass(frozen=True)
class CandidateFilterResult:
    audit_indices: npt.NDArray[np.int64]
    repair_indices: npt.NDArray[np.int64]
    profile: CandidateFilterProfile


@dataclass(frozen=True)
class DeficitRepairResult:
    selected_candidate_indices: npt.NDArray[np.int64]
    counts_before: npt.NDArray[np.uint32]
    counts_after: npt.NDArray[np.uint32]
    weight_sums_before: npt.NDArray[np.float32]
    weight_sums_after: npt.NDArray[np.float32]
    ess_before: npt.NDArray[np.float32]
    ess_after: npt.NDArray[np.float32]
    profile: DeficitRepairProfile | None = None


@dataclass(frozen=True)
class SupportEstimateResult:
    support_f: npt.NDArray[np.float32]
    density_m: npt.NDArray[np.float32]


@dataclass(frozen=True)
class PointOctreeProfile:
    bounds_ms: float
    index_setup_ms: float
    partition_ms: float
    flatten_ms: float
    output_copy_ms: float
    total_ms: float
    worker_count: int
    node_count: int


_loaded_library: ctypes.CDLL | None = None


def _find_cmake() -> str:
    executable = shutil.which("cmake")
    if executable:
        return executable
    if platform.system() != "Windows":
        raise CppSurfaceProbeSamplerUnavailable("cmake was not found on PATH")

    candidates: list[Path] = []
    for environment_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(environment_name)
        if not root:
            continue
        visual_studio = Path(root) / "Microsoft Visual Studio"
        if visual_studio.exists():
            candidates.extend(
                visual_studio.glob(
                    "*/*/Common7/IDE/CommonExtensions/Microsoft/"
                    "CMake/CMake/bin/cmake.exe"
                )
            )
    if candidates:
        return str(sorted(candidates, reverse=True)[0])
    raise CppSurfaceProbeSamplerUnavailable(
        "cmake was not found on PATH or in a Visual Studio installation"
    )


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
    except OSError as exc:
        raise CppSurfaceProbeSamplerUnavailable(
            f"failed to launch CMake: {exc}"
        ) from exc
    if result.returncode != 0:
        details = "\n".join(
            part
            for part in (result.stdout.strip(), result.stderr.strip())
            if part
        )
        raise CppSurfaceProbeSamplerUnavailable(
            f"CMake command failed: {' '.join(command)}\n{details}"
        )


def build_cpp_surface_probe_sampler() -> Path:
    cmake = _find_cmake()
    configure = [cmake, "-S", str(PROJECT_DIR), "-B", str(BUILD_DIR)]
    if platform.system() == "Windows" and not (
        BUILD_DIR / "CMakeCache.txt"
    ).exists():
        configure.extend(["-A", "x64"])
    _run_cmake(configure)
    _run_cmake(
        [
            cmake,
            "--build",
            str(BUILD_DIR),
            "--config",
            "Release",
            "--target",
            "surface_probe_sampler",
            "--parallel",
        ]
    )

    names = (
        ["surface_probe_sampler.dll"]
        if platform.system() == "Windows"
        else [
            "libsurface_probe_sampler.dylib",
            "libsurface_probe_sampler.so",
        ]
    )
    for name in names:
        matches = sorted(BUILD_DIR.rglob(name), key=lambda path: len(path.parts))
        if matches:
            return matches[0]
    raise CppSurfaceProbeSamplerUnavailable(
        "CMake build succeeded but no surface probe sampler library was found "
        f"in {BUILD_DIR}"
    )


def load_cpp_surface_probe_sampler() -> ctypes.CDLL:
    global _loaded_library
    if _loaded_library is not None:
        return _loaded_library

    library_path = build_cpp_surface_probe_sampler()
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise CppSurfaceProbeSamplerUnavailable(
            f"failed to load {library_path}: {exc}"
        ) from exc
    library.surface_probe_wse_eliminate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_WSEOptions),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.surface_probe_wse_eliminate.restype = ctypes.c_int
    library.surface_probe_wse_eliminate_adaptive.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_WSEOptions),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(_AdaptiveWSEProfile),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.surface_probe_wse_eliminate_adaptive.restype = ctypes.c_int
    library.surface_probe_deficit_repair.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(_RepairOptions),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(_RepairProfile),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.surface_probe_deficit_repair.restype = ctypes.c_int
    library.surface_probe_filter_audit_repair_candidates.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_CandidateFilterProfile),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.surface_probe_filter_audit_repair_candidates.restype = ctypes.c_int
    library.surface_probe_estimate_support.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(_SupportOptions),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.surface_probe_estimate_support.restype = ctypes.c_int
    library.surface_probe_build_point_octree.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_PointOctreeResult),
        ctypes.POINTER(_PointOctreeProfile),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.surface_probe_build_point_octree.restype = ctypes.c_int
    library.surface_probe_free_point_octree.argtypes = [
        ctypes.POINTER(_PointOctreeResult)
    ]
    library.surface_probe_free_point_octree.restype = None
    library.surface_probe_wse_version.argtypes = []
    library.surface_probe_wse_version.restype = ctypes.c_char_p
    _loaded_library = library
    return library


def cpp_surface_probe_sampler_version() -> str:
    raw = load_cpp_surface_probe_sampler().surface_probe_wse_version()
    return raw.decode("utf-8", errors="replace") if raw else "unknown"


def build_point_octree_cpp(
    positions: npt.ArrayLike,
    *,
    leaf_capacity: int,
    max_depth: int,
    profile_sink: list[PointOctreeProfile] | None = None,
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.int64],
    tuple[float, float, float],
    float,
]:
    position_array = np.ascontiguousarray(positions, dtype=np.float32)
    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if position_array.shape[0] == 0:
        raise ValueError("point octree requires at least one position")
    if not np.all(np.isfinite(position_array)):
        raise ValueError("point octree positions must be finite")
    point_count = int(position_array.shape[0])
    if point_count > np.iinfo(np.uint32).max:
        raise ValueError("native point octree supports at most uint32 points")
    leaf_capacity = int(leaf_capacity)
    max_depth = int(max_depth)
    if leaf_capacity <= 0 or max_depth <= 0:
        raise ValueError("leaf_capacity and max_depth must be positive")

    result = _PointOctreeResult()
    native_profile = _PointOctreeProfile() if profile_sink is not None else None
    error = ctypes.create_string_buffer(2048)
    library = load_cpp_surface_probe_sampler()
    ok = library.surface_probe_build_point_octree(
        position_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(point_count),
        ctypes.c_uint32(leaf_capacity),
        ctypes.c_uint32(max_depth),
        ctypes.byref(result),
        ctypes.byref(native_profile) if native_profile is not None else None,
        error,
        ctypes.sizeof(error),
    )
    if not ok:
        message = error.value.decode("utf-8", errors="replace")
        raise CppSurfaceProbeSamplerError(
            message or "native point octree construction failed"
        )
    try:
        nodes = np.ctypeslib.as_array(
            result.nodes, shape=(int(result.node_count) * 4,)
        ).copy().reshape((-1, 4))
        order = np.ctypeslib.as_array(
            result.probe_order, shape=(int(result.probe_count),)
        ).copy().astype(np.int64)
        root_center = tuple(float(value) for value in result.root_center)
        root_extent = float(result.root_extent)
        if native_profile is not None and profile_sink is not None:
            profile_sink.append(
                PointOctreeProfile(
                    bounds_ms=float(native_profile.bounds_ms),
                    index_setup_ms=float(native_profile.index_setup_ms),
                    partition_ms=float(native_profile.partition_ms),
                    flatten_ms=float(native_profile.flatten_ms),
                    output_copy_ms=float(native_profile.output_copy_ms),
                    total_ms=float(native_profile.total_ms),
                    worker_count=int(native_profile.worker_count),
                    node_count=int(native_profile.node_count),
                )
            )
        return nodes, order, root_center, root_extent
    finally:
        library.surface_probe_free_point_octree(ctypes.byref(result))


def weighted_sample_elimination_cpp(
    positions: npt.ArrayLike,
    normals: npt.ArrayLike,
    output_count: int,
    *,
    surface_area: float,
    normal_cosine_threshold: float = 0.5,
    plane_distance_scale: float = 0.35,
) -> tuple[npt.NDArray[np.int64], float]:
    position_array = np.ascontiguousarray(positions, dtype=np.float32)
    normal_array = np.ascontiguousarray(normals, dtype=np.float32)
    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if normal_array.shape != position_array.shape:
        raise ValueError("normals must have the same shape as positions")
    input_count = int(position_array.shape[0])
    output_count = int(output_count)
    if input_count > np.iinfo(np.uint32).max:
        raise ValueError("native surface probe WSE supports at most uint32 samples")
    if output_count <= 0 or output_count > input_count:
        raise ValueError(
            "output_count must be positive and no larger than the input count"
        )

    output = np.empty((output_count,), dtype=np.uint32)
    radius = ctypes.c_float()
    options = _WSEOptions(
        surface_area=float(surface_area),
        normal_cosine_threshold=float(normal_cosine_threshold),
        plane_distance_scale=float(plane_distance_scale),
    )
    error = ctypes.create_string_buffer(2048)
    library = load_cpp_surface_probe_sampler()
    ok = library.surface_probe_wse_eliminate(
        position_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        normal_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(input_count),
        ctypes.c_uint32(output_count),
        ctypes.byref(options),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(radius),
        error,
        ctypes.sizeof(error),
    )
    if not ok:
        message = error.value.decode("utf-8", errors="replace")
        raise CppSurfaceProbeSamplerError(
            message or "native surface probe weighted sample elimination failed"
        )
    return output.astype(np.int64), float(radius.value)


def adaptive_weighted_sample_elimination_cpp(
    positions: npt.ArrayLike,
    normals: npt.ArrayLike,
    relative_densities: npt.ArrayLike,
    output_count: int,
    *,
    surface_area: float,
    partition_masses: npt.ArrayLike | None = None,
    normal_cosine_threshold: float = 0.5,
    plane_distance_scale: float = 0.35,
    profile_sink: list[AdaptiveWSEProfile] | None = None,
) -> tuple[npt.NDArray[np.int64], float]:
    position_array = np.ascontiguousarray(positions, dtype=np.float32)
    normal_array = np.ascontiguousarray(normals, dtype=np.float32)
    density_array = np.ascontiguousarray(
        relative_densities, dtype=np.float32
    )
    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if normal_array.shape != position_array.shape:
        raise ValueError("normals must have the same shape as positions")
    input_count = int(position_array.shape[0])
    if density_array.shape != (input_count,) or not np.all(
        np.isfinite(density_array) & (density_array > 0.0)
    ):
        raise ValueError(
            "relative_densities must be finite, positive, and shape (N,)"
        )
    mass_array = np.ascontiguousarray(
        density_array if partition_masses is None else partition_masses,
        dtype=np.float32,
    )
    if mass_array.shape != (input_count,) or not np.all(
        np.isfinite(mass_array) & (mass_array > 0.0)
    ):
        raise ValueError(
            "partition_masses must be finite, positive, and shape (N,)"
        )
    output_count = int(output_count)
    if input_count > np.iinfo(np.uint32).max:
        raise ValueError("native adaptive WSE supports at most uint32 samples")
    if output_count <= 0 or output_count > input_count:
        raise ValueError(
            "output_count must be positive and no larger than the input count"
        )

    output = np.empty((output_count,), dtype=np.uint32)
    radius = ctypes.c_float()
    options = _WSEOptions(
        surface_area=float(surface_area),
        normal_cosine_threshold=float(normal_cosine_threshold),
        plane_distance_scale=float(plane_distance_scale),
    )
    error = ctypes.create_string_buffer(2048)
    native_profile = _AdaptiveWSEProfile() if profile_sink is not None else None
    library = load_cpp_surface_probe_sampler()
    ok = library.surface_probe_wse_eliminate_adaptive(
        position_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        normal_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        density_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mass_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(input_count),
        ctypes.c_uint32(output_count),
        ctypes.byref(options),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(radius),
        ctypes.byref(native_profile) if native_profile is not None else None,
        error,
        ctypes.sizeof(error),
    )
    if not ok:
        message = error.value.decode("utf-8", errors="replace")
        raise CppSurfaceProbeSamplerError(
            message or "native adaptive weighted sample elimination failed"
        )
    if native_profile is not None and profile_sink is not None:
        profile_sink.append(
            AdaptiveWSEProfile(
                setup_ms=float(native_profile.setup_ms),
                stage1_partition_ms=float(native_profile.stage1_partition_ms),
                stage1_eliminate_wall_ms=float(
                    native_profile.stage1_eliminate_wall_ms
                ),
                stage1_pack_cpu_ms=float(native_profile.stage1_pack_cpu_ms),
                stage1_grid_cpu_ms=float(native_profile.stage1_grid_cpu_ms),
                stage1_weights_cpu_ms=float(
                    native_profile.stage1_weights_cpu_ms
                ),
                stage1_heap_cpu_ms=float(native_profile.stage1_heap_cpu_ms),
                stage2_partition_ms=float(native_profile.stage2_partition_ms),
                stage2_eliminate_wall_ms=float(
                    native_profile.stage2_eliminate_wall_ms
                ),
                stage2_pack_cpu_ms=float(native_profile.stage2_pack_cpu_ms),
                stage2_grid_cpu_ms=float(native_profile.stage2_grid_cpu_ms),
                stage2_weights_cpu_ms=float(
                    native_profile.stage2_weights_cpu_ms
                ),
                stage2_heap_cpu_ms=float(native_profile.stage2_heap_cpu_ms),
                final_pack_ms=float(native_profile.final_pack_ms),
                final_grid_ms=float(native_profile.final_grid_ms),
                final_weights_ms=float(native_profile.final_weights_ms),
                final_heap_ms=float(native_profile.final_heap_ms),
                total_ms=float(native_profile.total_ms),
                stage1_input_count=int(native_profile.stage1_input_count),
                stage1_output_count=int(native_profile.stage1_output_count),
                stage2_output_count=int(native_profile.stage2_output_count),
                final_output_count=int(native_profile.final_output_count),
                stage1_partition_count=int(
                    native_profile.stage1_partition_count
                ),
                stage2_partition_count=int(
                    native_profile.stage2_partition_count
                ),
                parallel_path=bool(native_profile.parallel_path),
            )
        )
    return output.astype(np.int64), float(radius.value)


def filter_audit_repair_candidates_cpp(
    candidate_positions: npt.ArrayLike,
    candidate_normals: npt.ArrayLike,
    base_positions: npt.ArrayLike,
    base_normals: npt.ArrayLike,
    base_selected_indices: npt.ArrayLike,
    *,
    audit_cell_size: float,
    normal_cosine_threshold: float,
) -> CandidateFilterResult:
    candidate_p = np.ascontiguousarray(candidate_positions, dtype=np.float32)
    candidate_n = np.ascontiguousarray(candidate_normals, dtype=np.float32)
    base_p = np.ascontiguousarray(base_positions, dtype=np.float32)
    base_n = np.ascontiguousarray(base_normals, dtype=np.float32)
    if candidate_p.ndim != 2 or candidate_p.shape[1] != 3:
        raise ValueError("candidate_positions must have shape (N, 3)")
    if candidate_n.shape != candidate_p.shape:
        raise ValueError("candidate_normals must match candidate_positions")
    if base_p.ndim != 2 or base_p.shape[1] != 3:
        raise ValueError("base_positions must have shape (N, 3)")
    if base_n.shape != base_p.shape:
        raise ValueError("base_normals must match base_positions")
    if candidate_p.shape[0] == 0 or base_p.shape[0] == 0:
        raise ValueError("candidate and base arrays must be non-empty")
    if not np.all(np.isfinite(candidate_p)) or not np.all(
        np.isfinite(candidate_n)
    ):
        raise ValueError("candidate positions and normals must be finite")
    selected = np.ascontiguousarray(base_selected_indices, dtype=np.uint32)
    if selected.ndim != 1 or np.any(selected >= base_p.shape[0]):
        raise ValueError("base_selected_indices contains an invalid index")
    audit_cell_size = float(audit_cell_size)
    normal_cosine_threshold = float(normal_cosine_threshold)
    if not np.isfinite(audit_cell_size) or audit_cell_size <= 0.0:
        raise ValueError("audit_cell_size must be finite and positive")
    if not np.isfinite(normal_cosine_threshold) or not (
        -1.0 <= normal_cosine_threshold <= 1.0
    ):
        raise ValueError("normal_cosine_threshold must be within [-1, 1]")

    candidate_count = int(candidate_p.shape[0])
    audit_indices = np.empty((candidate_count,), dtype=np.uint32)
    repair_indices = np.empty((candidate_count,), dtype=np.uint32)
    audit_count = ctypes.c_uint32()
    repair_count = ctypes.c_uint32()
    profile = _CandidateFilterProfile()
    error = ctypes.create_string_buffer(2048)
    library = load_cpp_surface_probe_sampler()
    ok = library.surface_probe_filter_audit_repair_candidates(
        candidate_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        candidate_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(candidate_count),
        base_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        base_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(base_p.shape[0]),
        selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(selected.shape[0]),
        ctypes.c_double(audit_cell_size),
        ctypes.c_double(normal_cosine_threshold),
        audit_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(audit_count),
        repair_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(repair_count),
        ctypes.byref(profile),
        error,
        ctypes.sizeof(error),
    )
    if not ok:
        message = error.value.decode("utf-8", errors="replace")
        raise CppSurfaceProbeSamplerError(
            message or "native audit/repair candidate filtering failed"
        )
    return CandidateFilterResult(
        audit_indices=audit_indices[: int(audit_count.value)].astype(np.int64),
        repair_indices=repair_indices[: int(repair_count.value)].astype(np.int64),
        profile=CandidateFilterProfile(
            audit_partition_ms=float(profile.audit_partition_ms),
            audit_deduplicate_ms=float(profile.audit_deduplicate_ms),
            repair_partition_ms=float(profile.repair_partition_ms),
            repair_exclude_ms=float(profile.repair_exclude_ms),
            compact_ms=float(profile.compact_ms),
            total_ms=float(profile.total_ms),
            audit_output_count=int(profile.audit_output_count),
            repair_output_count=int(profile.repair_output_count),
            shard_count=int(profile.shard_count),
            worker_count=int(profile.worker_count),
        ),
    )


def _repair_arrays(
    positions: npt.ArrayLike,
    normals: npt.ArrayLike,
    instances: npt.ArrayLike,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_array = np.ascontiguousarray(positions, dtype=np.float32)
    normal_array = np.ascontiguousarray(normals, dtype=np.float32)
    instance_array = np.ascontiguousarray(instances, dtype=np.uint32)
    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError(f"{label}_positions must have shape (N, 3)")
    if normal_array.shape != position_array.shape:
        raise ValueError(f"{label}_normals must match {label}_positions")
    if instance_array.shape != (position_array.shape[0],):
        raise ValueError(f"{label}_instances must have shape (N,)")
    if not np.all(np.isfinite(position_array)) or not np.all(
        np.isfinite(normal_array)
    ):
        raise ValueError(f"{label} positions and normals must be finite")
    return position_array, normal_array, instance_array


def deficit_repair_cpp(
    base_positions: npt.ArrayLike,
    base_normals: npt.ArrayLike,
    base_instances: npt.ArrayLike,
    candidate_positions: npt.ArrayLike,
    candidate_normals: npt.ArrayLike,
    candidate_instances: npt.ArrayLike,
    audit_positions: npt.ArrayLike,
    audit_normals: npt.ArrayLike,
    audit_instances: npt.ArrayLike,
    instance_radii: npt.ArrayLike,
    *,
    min_gather_count: int = 4,
    max_repair_count: int,
    normal_cosine_threshold: float = 0.5,
    weight_epsilon: float = 1e-6,
) -> DeficitRepairResult:
    base_p, base_n, base_i = _repair_arrays(
        base_positions, base_normals, base_instances, "base"
    )
    candidate_p, candidate_n, candidate_i = _repair_arrays(
        candidate_positions,
        candidate_normals,
        candidate_instances,
        "candidate",
    )
    audit_p, audit_n, audit_i = _repair_arrays(
        audit_positions, audit_normals, audit_instances, "audit"
    )
    radii = np.ascontiguousarray(instance_radii, dtype=np.float32)
    if radii.ndim != 1 or radii.size == 0 or not np.all(
        np.isfinite(radii) & (radii > 0.0)
    ):
        raise ValueError("instance_radii must be a non-empty positive vector")
    if base_p.shape[0] == 0 or audit_p.shape[0] == 0:
        raise ValueError("base and audit inputs must be non-empty")
    min_gather_count = int(min_gather_count)
    if min_gather_count < 1 or min_gather_count > 32:
        raise ValueError("min_gather_count must be within [1, 32]")
    max_repair_count = max(0, min(int(max_repair_count), candidate_p.shape[0]))
    selected = np.empty((max(1, max_repair_count),), dtype=np.uint32)
    selected_count = ctypes.c_uint32()
    audit_count = audit_p.shape[0]
    counts_before = np.empty((audit_count,), dtype=np.uint32)
    counts_after = np.empty((audit_count,), dtype=np.uint32)
    weight_sums_before = np.empty((audit_count,), dtype=np.float32)
    weight_sums_after = np.empty((audit_count,), dtype=np.float32)
    ess_before = np.empty((audit_count,), dtype=np.float32)
    ess_after = np.empty((audit_count,), dtype=np.float32)
    profile = _RepairProfile()
    options = _RepairOptions(
        min_gather_count=min_gather_count,
        max_repair_count=max_repair_count,
        normal_cosine_threshold=float(normal_cosine_threshold),
        weight_epsilon=float(weight_epsilon),
    )
    error = ctypes.create_string_buffer(2048)
    library = load_cpp_surface_probe_sampler()
    ok = library.surface_probe_deficit_repair(
        base_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        base_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        base_i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(base_p.shape[0]),
        candidate_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        candidate_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        candidate_i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(candidate_p.shape[0]),
        audit_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        audit_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        audit_i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(audit_count),
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(radii.shape[0]),
        ctypes.byref(options),
        selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(selected_count),
        counts_before.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        counts_after.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        weight_sums_before.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weight_sums_after.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ess_before.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ess_after.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(profile),
        error,
        ctypes.sizeof(error),
    )
    if not ok:
        message = error.value.decode("utf-8", errors="replace")
        raise CppSurfaceProbeSamplerError(
            message or "native surface probe deficit repair failed"
        )
    count = int(selected_count.value)
    return DeficitRepairResult(
        selected_candidate_indices=selected[:count].astype(np.int64),
        counts_before=counts_before,
        counts_after=counts_after,
        weight_sums_before=weight_sums_before,
        weight_sums_after=weight_sums_after,
        ess_before=ess_before,
        ess_after=ess_after,
        profile=DeficitRepairProfile(
            acceleration_structure_ms=float(profile.acceleration_structure_ms),
            base_gather_ms=float(profile.base_gather_ms),
            coverage_build_ms=float(profile.coverage_build_ms),
            heap_build_ms=float(profile.heap_build_ms),
            greedy_select_ms=float(profile.greedy_select_ms),
            affected_audits_ms=float(profile.affected_audits_ms),
            final_gather_ms=float(profile.final_gather_ms),
            total_ms=float(profile.total_ms),
            coverage_pair_count=int(profile.coverage_pair_count),
            affected_audit_count=int(profile.affected_audit_count),
            worker_count=int(profile.worker_count),
        ),
    )


def _gather_weights_reference(
    query_position: np.ndarray,
    query_normal: np.ndarray,
    query_instance: int,
    probe_positions: np.ndarray,
    probe_normals: np.ndarray,
    probe_instances: np.ndarray,
    radius: float,
    normal_cosine_threshold: float,
    weight_epsilon: float,
) -> np.ndarray:
    mask = probe_instances == query_instance
    delta = probe_positions[mask] - query_position
    normals = probe_normals[mask]
    distance_squared = np.einsum("ij,ij->i", delta, delta)
    normal_cosine = normals @ query_normal
    valid = (distance_squared <= radius * radius) & (
        normal_cosine >= normal_cosine_threshold
    )
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32)
    distance_squared = distance_squared[valid]
    delta = delta[valid]
    normal_cosine = normal_cosine[valid]
    plane_sigma = max(radius * 0.25, 1e-6)
    plane_distance = np.abs(delta @ query_normal)
    plane_weight = np.exp(-0.5 * np.square(plane_distance / plane_sigma))
    spatial = np.maximum(1.0 - distance_squared / (radius * radius), 0.0)
    weights = spatial * spatial * plane_weight * np.maximum(normal_cosine, 0.0)
    positive = weights > weight_epsilon
    weights = weights[positive]
    order = np.argsort(-weights, kind="stable")[:96]
    weights = weights[order]
    return weights[:32].astype(np.float32)


def deficit_repair_python(
    base_positions: npt.ArrayLike,
    base_normals: npt.ArrayLike,
    base_instances: npt.ArrayLike,
    candidate_positions: npt.ArrayLike,
    candidate_normals: npt.ArrayLike,
    candidate_instances: npt.ArrayLike,
    audit_positions: npt.ArrayLike,
    audit_normals: npt.ArrayLike,
    audit_instances: npt.ArrayLike,
    instance_radii: npt.ArrayLike,
    *,
    min_gather_count: int = 4,
    max_repair_count: int,
    normal_cosine_threshold: float = 0.5,
    weight_epsilon: float = 1e-6,
) -> DeficitRepairResult:
    base_p, base_n, base_i = _repair_arrays(
        base_positions, base_normals, base_instances, "base"
    )
    candidate_p, candidate_n, candidate_i = _repair_arrays(
        candidate_positions,
        candidate_normals,
        candidate_instances,
        "candidate",
    )
    audit_p, audit_n, audit_i = _repair_arrays(
        audit_positions, audit_normals, audit_instances, "audit"
    )
    radii = np.ascontiguousarray(instance_radii, dtype=np.float32)
    min_gather_count = max(1, min(32, int(min_gather_count)))
    max_repair_count = max(0, min(int(max_repair_count), candidate_p.shape[0]))
    def summarize(probe_p, probe_n, probe_i):
        counts = np.zeros((audit_p.shape[0],), dtype=np.uint32)
        sums = np.zeros((audit_p.shape[0],), dtype=np.float32)
        ess = np.zeros((audit_p.shape[0],), dtype=np.float32)
        for audit in range(audit_p.shape[0]):
            weights = _gather_weights_reference(
                audit_p[audit],
                audit_n[audit],
                int(audit_i[audit]),
                probe_p,
                probe_n,
                probe_i,
                float(radii[int(audit_i[audit])]),
                normal_cosine_threshold,
                weight_epsilon,
            )
            counts[audit] = weights.size
            sums[audit] = np.sum(weights, dtype=np.float64)
            square_sum = np.sum(np.square(weights), dtype=np.float64)
            ess[audit] = (
                float(sums[audit]) ** 2 / square_sum
                if square_sum > 0.0
                else 0.0
            )
        return counts, sums, ess

    counts_before, sums_before, ess_before = summarize(base_p, base_n, base_i)
    deficits = np.maximum(
        min_gather_count - counts_before.astype(np.int64), 0
    )
    coverage: list[np.ndarray] = []
    for candidate in range(candidate_p.shape[0]):
        covered = []
        for audit in np.flatnonzero(deficits):
            weights = _gather_weights_reference(
                audit_p[audit],
                audit_n[audit],
                int(audit_i[audit]),
                candidate_p[candidate : candidate + 1],
                candidate_n[candidate : candidate + 1],
                candidate_i[candidate : candidate + 1],
                float(radii[int(audit_i[audit])]),
                normal_cosine_threshold,
                weight_epsilon,
            )
            if weights.size:
                covered.append(int(audit))
        coverage.append(np.asarray(covered, dtype=np.int64))

    selected: list[int] = []
    active = np.ones((candidate_p.shape[0],), dtype=np.bool_)
    while len(selected) < max_repair_count:
        best_index = -1
        best_score = 0
        for candidate in np.flatnonzero(active):
            d = deficits[coverage[int(candidate)]]
            d = d[d > 0]
            score = int(np.sum(2 * d - 1, dtype=np.int64))
            if score > best_score:
                best_score = score
                best_index = int(candidate)
        if best_index < 0:
            break
        active[best_index] = False
        selected.append(best_index)
        affected = coverage[best_index]
        deficits[affected] = np.maximum(deficits[affected] - 1, 0)

    selected_array = np.asarray(selected, dtype=np.int64)
    final_p = np.concatenate((base_p, candidate_p[selected_array]), axis=0)
    final_n = np.concatenate((base_n, candidate_n[selected_array]), axis=0)
    final_i = np.concatenate((base_i, candidate_i[selected_array]), axis=0)
    counts_after, sums_after, ess_after = summarize(final_p, final_n, final_i)
    return DeficitRepairResult(
        selected_candidate_indices=selected_array,
        counts_before=counts_before,
        counts_after=counts_after,
        weight_sums_before=sums_before,
        weight_sums_after=sums_after,
        ess_before=ess_before,
        ess_after=ess_after,
    )


def estimate_support_cpp(
    reference_positions: npt.ArrayLike,
    reference_normals: npt.ArrayLike,
    reference_instances: npt.ArrayLike,
    reference_area_weights: npt.ArrayLike,
    query_positions: npt.ArrayLike,
    query_normals: npt.ArrayLike,
    query_instances: npt.ArrayLike,
    instance_radii: npt.ArrayLike,
    *,
    normal_cosine_threshold: float = 0.5,
    weight_epsilon: float = 1e-6,
    max_density_multiplier: float = 8.0,
) -> SupportEstimateResult:
    reference_p, reference_n, reference_i = _repair_arrays(
        reference_positions,
        reference_normals,
        reference_instances,
        "reference",
    )
    query_p, query_n, query_i = _repair_arrays(
        query_positions, query_normals, query_instances, "query"
    )
    area_weights = np.ascontiguousarray(
        reference_area_weights, dtype=np.float32
    )
    radii = np.ascontiguousarray(instance_radii, dtype=np.float32)
    if area_weights.shape != (reference_p.shape[0],) or not np.all(
        np.isfinite(area_weights) & (area_weights >= 0.0)
    ):
        raise ValueError(
            "reference_area_weights must be finite, non-negative, and shape (N,)"
        )
    if radii.ndim != 1 or radii.size == 0 or not np.all(
        np.isfinite(radii) & (radii > 0.0)
    ):
        raise ValueError("instance_radii must be a non-empty positive vector")
    max_density_multiplier = max(1.0, float(max_density_multiplier))
    support_f = np.empty((query_p.shape[0],), dtype=np.float32)
    density_m = np.empty((query_p.shape[0],), dtype=np.float32)
    options = _SupportOptions(
        normal_cosine_threshold=float(normal_cosine_threshold),
        weight_epsilon=float(weight_epsilon),
        max_density_multiplier=max_density_multiplier,
    )
    error = ctypes.create_string_buffer(2048)
    library = load_cpp_surface_probe_sampler()
    ok = library.surface_probe_estimate_support(
        reference_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        reference_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        reference_i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        area_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(reference_p.shape[0]),
        query_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        query_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        query_i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(query_p.shape[0]),
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(radii.shape[0]),
        ctypes.byref(options),
        support_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        density_m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error,
        ctypes.sizeof(error),
    )
    if not ok:
        message = error.value.decode("utf-8", errors="replace")
        raise CppSurfaceProbeSamplerError(
            message or "native surface probe support estimation failed"
        )
    return SupportEstimateResult(support_f=support_f, density_m=density_m)


def estimate_support_python(
    reference_positions: npt.ArrayLike,
    reference_normals: npt.ArrayLike,
    reference_instances: npt.ArrayLike,
    reference_area_weights: npt.ArrayLike,
    query_positions: npt.ArrayLike,
    query_normals: npt.ArrayLike,
    query_instances: npt.ArrayLike,
    instance_radii: npt.ArrayLike,
    *,
    normal_cosine_threshold: float = 0.5,
    weight_epsilon: float = 1e-6,
    max_density_multiplier: float = 8.0,
) -> SupportEstimateResult:
    reference_p, reference_n, reference_i = _repair_arrays(
        reference_positions,
        reference_normals,
        reference_instances,
        "reference",
    )
    query_p, query_n, query_i = _repair_arrays(
        query_positions, query_normals, query_instances, "query"
    )
    area_weights = np.ascontiguousarray(
        reference_area_weights, dtype=np.float32
    )
    radii = np.ascontiguousarray(instance_radii, dtype=np.float32)
    if area_weights.shape != (reference_p.shape[0],) or not np.all(
        np.isfinite(area_weights) & (area_weights >= 0.0)
    ):
        raise ValueError(
            "reference_area_weights must be finite, non-negative, and shape (N,)"
        )
    if radii.ndim != 1 or radii.size == 0 or not np.all(
        np.isfinite(radii) & (radii > 0.0)
    ):
        raise ValueError("instance_radii must be a non-empty positive vector")
    if (
        (reference_i.size and int(np.max(reference_i)) >= radii.size)
        or (query_i.size and int(np.max(query_i)) >= radii.size)
    ):
        raise ValueError("sample instance index exceeds instance_radii")
    support_f = np.empty((query_p.shape[0],), dtype=np.float32)
    density_m = np.empty((query_p.shape[0],), dtype=np.float32)
    max_density_multiplier = max(1.0, float(max_density_multiplier))
    for query in range(query_p.shape[0]):
        instance = int(query_i[query])
        radius = float(radii[instance])
        instance_mask = reference_i == instance
        delta = reference_p[instance_mask] - query_p[query]
        normals = reference_n[instance_mask]
        local_area_weights = area_weights[instance_mask]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        normal_cosine = normals @ query_n[query]
        valid = (distance_squared <= radius * radius) & (
            normal_cosine >= normal_cosine_threshold
        )
        if np.any(valid):
            local_delta = delta[valid]
            d2 = distance_squared[valid]
            cosine = normal_cosine[valid]
            plane_sigma = max(radius * 0.25, 1e-6)
            plane = np.exp(
                -0.5
                * np.square((local_delta @ query_n[query]) / plane_sigma)
            )
            spatial = np.maximum(1.0 - d2 / (radius * radius), 0.0)
            kernel = spatial * spatial * plane * np.maximum(cosine, 0.0)
            kernel = np.where(kernel > weight_epsilon, kernel, 0.0)
            support = float(
                np.sum(kernel * local_area_weights[valid], dtype=np.float64)
            )
        else:
            support = 0.0
        flat_support = np.pi * radius * radius / 3.0
        value = float(np.clip(support / max(flat_support, 1e-30), 0.0, 1.0))
        support_f[query] = value
        density_m[query] = min(
            max_density_multiplier,
            1.0 / max(value, 1.0 / max_density_multiplier),
        )
    return SupportEstimateResult(support_f=support_f, density_m=density_m)
