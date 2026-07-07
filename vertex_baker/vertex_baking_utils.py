from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt


ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "cpp"
BUILD_DIR = ROOT / "build" / "vertex_baking_utils"


class NativeBakeError(RuntimeError):
    pass


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
    source_paths = [
        CPP_DIR / "CMakeLists.txt",
        CPP_DIR / "vertex_baking_utils.cpp",
        CPP_DIR / "vertex_baking_utils.h",
    ]
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
