import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from sparse_shadow_tree import SparseShadowTreeEncoder


SHADOW_RESOLUTION_ALIASES = {
    "512": 512,
    "512p": 512,
    "1k": 1024,
    "1024": 1024,
    "1024p": 1024,
    "2k": 2048,
    "2048": 2048,
    "2048p": 2048,
    "4k": 4096,
    "4096": 4096,
    "4096p": 4096,
    "8k": 8192,
    "8192": 8192,
    "8192p": 8192,
}


def _parse_shadow_resolution(value: str) -> int | tuple[int, int]:
    token = value.strip().lower().replace("_", "").replace("-", "")
    for separator in ("x", "*", ","):
        if separator in token:
            parts = token.split(separator)
            if len(parts) != 2:
                break
            try:
                width = int(parts[0])
                height = int(parts[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError("shadow resolution size must be WIDTHxHEIGHT") from exc
            if width < 1 or height < 1:
                raise argparse.ArgumentTypeError("shadow resolution width and height must be positive")
            return (width, height)

    if token in SHADOW_RESOLUTION_ALIASES:
        return SHADOW_RESOLUTION_ALIASES[token]
    try:
        resolution = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shadow resolution must be WIDTHxHEIGHT, a positive integer, or one of 512, 1k, 2k, 4k, 8k"
        ) from exc
    if resolution < 1:
        raise argparse.ArgumentTypeError("shadow resolution must be positive")
    return resolution


def _shadow_resolution_budget(value: int | tuple[int, int]) -> int:
    if isinstance(value, tuple):
        return max(1, int(value[0]), int(value[1]))
    return max(1, int(value))


def _shadow_resolution_label(value: int | tuple[int, int]) -> str:
    if isinstance(value, tuple):
        return f"{max(1, int(value[0]))}x{max(1, int(value[1]))}"
    return str(max(1, int(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PythonRenderer entry point.")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Path to scene file (.json for scene description, other formats load as asset).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window or swapchain and exit after rendering frames.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=64,
        help="Number of frames to render in headless mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path when running headless.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Render width (also window width when interactive).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Render height (also window height when interactive).",
    )
    parser.add_argument(
        "--vsync",
        action="store_true",
        help="Enable V-Sync when running with a window.",
    )
    parser.add_argument(
        "--no-srgb",
        action="store_true",
        help="Disable sRGB conversion when saving headless output.",
    )
    parser.add_argument(
        "--camera-move-test",
        action="store_true",
        help="Enable scripted camera controller motion test.",
    )
    parser.add_argument(
        "--enable-extension",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="Enable optional rendering extensions. Built-in: static_shadow_sst. External form: python.module:ExtensionClass.",
    )
    parser.add_argument(
        "--static-shadow-mode",
        choices=("none", "realtime", "depth", "sst", "packed", "compact", "compact-pcf", "decompressed", "decomp"),
        default="none",
        help="Pre-bake static shadowing before render and select a runtime mode. 'compact-pcf' uses Compact SST with 3x3 PCF; 'decompressed' samples a GPU-decompressed SST depth texture.",
    )
    parser.add_argument(
        "--static-shadow-compare-modes",
        type=str,
        default=None,
        help="Comma-separated runtime static shadow modes to render and compare headlessly, e.g. depth,compact,compact-pcf.",
    )
    parser.add_argument(
        "--static-shadow-compare-output-dir",
        type=str,
        default="static_shadow_compare",
        help="Output directory for --static-shadow-compare-modes images and reports.",
    )
    parser.add_argument(
        "--static-shadow-compare-report",
        type=str,
        default=None,
        help="Optional JSON report path for --static-shadow-compare-modes. Defaults inside the output dir.",
    )
    parser.add_argument(
        "--static-shadow-compare-markdown-output",
        type=str,
        default=None,
        help="Optional Markdown report path for --static-shadow-compare-modes. Defaults inside the output dir.",
    )
    parser.add_argument(
        "--static-shadow-compare-report-inputs",
        type=str,
        default=None,
        help="Comma-separated existing static shadow compare JSON files to summarize into --static-shadow-compare-markdown-output without rerunning renders.",
    )
    parser.add_argument(
        "--static-shadow-compare-diff-scale",
        type=float,
        default=8.0,
        help="Multiplier for saved static shadow compare absolute-difference preview images.",
    )
    parser.add_argument(
        "--benchmark-sst",
        action="store_true",
        help="Run a headless Static Shadow Depth Map vs SST decode benchmark and exit.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help="Optional JSON output path for --benchmark-sst.",
    )
    parser.add_argument(
        "--benchmark-markdown-output",
        type=str,
        default=None,
        help="Optional Markdown summary output path for --benchmark-sst.",
    )
    parser.add_argument(
        "--sst-report-inputs",
        type=str,
        default=None,
        help="Comma-separated existing SST benchmark JSON files to summarize into --benchmark-markdown-output without rerunning bake.",
    )
    parser.add_argument(
        "--shadow-resolution",
        "--sst-resolution",
        dest="shadow_resolution",
        type=_parse_shadow_resolution,
        default=2048,
        metavar="N|WIDTHxHEIGHT|1k|2k|4k|8k",
        help="Static shadow/SST resolution budget, preset, or explicit non-square size. Used by runtime static shadow modes, compare, and --benchmark-sst.",
    )
    parser.add_argument(
        "--sst-tile-size",
        type=int,
        default=128,
        help="SST benchmark tile size.",
    )
    parser.add_argument(
        "--sst-encoder",
        choices=("auto", "cpp", "python"),
        default="auto",
        help="SST encoder backend. auto builds/uses the CMake C++ encoder when available, then falls back to Python.",
    )
    parser.add_argument(
        "--sst-min-leaf-size",
        type=int,
        default=2,
        help="SST minimum quadtree leaf size. Defaults to the Bistro-backed quality preset leaf=2.",
    )
    parser.add_argument(
        "--sst-fit-profile",
        choices=("bias", "half", "visible", "relaxed", "loose", "dual_bias", "dual_half_visible", "dual_visible", "dual_relaxed_visible", "dual_loose_visible"),
        default=None,
        help="SST fitting profile used by runtime static shadow modes and compare. Benchmark sweeps still use --sst-benchmark-variant/--sst-sweep-variants.",
    )
    parser.add_argument(
        "--sst-preset",
        choices=("quality", "high-compression", "high_compression", "manual"),
        default="quality",
        help="Runtime SST preset. quality=Dual Visible/tile128/leaf2, high-compression=Dual Relaxed Visible/tile128/leaf2; explicit SST options still override.",
    )
    parser.add_argument(
        "--static-shadow-mask",
        choices=("off", "full", "adaptive", "adaptive-wave"),
        default="off",
        help="Screen-space static shadow mask mode. adaptive-wave uses wave-level DistributeWork after a sparse 4x4 bootstrap.",
    )
    parser.add_argument(
        "--static-shadow-mask-threshold",
        type=float,
        default=0.02,
        help="Adaptive shadowmask max-min neighbor threshold. Lower values shade more pixels and reduce interpolation error.",
    )
    parser.add_argument(
        "--static-shadow-mask-bootstrap-passes",
        type=int,
        default=2,
        help="Adaptive shadowmask passes forced to true SST sampling before interpolation is allowed. 0 is fastest, 2 is conservative.",
    )
    parser.add_argument(
        "--sst-plane-error-threshold",
        type=float,
        default=0.0015,
        help="SST benchmark normalized depth error threshold for plane leaves.",
    )
    parser.add_argument(
        "--sst-constant-epsilon",
        type=float,
        default=0.0005,
        help="SST benchmark normalized depth range treated as constant.",
    )
    parser.add_argument(
        "--sst-plane-quantization-search-radius",
        type=int,
        default=0,
        help="Neighbor radius on the packed 16-bit slope grid when validating fitted SST planes.",
    )
    parser.add_argument(
        "--sst-dual-depth-slack",
        type=float,
        default=0.0015,
        help="Maximum normalized depth slack above first layer for capped dual-layer fitting.",
    )
    parser.add_argument(
        "--sst-shadow-bias",
        type=float,
        default=None,
        help="Normalized depth bias used to classify practical SST leak/over-shadow risk. Defaults to the scene static shadow bias.",
    )
    parser.add_argument(
        "--sst-forced-leaf-error-cap",
        type=float,
        default=None,
        help="Optional normalized max error cap for leaves forced by --sst-min-leaf-size. Above this, encoding may split below the min leaf.",
    )
    parser.add_argument(
        "--sst-forced-split-bias-fit",
        action="store_true",
        help="When forced-leaf error cap triggers extra splitting, fit that subtree with dual-bias visibility tolerance.",
    )
    parser.add_argument(
        "--sst-benchmark-variant",
        choices=(
            "all",
            "both",
            "single",
            "dual",
            "dual_raw",
            "dual_capped",
            "dual_bias",
            "dual_half_visible",
            "dual_visible",
            "dual_relaxed_visible",
            "dual_loose_visible",
            "dual_safe",
        ),
        default="all",
        help="Which SST fitting variant to benchmark.",
    )
    parser.add_argument(
        "--sst-debug-output-dir",
        type=str,
        default=None,
        help="Optional directory for SST benchmark debug map export (.npy plus PNG previews when Pillow is available).",
    )
    parser.add_argument(
        "--sst-gpu-decompress",
        action="store_true",
        help="In --benchmark-sst, run a GPU compute pass that decompresses Compact SST back to an r32 depth texture and compare it to CPU decode.",
    )
    parser.add_argument(
        "--sst-debug-variants",
        type=str,
        default=None,
        help="Comma-separated SST variants to export into --sst-debug-output-dir. Defaults to --sst-benchmark-variant selection.",
    )
    parser.add_argument(
        "--sst-sweep",
        action="store_true",
        help="Sweep SST fitting parameters after one bake/readback and report Pareto candidates.",
    )
    parser.add_argument(
        "--sst-sweep-plane-error-thresholds",
        type=str,
        default="0.00075,0.001,0.0015,0.002,0.003",
        help="Comma-separated normalized plane error thresholds used by --sst-sweep.",
    )
    parser.add_argument(
        "--sst-sweep-dual-depth-slacks",
        type=str,
        default="0.00075,0.0015,0.003",
        help="Comma-separated normalized dual-layer slacks used by --sst-sweep for capped variants.",
    )
    parser.add_argument(
        "--sst-sweep-plane-quantization-search-radii",
        type=str,
        default=None,
        help="Comma-separated packed slope quantization search radii used by --sst-sweep. Defaults to --sst-plane-quantization-search-radius.",
    )
    parser.add_argument(
        "--sst-sweep-tile-sizes",
        type=str,
        default=None,
        help="Comma-separated SST tile sizes used by --sst-sweep. Defaults to --sst-tile-size.",
    )
    parser.add_argument(
        "--sst-sweep-min-leaf-sizes",
        type=str,
        default=None,
        help="Comma-separated SST minimum leaf sizes used by --sst-sweep. Defaults to --sst-min-leaf-size.",
    )
    parser.add_argument(
        "--sst-sweep-forced-leaf-error-caps",
        type=str,
        default=None,
        help="Comma-separated forced-leaf error caps used by --sst-sweep. Use 'none' for uncapped. Defaults to --sst-forced-leaf-error-cap.",
    )
    parser.add_argument(
        "--sst-sweep-forced-split-bias-fit",
        type=str,
        default=None,
        help="Comma-separated booleans for forced-split bias fitting used by --sst-sweep. Defaults to --sst-forced-split-bias-fit.",
    )
    parser.add_argument(
        "--sst-sweep-variants",
        type=str,
        default="single,dual_capped,dual_bias,dual_half_visible,dual_visible,dual_relaxed_visible,dual_loose_visible,dual_safe",
        help="Comma-separated SST variants used by --sst-sweep.",
    )
    return parser.parse_args()


def _normalize_extension_name(value: str) -> str:
    raw = value.strip()
    if ":" in raw:
        return raw
    name = raw.lower().replace("-", "_")
    aliases = {
        "static_shadow": "static_shadow_sst",
        "sst": "static_shadow_sst",
        "static_sst": "static_shadow_sst",
    }
    return aliases.get(name, name)


def _enabled_extensions_from_args(args: argparse.Namespace, static_shadow_mode: int | None) -> tuple[str, ...]:
    enabled: list[str] = []
    for value in args.enable_extension or ():
        for item in value.split(","):
            name = _normalize_extension_name(item)
            if name and name not in enabled:
                enabled.append(name)

    static_shadow_requested = static_shadow_mode is not None or args.static_shadow_mask != "off"
    if static_shadow_requested and "static_shadow_sst" not in enabled:
        enabled.append("static_shadow_sst")
    return tuple(enabled)


def _parse_float_list(value: str, fallback: tuple[float, ...]) -> tuple[float, ...]:
    parsed: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(max(0.0, float(item)))
    return tuple(parsed) if parsed else fallback


def _parse_static_shadow_mode(value: str) -> int | None:
    mode = value.strip().lower()
    if mode in ("", "none", "off"):
        return None

    from scene import Scene

    modes = {
        "realtime": Scene.SHADOW_MODE_REALTIME,
        "rt": Scene.SHADOW_MODE_REALTIME,
        "depth": Scene.SHADOW_MODE_DEPTH_TEXTURE,
        "depth-texture": Scene.SHADOW_MODE_DEPTH_TEXTURE,
        "sst": Scene.SHADOW_MODE_SST,
        "packed": Scene.SHADOW_MODE_PACKED_SST,
        "packed-sst": Scene.SHADOW_MODE_PACKED_SST,
        "compact": Scene.SHADOW_MODE_COMPACT_SST,
        "compact-sst": Scene.SHADOW_MODE_COMPACT_SST,
        "compact-pcf": Scene.SHADOW_MODE_COMPACT_SST_PCF3,
        "compact-pcf3": Scene.SHADOW_MODE_COMPACT_SST_PCF3,
        "decompressed": Scene.SHADOW_MODE_DECOMPRESSED_SST,
        "decompressed-sst": Scene.SHADOW_MODE_DECOMPRESSED_SST,
        "decomp": Scene.SHADOW_MODE_DECOMPRESSED_SST,
    }
    if mode not in modes:
        raise ValueError(f"Unsupported static shadow mode '{value}'")
    return modes[mode]


def _parse_static_shadow_mode_list(value: str) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        mode = _parse_static_shadow_mode(token)
        if mode is None:
            raise ValueError("--static-shadow-compare-modes does not support 'none'")
        parsed.append((_static_shadow_mode_slug(token), mode))
    if len(parsed) < 2:
        raise ValueError("--static-shadow-compare-modes requires at least two modes")
    return tuple(parsed)


def _static_shadow_mode_slug(value: str) -> str:
    slug = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "rt": "realtime",
        "depth-texture": "depth",
        "packed-sst": "packed",
        "compact-sst": "compact",
        "compact-pcf3": "compact-pcf",
        "decompressed-sst": "decompressed",
        "decomp": "decompressed",
    }
    return aliases.get(slug, slug)


def _parse_optional_float_list(value: str | None, fallback: tuple[float | None, ...]) -> tuple[float | None, ...]:
    if value is None:
        return fallback
    parsed: list[float | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in ("none", "null", "off"):
            parsed.append(None)
        else:
            parsed.append(max(0.0, float(item)))
    return tuple(parsed) if parsed else fallback


def _parse_int_list(value: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return fallback
    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(max(0, int(item)))
    return tuple(parsed) if parsed else fallback


def _parse_bool_list(value: str | None, fallback: tuple[bool, ...]) -> tuple[bool, ...]:
    if value is None:
        return fallback
    parsed: list[bool] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in ("1", "true", "yes", "on", "bias"):
            parsed.append(True)
        elif item in ("0", "false", "no", "off", "none"):
            parsed.append(False)
        else:
            raise ValueError(f"Unsupported boolean value '{item}'")
    return tuple(parsed) if parsed else fallback


def _parse_variant_list(value: str) -> tuple[str, ...]:
    valid = {
        "single",
        "dual_raw",
        "dual_capped",
        "dual_bias",
        "dual_half_visible",
        "dual_visible",
        "dual_relaxed_visible",
        "dual_loose_visible",
        "dual_safe",
    }
    variants: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in valid:
            raise ValueError(f"Unsupported SST sweep variant '{item}'. Valid variants: {sorted(valid)}")
        if item not in variants:
            variants.append(item)
    return tuple(variants) if variants else ("dual_visible",)


def _selected_benchmark_variants(variant: str) -> tuple[str, ...]:
    if variant == "all":
        return (
            "single",
            "dual_raw",
            "dual_capped",
            "dual_bias",
            "dual_half_visible",
            "dual_visible",
            "dual_relaxed_visible",
            "dual_loose_visible",
            "dual_safe",
        )
    if variant == "both":
        return ("single", "dual_capped")
    if variant == "dual":
        return ("dual_capped",)
    return (variant,)


DUAL_VARIANTS = {
    "dual_raw",
    "dual_capped",
    "dual_bias",
    "dual_half_visible",
    "dual_visible",
    "dual_relaxed_visible",
    "dual_loose_visible",
    "dual_safe",
}

DUAL_LEAK_LIMITED_VARIANTS = {
    "dual_bias",
    "dual_half_visible",
    "dual_visible",
    "dual_relaxed_visible",
    "dual_loose_visible",
}


def _visibility_tolerance_for_variant(variant: str, shadow_bias: float) -> float:
    multipliers = {
        "dual_half_visible": 0.5,
        "dual_visible": 1.0,
        "dual_relaxed_visible": 2.0,
        "dual_loose_visible": 4.0,
    }
    return max(0.0, shadow_bias * multipliers.get(variant, 0.0))


def _runtime_sst_fit_profile_value(value: str | None):
    if value is None:
        return None

    from scene import Scene

    profiles = {
        "bias": Scene.SST_FIT_DUAL_BIAS,
        "dual_bias": Scene.SST_FIT_DUAL_BIAS,
        "half": Scene.SST_FIT_DUAL_HALF_VISIBLE,
        "dual_half_visible": Scene.SST_FIT_DUAL_HALF_VISIBLE,
        "visible": Scene.SST_FIT_DUAL_VISIBLE,
        "dual_visible": Scene.SST_FIT_DUAL_VISIBLE,
        "relaxed": Scene.SST_FIT_DUAL_RELAXED_VISIBLE,
        "dual_relaxed_visible": Scene.SST_FIT_DUAL_RELAXED_VISIBLE,
        "loose": Scene.SST_FIT_DUAL_LOOSE_VISIBLE,
        "dual_loose_visible": Scene.SST_FIT_DUAL_LOOSE_VISIBLE,
    }
    return profiles[value]


def _runtime_sst_preset_value(value: str | None):
    from scene import Scene

    presets = {
        None: Scene.SST_PRESET_QUALITY,
        "quality": Scene.SST_PRESET_QUALITY,
        "high-compression": Scene.SST_PRESET_HIGH_COMPRESSION,
        "high_compression": Scene.SST_PRESET_HIGH_COMPRESSION,
        "manual": Scene.SST_PRESET_MANUAL,
    }
    return presets[value]


def _apply_runtime_sst_options(scene, args: argparse.Namespace) -> None:
    scene._apply_sst_preset(_runtime_sst_preset_value(args.sst_preset))

    profile = _runtime_sst_fit_profile_value(args.sst_fit_profile)
    if profile is not None:
        scene._set_sst_fit_profile(profile)

    tile_profile_by_size = {
        64: 0,
        128: 1,
        256: 2,
    }
    tile_profile = tile_profile_by_size.get(max(1, int(args.sst_tile_size)), scene.sst_tile_profile)
    scene._set_sst_encoder_options(
        tile_profile,
        max(1, int(args.sst_min_leaf_size)),
        max(0, int(args.sst_plane_quantization_search_radius)),
        max(0.0, float(args.sst_dual_depth_slack) / max(float(scene.static_shadow_depth_bias), 1e-12)),
    )


def _make_sst_encoder(
    args: argparse.Namespace,
    variant: str,
    shadow_bias: float,
    plane_error: float,
    dual_slack: float,
    quantization_radius: int | None = None,
    tile_size: int | None = None,
    min_leaf_size: int | None = None,
    forced_leaf_error_cap: float | None = None,
    forced_split_bias_fit: bool | None = None,
):
    from static_shadow_tree_encoder import create_sparse_shadow_tree_encoder

    use_dual = variant in DUAL_VARIANTS
    visibility_tolerance = _visibility_tolerance_for_variant(variant, shadow_bias)
    return create_sparse_shadow_tree_encoder(
        backend=args.sst_encoder,
        tile_size=max(1, args.sst_tile_size if tile_size is None else tile_size),
        min_leaf_size=max(1, args.sst_min_leaf_size if min_leaf_size is None else min_leaf_size),
        plane_error_threshold=max(0.0, plane_error),
        constant_epsilon=max(0.0, args.sst_constant_epsilon),
        use_dual_layer=use_dual,
        dual_depth_slack=(None if variant == "dual_raw" else max(0.0, dual_slack)),
        dual_conservative=(variant == "dual_safe"),
        dual_max_leak=(shadow_bias if variant in DUAL_LEAK_LIMITED_VARIANTS else None),
        dual_visibility_tolerance=visibility_tolerance,
        shadow_bias=shadow_bias,
        plane_quantization_search_radius=max(
            0,
            args.sst_plane_quantization_search_radius if quantization_radius is None else quantization_radius,
        ),
        forced_leaf_error_cap=args.sst_forced_leaf_error_cap if forced_leaf_error_cap is None else forced_leaf_error_cap,
        forced_split_bias_fit=args.sst_forced_split_bias_fit if forced_split_bias_fit is None else forced_split_bias_fit,
    )


def _gpu_decompress_compact_sst(device, decompressor, encoded, reference_depth: np.ndarray) -> dict:
    import slangpy as spy

    from sst_decompress import make_sst_buffer

    reference = np.asarray(reference_depth, dtype=np.float32).squeeze()
    height, width = reference.shape
    output = device.create_texture(
        format=spy.Format.r32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="sst_gpu_decompressed_depth",
    )
    words_buffer = make_sst_buffer(device, encoded.compact_words, "sst_benchmark_compact_words")
    roots_buffer = make_sst_buffer(device, encoded.compact_tile_roots, "sst_benchmark_compact_roots")

    dispatch_start = time.perf_counter()
    command_encoder = device.create_command_encoder()
    decompressor.execute(
        command_encoder,
        output,
        words_buffer,
        roots_buffer,
        (width, height),
        encoded.tile_grid,
        encoded.tile_size,
        encoded.stats.max_traversal_steps,
        int(encoded.compact_words.size),
        encoded.stats.branch_10bit_start_level,
    )
    device.submit_command_buffer(command_encoder.finish())
    device.wait()
    dispatch_seconds = time.perf_counter() - dispatch_start

    readback_start = time.perf_counter()
    gpu_decoded = np.asarray(output.to_numpy(), dtype=np.float32).squeeze()
    readback_seconds = time.perf_counter() - readback_start

    encoder = SparseShadowTreeEncoder(
        tile_size=encoded.tile_size,
        min_leaf_size=encoded.min_leaf_size,
        shadow_bias=float(encoded.stats.shadow_bias),
    )
    encoder._branch_10bit_start_level = int(encoded.stats.branch_10bit_start_level)
    cpu_compact = encoder.decode_compact(reference.shape, encoded.compact_words, encoded.compact_tile_roots, encoded.tile_grid)

    gpu_cpu_delta = np.abs(gpu_decoded - cpu_compact)
    gpu_source_delta = gpu_decoded - reference
    gpu_source_abs = np.abs(gpu_source_delta)
    return {
        "gpu_decompress_dispatch_seconds": float(dispatch_seconds),
        "gpu_decompress_readback_seconds": float(readback_seconds),
        "gpu_decompress_valid": bool(np.max(gpu_cpu_delta) <= 2e-6) if gpu_cpu_delta.size else True,
        "gpu_decompress_vs_cpu_max_delta_percent": float(np.max(gpu_cpu_delta) * 100.0) if gpu_cpu_delta.size else 0.0,
        "gpu_decompress_vs_cpu_mean_delta_percent": float(np.mean(gpu_cpu_delta) * 100.0) if gpu_cpu_delta.size else 0.0,
        "gpu_decompress_vs_source_mean_error_percent": float(np.mean(gpu_source_abs) * 100.0) if gpu_source_abs.size else 0.0,
        "gpu_decompress_vs_source_rmse_percent": float(np.sqrt(np.mean(gpu_source_delta * gpu_source_delta)) * 100.0) if gpu_source_delta.size else 0.0,
        "gpu_decompress_vs_source_max_error_percent": float(np.max(gpu_source_abs) * 100.0) if gpu_source_abs.size else 0.0,
    }


def _encode_sst_variant(
    args: argparse.Namespace,
    front_depth,
    second_depth,
    variant: str,
    shadow_bias: float,
    plane_error: float,
    dual_slack: float,
    quantization_radius: int | None = None,
    tile_size: int | None = None,
    min_leaf_size: int | None = None,
    forced_leaf_error_cap: float | None = None,
    forced_split_bias_fit: bool | None = None,
    device=None,
    gpu_decompressor=None,
) -> dict:
    use_dual = variant in DUAL_VARIANTS
    radius = max(0, args.sst_plane_quantization_search_radius if quantization_radius is None else quantization_radius)
    effective_tile_size = max(1, args.sst_tile_size if tile_size is None else tile_size)
    effective_min_leaf_size = max(1, args.sst_min_leaf_size if min_leaf_size is None else min_leaf_size)
    encoder = _make_sst_encoder(
        args,
        variant,
        shadow_bias,
        plane_error,
        dual_slack,
        radius,
        effective_tile_size,
        effective_min_leaf_size,
        forced_leaf_error_cap,
        forced_split_bias_fit,
    )
    encode_start = time.perf_counter()
    encoded = encoder.encode(front_depth, second_depth if use_dual else None)
    encode_seconds = time.perf_counter() - encode_start
    stats = asdict(encoded.stats)
    timing = {
        "encode": encode_seconds,
    }
    if gpu_decompressor is not None and device is not None:
        gpu_stats = _gpu_decompress_compact_sst(device, gpu_decompressor, encoded, front_depth)
        stats.update(gpu_stats)
        timing["gpu_decompress_dispatch"] = gpu_stats["gpu_decompress_dispatch_seconds"]
        timing["gpu_decompress_readback"] = gpu_stats["gpu_decompress_readback_seconds"]
    return {
        "variant": variant,
        "plane_error_threshold": max(0.0, plane_error),
        "dual_depth_slack": None if variant == "dual_raw" else (max(0.0, dual_slack) if use_dual else None),
        "plane_quantization_search_radius": radius,
        "tile_size": effective_tile_size,
        "min_leaf_size": effective_min_leaf_size,
        "forced_leaf_error_cap": args.sst_forced_leaf_error_cap if forced_leaf_error_cap is None else forced_leaf_error_cap,
        "forced_split_bias_fit": args.sst_forced_split_bias_fit if forced_split_bias_fit is None else forced_split_bias_fit,
        "stats": stats,
        "timing_seconds": timing,
    }


def _is_pareto_better_or_equal(left: dict, right: dict) -> bool:
    left_stats = left["stats"]
    right_stats = right["stats"]
    return (
        left_stats["packed_compression_ratio"] >= right_stats["packed_compression_ratio"]
        and left_stats["packed_mean_error_percent"] <= right_stats["packed_mean_error_percent"]
        and left_stats.get("packed_visibility_mismatch_percent", 0.0) <= right_stats.get("packed_visibility_mismatch_percent", 0.0)
        and left_stats["packed_leak_over_full_bias_percent"] <= right_stats["packed_leak_over_full_bias_percent"]
        and left_stats["packed_conservative_over_full_bias_percent"] <= right_stats["packed_conservative_over_full_bias_percent"]
    )


def _compute_sweep_pareto(results: list[dict]) -> list[dict]:
    pareto: list[dict] = []
    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            if _is_pareto_better_or_equal(other, candidate):
                other_stats = other["stats"]
                candidate_stats = candidate["stats"]
                strictly_better = (
                    other_stats["packed_compression_ratio"] > candidate_stats["packed_compression_ratio"]
                    or other_stats["packed_mean_error_percent"] < candidate_stats["packed_mean_error_percent"]
                    or other_stats.get("packed_visibility_mismatch_percent", 0.0) < candidate_stats.get("packed_visibility_mismatch_percent", 0.0)
                    or other_stats["packed_leak_over_full_bias_percent"] < candidate_stats["packed_leak_over_full_bias_percent"]
                    or other_stats["packed_conservative_over_full_bias_percent"] < candidate_stats["packed_conservative_over_full_bias_percent"]
                )
                if strictly_better:
                    dominated = True
                    break
        if not dominated:
            pareto.append(candidate)

    best_by_metric: dict[tuple, dict] = {}
    for candidate in pareto:
        stats = candidate["stats"]
        metric_key = (
            round(float(stats["packed_compression_ratio"]), 6),
            round(float(stats["packed_mean_error_percent"]), 6),
            round(float(stats["packed_rmse_error_percent"]), 6),
            round(float(stats["packed_max_error_percent"]), 6),
            round(float(stats.get("packed_visibility_mismatch_percent", 0.0)), 6),
            round(float(stats["packed_leak_over_full_bias_percent"]), 6),
            round(float(stats["packed_conservative_over_full_bias_percent"]), 6),
        )
        dual_slack = candidate["dual_depth_slack"]
        preference = (
            candidate["variant"],
            int(candidate.get("tile_size", 0)),
            int(candidate.get("min_leaf_size", 0)),
            -1.0 if candidate.get("forced_leaf_error_cap") is None else float(candidate.get("forced_leaf_error_cap")),
            bool(candidate.get("forced_split_bias_fit", False)),
            float(candidate["plane_error_threshold"]),
            -1.0 if dual_slack is None else float(dual_slack),
            int(candidate.get("plane_quantization_search_radius", 0)),
        )
        previous = best_by_metric.get(metric_key)
        if previous is None:
            best_by_metric[metric_key] = candidate
            continue
        previous_slack = previous["dual_depth_slack"]
        previous_preference = (
            previous["variant"],
            int(previous.get("tile_size", 0)),
            int(previous.get("min_leaf_size", 0)),
            -1.0 if previous.get("forced_leaf_error_cap") is None else float(previous.get("forced_leaf_error_cap")),
            bool(previous.get("forced_split_bias_fit", False)),
            float(previous["plane_error_threshold"]),
            -1.0 if previous_slack is None else float(previous_slack),
            int(previous.get("plane_quantization_search_radius", 0)),
        )
        if preference < previous_preference:
            best_by_metric[metric_key] = candidate

    return sorted(
        best_by_metric.values(),
        key=lambda item: (
            item["stats"]["packed_leak_over_full_bias_percent"],
            item["stats"]["packed_conservative_over_full_bias_percent"],
            item["stats"].get("packed_visibility_mismatch_percent", 0.0),
            -item["stats"]["packed_compression_ratio"],
        ),
    )


def _summarize_sweep_candidate(candidate: dict, label: str) -> dict:
    stats = candidate["stats"]
    return {
        "label": label,
        "variant": candidate["variant"],
        "tile_size": candidate.get("tile_size", candidate["stats"].get("tile_size", 0)),
        "min_leaf_size": candidate.get("min_leaf_size", 0),
        "forced_leaf_error_cap": candidate.get("forced_leaf_error_cap"),
        "forced_split_bias_fit": candidate.get("forced_split_bias_fit", False),
        "plane_error_threshold": candidate["plane_error_threshold"],
        "dual_depth_slack": candidate["dual_depth_slack"],
        "plane_quantization_search_radius": candidate.get("plane_quantization_search_radius", 0),
        "packed_compression_ratio": stats["packed_compression_ratio"],
        "packed_bits_per_texel": _bits_per_texel(stats, "packed_encoded_bytes"),
        "packed_encoded_bytes": int(stats.get("packed_encoded_bytes", 0)),
        "packed_decompressed_working_set_ratio": _packed_decompressed_working_set_ratio(stats),
        "packed_decompressed_working_set_bytes": _packed_decompressed_working_set_bytes(stats),
        "fixed64_compression_ratio": stats["fixed64_compression_ratio"],
        "packed_mean_error_percent": stats["packed_mean_error_percent"],
        "packed_rmse_error_percent": stats["packed_rmse_error_percent"],
        "packed_visibility_mismatch_percent": stats.get("packed_visibility_mismatch_percent", 0.0),
        "packed_false_lit_percent": stats.get("packed_false_lit_percent", 0.0),
        "packed_false_shadow_percent": stats.get("packed_false_shadow_percent", 0.0),
        "packed_depth_pcf3_mae_percent": stats.get("packed_depth_pcf3_mae_percent"),
        "packed_depth_pcf3_max_error_percent": stats.get("packed_depth_pcf3_max_error_percent"),
        "packed_hard_vs_depth_pcf3_mae_percent": stats.get("packed_hard_vs_depth_pcf3_mae_percent"),
        "packed_leak_over_full_bias_percent": stats["packed_leak_over_full_bias_percent"],
        "packed_conservative_over_full_bias_percent": stats["packed_conservative_over_full_bias_percent"],
        "packed_abs_error_within_1_bias_percent": _bias_cdf_value(stats, 1.0),
        "packed_abs_error_p99_percent": _percentile_value(stats, "p99"),
        "packed_morton_decode_valid": stats.get("packed_morton_decode_valid", False),
    }


def _best_sweep_candidate(results: list[dict], label: str, predicate) -> dict | None:
    candidates = [result for result in results if predicate(result["stats"])]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda result: (
            result["stats"]["packed_compression_ratio"],
            -result["stats"].get("packed_visibility_mismatch_percent", 0.0),
            -result["stats"].get("packed_false_lit_percent", 0.0),
        ),
    )
    return _summarize_sweep_candidate(best, label)


def _compute_sweep_recommendations(results: list[dict]) -> list[dict]:
    recommendation_specs = (
        (
            "max_compression_no_false_lit_no_gt_bias_error",
            lambda stats: (
                stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
                and stats["packed_conservative_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_vis_mismatch_le_1_percent",
            lambda stats: (
                stats.get("packed_visibility_mismatch_percent", 0.0) <= 1.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_vis_mismatch_le_2_5_percent",
            lambda stats: (
                stats.get("packed_visibility_mismatch_percent", 0.0) <= 2.5
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_vis_mismatch_le_5_percent",
            lambda stats: (
                stats.get("packed_visibility_mismatch_percent", 0.0) <= 5.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_false_shadow_le_5_percent",
            lambda stats: (
                stats.get("packed_false_shadow_percent", 0.0) <= 5.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_shadow_over_1_bias_le_5_percent",
            lambda stats: (
                stats.get("packed_conservative_over_full_bias_percent", 0.0) <= 5.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_abs_error_within_1_bias_ge_95_percent",
            lambda stats: (
                _bias_cdf_value(stats, 1.0) >= 95.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_mean_error_le_0_1_percent",
            lambda stats: (
                stats["packed_mean_error_percent"] <= 0.1
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_false_lit_le_1_percent",
            lambda stats: (
                stats.get("packed_false_lit_percent", 0.0) <= 1.0
                and stats["packed_leak_over_full_bias_percent"] <= 5.0
            ),
        ),
        (
            "max_compression_pcf3_mae_le_1_percent",
            lambda stats: (
                stats.get("packed_depth_pcf3_mae_percent", 1e9) <= 1.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_pcf3_mae_le_2_percent",
            lambda stats: (
                stats.get("packed_depth_pcf3_mae_percent", 1e9) <= 2.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_pcf3_mae_le_3_percent",
            lambda stats: (
                stats.get("packed_depth_pcf3_mae_percent", 1e9) <= 3.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
        (
            "max_compression_pcf3_mae_le_5_percent",
            lambda stats: (
                stats.get("packed_depth_pcf3_mae_percent", 1e9) <= 5.0
                and stats.get("packed_false_lit_percent", 0.0) <= 1e-7
                and stats["packed_leak_over_full_bias_percent"] <= 1e-7
            ),
        ),
    )
    recommendations: list[dict] = []
    seen_labels: set[str] = set()
    for label, predicate in recommendation_specs:
        recommendation = _best_sweep_candidate(results, label, predicate)
        if recommendation is not None and recommendation["label"] not in seen_labels:
            seen_labels.add(recommendation["label"])
            recommendations.append(recommendation)
    return recommendations


def _markdown_escape(value) -> str:
    return str(value).replace("|", "\\|")


def _format_percent(value: float) -> str:
    return f"{float(value):.4f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.2f}x"


def _format_bpt(value: float) -> str:
    return f"{float(value):.3f}"


def _save_png(path: Path, array: np.ndarray) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return True


def _save_depth_debug_map(output_dir: Path, stem: str, depth: np.ndarray) -> bool:
    array = np.asarray(depth, dtype=np.float32).squeeze()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", array)
    preview = np.clip(array, 0.0, 1.0)
    return _save_png(output_dir / f"{stem}.png", np.uint16(np.round(preview * 65535.0)))


def _save_abs_error_bias_map(output_dir: Path, stem: str, error: np.ndarray, shadow_bias: float) -> bool:
    array = np.asarray(error, dtype=np.float32).squeeze()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", array)
    scale = max(float(shadow_bias) * 4.0, 1e-12)
    preview = np.clip(np.abs(array) / scale, 0.0, 1.0)
    return _save_png(output_dir / f"{stem}.png", np.uint8(np.round(preview * 255.0)))


def _save_signed_error_rgb_map(output_dir: Path, stem: str, error: np.ndarray, shadow_bias: float) -> bool:
    array = np.asarray(error, dtype=np.float32).squeeze()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", array)
    scale = max(float(shadow_bias) * 2.0, 1e-12)
    leak = np.clip(np.maximum(array, 0.0) / scale, 0.0, 1.0)
    shadow = np.clip(np.maximum(-array, 0.0) / scale, 0.0, 1.0)
    neutral = np.clip(1.0 - np.maximum(leak, shadow), 0.0, 1.0) * 0.15
    rgb = np.stack([leak + neutral, neutral, shadow + neutral], axis=-1)
    return _save_png(output_dir / f"{stem}.png", np.uint8(np.round(np.clip(rgb, 0.0, 1.0) * 255.0)))


def _save_visibility_mismatch_maps(
    output_dir: Path,
    stem: str,
    reference_depth: np.ndarray,
    encoded_depth: np.ndarray,
    shadow_bias: float,
    offsets_in_bias: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(reference_depth, dtype=np.float32).squeeze()
    encoded = np.asarray(encoded_depth, dtype=np.float32).squeeze()
    details: list[dict] = []
    for offset in offsets_in_bias:
        receiver_depth = reference + float(shadow_bias) * float(offset)
        reference_visible = receiver_depth <= reference + float(shadow_bias)
        encoded_visible = receiver_depth <= encoded + float(shadow_bias)
        false_lit = (~reference_visible) & encoded_visible
        false_shadow = reference_visible & (~encoded_visible)
        mask = np.zeros((*reference.shape, 3), dtype=np.uint8)
        mask[..., 0] = np.uint8(false_lit) * 255
        mask[..., 2] = np.uint8(false_shadow) * 255
        offset_name = str(offset).replace(".", "p")
        np.save(output_dir / f"{stem}_visibility_o{offset_name}.npy", mask)
        _save_png(output_dir / f"{stem}_visibility_o{offset_name}.png", mask)
        details.append(
            {
                "offset_in_bias": float(offset),
                "false_lit_percent": float(np.mean(false_lit) * 100.0),
                "false_shadow_percent": float(np.mean(false_shadow) * 100.0),
                "mismatch_percent": float(np.mean(false_lit | false_shadow) * 100.0),
            }
        )
    return details


def _export_sst_debug_maps(
    args: argparse.Namespace,
    output_dir: Path,
    front_depth: np.ndarray,
    second_depth: np.ndarray,
    shadow_bias: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    front = np.asarray(front_depth, dtype=np.float32).squeeze()
    second = np.asarray(second_depth, dtype=np.float32).squeeze()
    png_enabled = _save_depth_debug_map(output_dir, "static_shadow_depth", front)
    png_enabled = _save_depth_debug_map(output_dir, "static_shadow_second_depth", second) or png_enabled

    variants = (
        _parse_variant_list(args.sst_debug_variants)
        if args.sst_debug_variants
        else _selected_benchmark_variants(args.sst_benchmark_variant)
    )
    metadata = {
        "shadow_bias": float(shadow_bias),
        "variants": {},
        "png_enabled": bool(png_enabled),
    }

    for variant in variants:
        use_dual = variant in DUAL_VARIANTS
        encoder = _make_sst_encoder(
            args,
            variant,
            shadow_bias,
            max(0.0, args.sst_plane_error_threshold),
            max(0.0, args.sst_dual_depth_slack),
        )
        encoded = encoder.encode(front, second if use_dual else None)
        decoded = encoder.decode_compact(front.shape, encoded.compact_words, encoded.compact_tile_roots, encoded.tile_grid)
        error = decoded - front
        variant_stem = variant.replace(" ", "_")
        _save_depth_debug_map(output_dir, f"{variant_stem}_decoded_depth", decoded)
        _save_abs_error_bias_map(output_dir, f"{variant_stem}_abs_error_bias4", error, shadow_bias)
        _save_signed_error_rgb_map(output_dir, f"{variant_stem}_signed_error", error, shadow_bias)
        visibility_details = _save_visibility_mismatch_maps(
            output_dir,
            variant_stem,
            front,
            decoded,
            shadow_bias,
        )
        metadata["variants"][variant] = {
            "stats": asdict(encoded.stats),
            "visibility_debug": visibility_details,
        }

    (output_dir / "sst_debug_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _bits_per_texel(stats: dict, byte_key: str) -> float:
    pixel_count = max(int(stats.get("width", 0)) * int(stats.get("height", 0)), 1)
    return float(stats[byte_key]) * 8.0 / float(pixel_count)


def _tile_root_bytes(stats: dict) -> int:
    return int(stats.get("tile_count", 0)) * 4


def _packed_stream_bytes(stats: dict) -> int:
    return max(int(stats.get("packed_encoded_bytes", 0)) - _tile_root_bytes(stats), 0)


def _fixed64_node_bytes(stats: dict) -> int:
    return int(stats.get("node_count", 0)) * 8


def _decompressed_depth_bytes(stats: dict) -> int:
    return int(stats.get("decompressed_depth_bytes", stats.get("original_bytes", 0)))


def _packed_decompressed_working_set_bytes(stats: dict) -> int:
    return int(
        stats.get(
            "packed_decompressed_working_set_bytes",
            int(stats.get("packed_encoded_bytes", 0)) + _decompressed_depth_bytes(stats),
        )
    )


def _packed_decompressed_working_set_ratio(stats: dict) -> float:
    if "packed_decompressed_working_set_ratio" in stats:
        return float(stats["packed_decompressed_working_set_ratio"])
    working_set_bytes = _packed_decompressed_working_set_bytes(stats)
    return float(stats.get("original_bytes", 0)) / float(working_set_bytes) if working_set_bytes > 0 else 0.0


def _bytes_to_bpt(stats: dict, byte_count: int) -> float:
    pixel_count = max(int(stats.get("width", 0)) * int(stats.get("height", 0)), 1)
    return float(byte_count) * 8.0 / float(pixel_count)


def _compact_word_breakdown(stats: dict) -> tuple[int, int, int, int]:
    branch_words = int(stats.get("compact_branch_words", int(stats.get("branch_node_count", 0)) * 2))
    plane30_words = int(stats.get("compact_30bit_plane_words", int(stats.get("uniform_plane_node_count", 0))))
    plane_nodes = int(stats.get("plane_node_count", 0))
    plane30_nodes = int(stats.get("uniform_plane_node_count", 0))
    plane62_words = int(stats.get("compact_62bit_plane_words", max(plane_nodes - plane30_nodes, 0) * 2))
    total_words = int(stats.get("compact_node_words", branch_words + plane30_words + plane62_words))
    return branch_words, plane30_words, plane62_words, total_words


def _visibility_probe_detail(stats: dict, offset_in_bias: float) -> dict:
    for detail in stats.get("packed_visibility_probe_details", ()):
        if abs(float(detail.get("offset_in_bias", -999.0)) - float(offset_in_bias)) < 1e-5:
            return detail
    return {}


def _visibility_probe_mismatch(stats: dict, offset_in_bias: float) -> float:
    return float(_visibility_probe_detail(stats, offset_in_bias).get("mismatch_percent", 0.0))


def _percentile_value(stats: dict, key: str) -> float:
    return float(stats.get("packed_abs_error_percentiles", {}).get(key, 0.0))


def _bias_cdf_value(stats: dict, threshold_in_bias: float) -> float:
    for item in stats.get("packed_abs_error_bias_cdf", ()):
        if abs(float(item.get("threshold_in_bias", -999.0)) - float(threshold_in_bias)) < 1e-5:
            return float(item.get("within_percent", 0.0))
    if abs(float(threshold_in_bias) - 1.0) < 1e-5:
        return max(
            0.0,
            100.0
            - float(stats.get("packed_leak_over_full_bias_percent", 0.0))
            - float(stats.get("packed_conservative_over_full_bias_percent", 0.0)),
        )
    if abs(float(threshold_in_bias) - 0.5) < 1e-5:
        return max(
            0.0,
            100.0
            - float(stats.get("packed_leak_over_half_bias_percent", 0.0))
            - float(stats.get("packed_conservative_over_half_bias_percent", 0.0)),
        )
    return 0.0


def _error_distribution_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    if not stats.get("packed_abs_error_bias_cdf") or not stats.get("packed_abs_error_percentiles"):
        return None
    return (
        f"| {int(report.get('shadow_resolution', 0))} "
        f"| {_markdown_escape(variant)} "
        f"| {_format_percent(_percentile_value(stats, 'p95'))} "
        f"| {_format_percent(_percentile_value(stats, 'p99'))} "
        f"| {_format_percent(_percentile_value(stats, 'p999'))} "
        f"| {_format_percent(_bias_cdf_value(stats, 0.5))} "
        f"| {_format_percent(_bias_cdf_value(stats, 1.0))} "
        f"| {_format_percent(_bias_cdf_value(stats, 2.0))} "
        f"| {_format_percent(stats.get('packed_leak_over_full_bias_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_conservative_over_full_bias_percent', 0.0))} |"
    )


def _dual_layer_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    if not stats.get("dual_layer", False):
        return None
    if "dual_second_hit_percent" not in stats:
        return None
    return (
        f"| {int(report.get('shadow_resolution', 0))} "
        f"| {_markdown_escape(variant)} "
        f"| {_format_percent(stats.get('dual_second_hit_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_raw_gap_mean_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_raw_gap_p95_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_raw_gap_max_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_capped_gap_mean_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_capped_gap_p95_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_capped_gap_max_percent', 0.0))} "
        f"| {_format_percent(stats.get('dual_slack_clamped_percent', 0.0))} |"
    )


def _variant_markdown_row(name: str, stats: dict) -> str:
    return (
        f"| {_markdown_escape(name)} "
        f"| {_format_ratio(stats['packed_compression_ratio'])} "
        f"| {_format_bpt(_bits_per_texel(stats, 'packed_encoded_bytes'))} "
        f"| {_format_ratio(stats['fixed64_compression_ratio'])} "
        f"| {_format_bpt(_bits_per_texel(stats, 'fixed64_encoded_bytes'))} "
        f"| {int(stats['node_count'])} "
        f"| {int(stats['uniform_plane_node_count'])} "
        f"| {_format_percent(stats['packed_mean_error_percent'])} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_shadow_percent', 0.0))} "
        f"| {_format_percent(stats['packed_leak_over_full_bias_percent'])} "
        f"| {_markdown_escape(stats.get('packed_morton_decode_valid', False))} "
        f"| {_markdown_escape(stats.get('compact_fixed64_decode_valid', False))} |"
    )


def _candidate_tile_size(candidate: dict, default_tile_size: int = 0) -> int:
    return int(candidate.get("tile_size", default_tile_size))


def _candidate_min_leaf_size(candidate: dict, default_min_leaf_size: int = 0) -> int:
    return int(candidate.get("min_leaf_size", default_min_leaf_size))


def _format_optional_float(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6f}"


def _format_bool(value) -> str:
    return "yes" if bool(value) else "no"


def _format_optional_percent(value) -> str:
    if value is None:
        return "n/a"
    return _format_percent(float(value))


def _recommendation_packed_bpt(recommendation: dict) -> float:
    if "packed_bits_per_texel" in recommendation:
        return float(recommendation["packed_bits_per_texel"])
    ratio = float(recommendation.get("packed_compression_ratio", 0.0))
    return 32.0 / ratio if ratio > 0.0 else 0.0


def _recommendation_decomp_ratio(recommendation: dict) -> float:
    if "packed_decompressed_working_set_ratio" in recommendation:
        return float(recommendation["packed_decompressed_working_set_ratio"])
    ratio = float(recommendation.get("packed_compression_ratio", 0.0))
    return 1.0 / (1.0 + (1.0 / ratio)) if ratio > 0.0 else 0.0


def _sweep_candidate_markdown_row(candidate: dict, default_tile_size: int = 0, default_min_leaf_size: int = 0) -> str:
    stats = candidate["stats"]
    return (
        f"| {_markdown_escape(candidate['variant'])} "
        f"| {_candidate_tile_size(candidate, default_tile_size)} "
        f"| {_candidate_min_leaf_size(candidate, default_min_leaf_size)} "
        f"| {_format_optional_float(candidate.get('forced_leaf_error_cap'))} "
        f"| {_format_bool(candidate.get('forced_split_bias_fit', False))} "
        f"| {float(candidate['plane_error_threshold']):.6f} "
        f"| {_markdown_escape(candidate['dual_depth_slack'])} "
        f"| {int(candidate.get('plane_quantization_search_radius', 0))} "
        f"| {_format_ratio(stats['packed_compression_ratio'])} "
        f"| {_format_percent(stats['packed_mean_error_percent'])} "
        f"| {_format_percent(stats.get('forced_leaf_pixel_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_shadow_percent', 0.0))} "
        f"| {_format_percent(stats['packed_leak_over_full_bias_percent'])} |"
    )


def _sweep_probe_markdown_row(candidate: dict, default_tile_size: int = 0, default_min_leaf_size: int = 0) -> str:
    stats = candidate["stats"]
    return (
        f"| {_markdown_escape(candidate['variant'])} "
        f"| {_candidate_tile_size(candidate, default_tile_size)} "
        f"| {_candidate_min_leaf_size(candidate, default_min_leaf_size)} "
        f"| {_format_optional_float(candidate.get('forced_leaf_error_cap'))} "
        f"| {_format_bool(candidate.get('forced_split_bias_fit', False))} "
        f"| {int(candidate.get('plane_quantization_search_radius', 0))} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 0.0))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 0.5))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 1.0))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 2.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_shadow_percent', 0.0))} |"
    )


def _recommendation_markdown_row(recommendation: dict, default_tile_size: int = 0, default_min_leaf_size: int = 0) -> str:
    return (
        f"| {_markdown_escape(recommendation['label'])} "
        f"| {_markdown_escape(recommendation['variant'])} "
        f"| {_candidate_tile_size(recommendation, default_tile_size)} "
        f"| {_candidate_min_leaf_size(recommendation, default_min_leaf_size)} "
        f"| {_format_optional_float(recommendation.get('forced_leaf_error_cap'))} "
        f"| {_format_bool(recommendation.get('forced_split_bias_fit', False))} "
        f"| {float(recommendation['plane_error_threshold']):.6f} "
        f"| {_markdown_escape(recommendation['dual_depth_slack'])} "
        f"| {int(recommendation.get('plane_quantization_search_radius', 0))} "
        f"| {_format_ratio(recommendation['packed_compression_ratio'])} "
        f"| {_format_bpt(_recommendation_packed_bpt(recommendation))} "
        f"| {_format_ratio(_recommendation_decomp_ratio(recommendation))} "
        f"| {_format_percent(recommendation['packed_visibility_mismatch_percent'])} "
        f"| {_format_optional_percent(recommendation.get('packed_depth_pcf3_mae_percent'))} "
        f"| {_format_percent(recommendation['packed_false_lit_percent'])} "
        f"| {_format_percent(recommendation['packed_false_shadow_percent'])} "
        f"| {_format_percent(recommendation['packed_leak_over_full_bias_percent'])} "
        f"| {_format_percent(recommendation.get('packed_conservative_over_full_bias_percent', 0.0))} "
        f"| {_format_percent(recommendation.get('packed_abs_error_within_1_bias_percent', 0.0))} |"
    )


def build_sst_markdown_report(report: dict) -> str:
    lines: list[str] = [
        "# SST Benchmark Report",
        "",
        f"- Scene: `{report['scene']}`",
        f"- Shadow resolution: `{report['shadow_resolution']}`",
        f"- Tile size: `{report['settings']['tile_size']}`",
        f"- Min leaf size: `{report['settings']['min_leaf_size']}`",
        f"- Plane quantization search radius: `{report['settings'].get('plane_quantization_search_radius', 0)}`",
        f"- Shadow bias: `{report['settings']['shadow_bias']}`",
        f"- Bake time: `{report['timing_seconds']['bake']:.3f}s`",
        f"- Readback time: `{report['timing_seconds']['readback']:.3f}s`",
        "",
        "## Variants",
        "",
        "| Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    if report.get("debug_output"):
        debug_output = report["debug_output"]
        lines[11:11] = [
            f"- Debug maps: `{debug_output['directory']}`",
            f"- Debug variants: `{', '.join(debug_output.get('variants', []))}`",
        ]
    for name, variant_report in report.get("variants", {}).items():
        lines.append(_variant_markdown_row(name, variant_report["stats"]))

    if any(
        variant_report.get("stats", {}).get("packed_abs_error_bias_cdf")
        for variant_report in report.get("variants", {}).values()
    ):
        lines.extend(
            [
                "",
                "## Error Distribution",
                "",
                "| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in report.get("variants", {}):
            row = _error_distribution_markdown_row(report, name)
            if row is not None:
                lines.append(row)

    if any(
        "packed_depth_pcf3_mae_percent" in variant_report.get("stats", {})
        for variant_report in report.get("variants", {}).values()
    ):
        lines.extend(
            [
                "",
                "## Depth Texture PCF3 Delta",
                "",
                "This estimates the shader-side `Compact SST PCF3` mode against the existing `Depth Texture` PCF3 path.",
                "",
                "| Resolution | Variant | SST PCF3 vs Depth PCF3 MAE | SST PCF3 max | SST hard vs Depth PCF3 MAE | SST hard max | @0B hard MAE | @1B hard MAE |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in report.get("variants", {}):
            row = _pcf3_delta_markdown_row(report, name)
            if row is not None:
                lines.append(row)

    if any(
        "gpu_decompress_valid" in variant_report.get("stats", {})
        for variant_report in report.get("variants", {}).values()
    ):
        lines.extend(
            [
                "",
                "## GPU Decompression",
                "",
                "This validates the paper-style Compact SST stream by decompressing it on the GPU back to an `r32_float` depth texture.",
                "",
                "| Resolution | Variant | GPU == CPU | GPU vs CPU mean | GPU vs CPU max | GPU vs source mean | GPU vs source RMSE | GPU vs source max | Dispatch | Readback |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in report.get("variants", {}):
            row = _gpu_decompress_markdown_row(report, name)
            if row is not None:
                lines.append(row)

    if any(variant_report.get("stats", {}).get("dual_layer", False) for variant_report in report.get("variants", {}).values()):
        lines.extend(
            [
                "",
                "## Dual Layer Utilization",
                "",
                "| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in report.get("variants", {}):
            row = _dual_layer_markdown_row(report, name)
            if row is not None:
                lines.append(row)

    lines.extend(
        [
            "",
            "## Memory Breakdown",
            "",
            "Decomp working set counts the persistent Compact SST stream plus one full-resolution `r32_float` decompressed depth texture.",
            "",
            "| Resolution | Variant | Packed bytes | Packed ratio | Packed stream bpt | Tile roots bpt | Packed total bpt | Decomp texture bytes | Packed+decomp bytes | Packed+decomp bpt | Packed+decomp ratio | Fixed64 bytes | Fixed64 ratio |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in report.get("variants", {}):
        row = _memory_breakdown_markdown_row(report, name)
        if row is not None:
            lines.append(row)

    recommendations = report.get("sweep_recommendations", [])
    if recommendations:
        lines.extend(
            [
                "",
                "## Recommendations",
                "",
                "| Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Packed bpt | Packed+decomp ratio | Vis mismatch | PCF3 MAE | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for recommendation in recommendations:
            lines.append(
                _recommendation_markdown_row(
                    recommendation,
                    int(report["settings"].get("tile_size", 0)),
                    int(report["settings"].get("min_leaf_size", 0)),
                )
            )

    pareto = report.get("sweep_pareto", [])
    if pareto:
        lines.extend(
            [
                "",
                "## Pareto Front",
                "",
                "| Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in pareto[:16]:
            lines.append(
                _sweep_candidate_markdown_row(
                    candidate,
                    int(report["settings"].get("tile_size", 0)),
                    int(report["settings"].get("min_leaf_size", 0)),
                )
            )

        if any(candidate.get("stats", {}).get("packed_visibility_probe_details") for candidate in pareto[:16]):
            lines.extend(
                [
                    "",
                    "## Pareto Visibility Probes",
                    "",
                    "| Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for candidate in pareto[:16]:
                lines.append(
                    _sweep_probe_markdown_row(
                        candidate,
                        int(report["settings"].get("tile_size", 0)),
                        int(report["settings"].get("min_leaf_size", 0)),
                    )
                )

    return "\n".join(lines) + "\n"


def _parse_path_list(value: str) -> list[Path]:
    paths: list[Path] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            paths.append(Path(item))
    return paths


def _load_sst_reports(paths: list[Path]) -> list[dict]:
    reports: list[dict] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("sweep_results"):
            settings = report.get("settings", {})
            for result in report["sweep_results"]:
                result.setdefault("tile_size", settings.get("tile_size", 0))
                result.setdefault("min_leaf_size", settings.get("min_leaf_size", 0))
            report["sweep_recommendations"] = _compute_sweep_recommendations(report["sweep_results"])
            report["sweep_pareto"] = _compute_sweep_pareto(report["sweep_results"])
        report["_source_path"] = str(path)
        reports.append(report)
    return reports


def _comparison_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {_format_ratio(stats['packed_compression_ratio'])} "
        f"| {_format_bpt(_bits_per_texel(stats, 'packed_encoded_bytes'))} "
        f"| {_format_ratio(stats['fixed64_compression_ratio'])} "
        f"| {_format_bpt(_bits_per_texel(stats, 'fixed64_encoded_bytes'))} "
        f"| {int(stats['node_count'])} "
        f"| {int(stats['uniform_plane_node_count'])} "
        f"| {_format_percent(stats['packed_mean_error_percent'])} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_shadow_percent', 0.0))} "
        f"| {_format_percent(stats['packed_leak_over_full_bias_percent'])} "
        f"| {_markdown_escape(stats.get('packed_morton_decode_valid', False))} "
        f"| {_markdown_escape(stats.get('compact_fixed64_decode_valid', False))} |"
    )


def _profile_trend_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {_format_ratio(stats['packed_compression_ratio'])} "
        f"| {_format_bpt(_bits_per_texel(stats, 'packed_encoded_bytes'))} "
        f"| {_format_percent(stats['packed_mean_error_percent'])} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats['packed_leak_over_full_bias_percent'])} |"
    )


def _memory_breakdown_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    root_bytes = _tile_root_bytes(stats)
    packed_stream_bytes = _packed_stream_bytes(stats)
    packed_total_bytes = int(stats.get("packed_encoded_bytes", packed_stream_bytes + root_bytes))
    decompressed_bytes = _decompressed_depth_bytes(stats)
    decomp_working_set_bytes = _packed_decompressed_working_set_bytes(stats)
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {packed_total_bytes:,} "
        f"| {_format_ratio(stats['packed_compression_ratio'])} "
        f"| {_format_bpt(_bytes_to_bpt(stats, packed_stream_bytes))} "
        f"| {_format_bpt(_bytes_to_bpt(stats, root_bytes))} "
        f"| {_format_bpt(_bits_per_texel(stats, 'packed_encoded_bytes'))} "
        f"| {decompressed_bytes:,} "
        f"| {decomp_working_set_bytes:,} "
        f"| {_format_bpt(_bytes_to_bpt(stats, decomp_working_set_bytes))} "
        f"| {_format_ratio(_packed_decompressed_working_set_ratio(stats))} "
        f"| {int(stats.get('fixed64_encoded_bytes', 0)):,} "
        f"| {_format_ratio(stats['fixed64_compression_ratio'])} |"
    )


def _paper_stream_breakdown_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    branch_words, plane30_words, plane62_words, total_words = _compact_word_breakdown(stats)
    root_bytes = int(stats.get("compact_tile_root_bytes", _tile_root_bytes(stats)))
    total_bytes = total_words * 4 + root_bytes
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {total_words} "
        f"| {_format_bpt(_bytes_to_bpt(stats, branch_words * 4))} "
        f"| {_format_bpt(_bytes_to_bpt(stats, plane30_words * 4))} "
        f"| {_format_bpt(_bytes_to_bpt(stats, plane62_words * 4))} "
        f"| {_format_bpt(_bytes_to_bpt(stats, root_bytes))} "
        f"| {_format_bpt(_bytes_to_bpt(stats, total_bytes))} |"
    )


def _branch_offset_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    max_13bit_offset = int(stats.get('compact_branch_13bit_max_offset', stats.get('max_compact_branch_offset', 0)))
    max_10bit_offset = int(stats.get('compact_branch_10bit_max_offset', 0))
    capacity_13bit = float(stats.get('compact_branch_13bit_capacity_percent', max_13bit_offset * 100.0 / float((1 << 13) - 1)))
    capacity_10bit = float(stats.get('compact_branch_10bit_capacity_percent', max_10bit_offset * 100.0 / float((1 << 10) - 1)))
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {int(stats.get('branch_10bit_start_level', 0))} "
        f"| {int(stats.get('branch_13bit_node_count', 0))} "
        f"| {max_13bit_offset} "
        f"| {_format_percent(capacity_13bit)} "
        f"| {int(stats.get('branch_10bit_node_count', 0))} "
        f"| {max_10bit_offset} "
        f"| {_format_percent(capacity_10bit)} "
        f"| {int(stats.get('compact_branch_offset_overflow_count', 0))} |"
    )


def _forced_leaf_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {int(stats.get('forced_leaf_node_count', 0))} "
        f"| {_format_percent(float(stats.get('forced_leaf_pixel_percent', 0.0)))} "
        f"| {_format_percent(float(stats.get('forced_leaf_mean_error_percent', 0.0)))} "
        f"| {_format_percent(float(stats.get('forced_leaf_max_error_percent', 0.0)))} |"
    )


def _node_percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) * 100.0 / float(total)


def _node_composition_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    total_nodes = int(stats.get("node_count", 0))
    branch_nodes = int(stats.get("branch_node_count", 0))
    plane_nodes = int(stats.get("plane_node_count", 0))
    plane30_nodes = int(stats.get("uniform_plane_node_count", 0))
    plane62_nodes = max(plane_nodes - plane30_nodes, 0)
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {branch_nodes} "
        f"| {_format_percent(_node_percent(branch_nodes, total_nodes))} "
        f"| {plane30_nodes} "
        f"| {_format_percent(_node_percent(plane30_nodes, total_nodes))} "
        f"| {plane62_nodes} "
        f"| {_format_percent(_node_percent(plane62_nodes, total_nodes))} "
        f"| {total_nodes} |"
    )


def _visibility_probe_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 0.0))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 0.5))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 1.0))} "
        f"| {_format_percent(_visibility_probe_mismatch(stats, 2.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_shadow_percent', 0.0))} |"
    )


def _pcf_sweep_candidate_markdown_row(report: dict, candidate: dict) -> str:
    stats = candidate["stats"]
    return (
        f"| {_markdown_escape(report.get('_source_path', '<memory>'))} "
        f"| {int(report.get('shadow_resolution', 0))} "
        f"| {_markdown_escape(candidate['variant'])} "
        f"| {_candidate_min_leaf_size(candidate, int(report.get('settings', {}).get('min_leaf_size', 0)))} "
        f"| {_format_ratio(stats['packed_compression_ratio'])} "
        f"| {_format_bpt(_bits_per_texel(stats, 'packed_encoded_bytes'))} "
        f"| {_format_ratio(_packed_decompressed_working_set_ratio(stats))} "
        f"| {_format_percent(stats.get('packed_depth_pcf3_mae_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_visibility_mismatch_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_false_lit_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_conservative_over_full_bias_percent', 0.0))} "
        f"| {_format_percent(_bias_cdf_value(stats, 1.0))} |"
    )


def _pcf3_probe_metric(stats: dict, offset: float, metric: str) -> float:
    for detail in stats.get("packed_depth_pcf3_probe_details", ()):
        if abs(float(detail.get("offset_in_bias", 0.0)) - offset) < 1e-6:
            return float(detail.get(metric, 0.0))
    return 0.0


def _pcf3_delta_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    if "packed_depth_pcf3_mae_percent" not in stats:
        return None
    return (
        f"| {int(report['shadow_resolution'])} "
        f"| {_markdown_escape(variant)} "
        f"| {_format_percent(stats.get('packed_depth_pcf3_mae_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_depth_pcf3_max_error_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_hard_vs_depth_pcf3_mae_percent', 0.0))} "
        f"| {_format_percent(stats.get('packed_hard_vs_depth_pcf3_max_error_percent', 0.0))} "
        f"| {_format_percent(_pcf3_probe_metric(stats, 0.0, 'hard_vs_pcf3_mae_percent'))} "
        f"| {_format_percent(_pcf3_probe_metric(stats, 1.0, 'hard_vs_pcf3_mae_percent'))} |"
    )


def _gpu_decompress_markdown_row(report: dict, variant: str) -> str | None:
    variant_report = report.get("variants", {}).get(variant)
    if variant_report is None:
        return None
    stats = variant_report["stats"]
    if "gpu_decompress_valid" not in stats:
        return None
    timing = variant_report.get("timing_seconds", {})
    return (
        f"| {int(report.get('shadow_resolution', 0))} "
        f"| {_markdown_escape(variant)} "
        f"| {_markdown_escape(stats.get('gpu_decompress_valid', False))} "
        f"| {_format_percent(stats.get('gpu_decompress_vs_cpu_mean_delta_percent', 0.0))} "
        f"| {_format_percent(stats.get('gpu_decompress_vs_cpu_max_delta_percent', 0.0))} "
        f"| {_format_percent(stats.get('gpu_decompress_vs_source_mean_error_percent', 0.0))} "
        f"| {_format_percent(stats.get('gpu_decompress_vs_source_rmse_percent', 0.0))} "
        f"| {_format_percent(stats.get('gpu_decompress_vs_source_max_error_percent', 0.0))} "
        f"| {float(timing.get('gpu_decompress_dispatch', stats.get('gpu_decompress_dispatch_seconds', 0.0))):.3f}s "
        f"| {float(timing.get('gpu_decompress_readback', stats.get('gpu_decompress_readback_seconds', 0.0))):.3f}s |"
    )


def build_sst_multi_markdown_report(reports: list[dict]) -> str:
    sorted_reports = sorted(reports, key=lambda report: (int(report.get("shadow_resolution", 0)), report.get("_source_path", "")))
    matrix_reports_by_resolution: dict[int, dict] = {}
    for report in sorted_reports:
        resolution = int(report.get("shadow_resolution", 0))
        current = matrix_reports_by_resolution.get(resolution)
        if current is None or len(report.get("variants", {})) > len(current.get("variants", {})):
            matrix_reports_by_resolution[resolution] = report
    matrix_reports = [matrix_reports_by_resolution[key] for key in sorted(matrix_reports_by_resolution)]
    lines: list[str] = [
        "# SST Benchmark Comparison",
        "",
    ]
    if sorted_reports:
        scenes = sorted({str(report.get("scene", "unknown")) for report in sorted_reports})
        lines.append(f"- Scenes: `{', '.join(scenes)}`")
        lines.append(
            "- Sources: "
            + ", ".join(f"`{report.get('_source_path', '<memory>')}`" for report in sorted_reports)
        )
        first_settings = sorted_reports[0].get("settings", {})
        if first_settings:
            lines.append(f"- Tile size: `{first_settings.get('tile_size')}`")
            lines.append(f"- Min leaf size: `{first_settings.get('min_leaf_size')}`")
            lines.append(f"- Plane quantization search radius: `{first_settings.get('plane_quantization_search_radius', 0)}`")
            lines.append(f"- Shadow bias: `{first_settings.get('shadow_bias')}`")
    lines.extend(
        [
            "",
            "## Resolution / Variant Matrix",
            "",
            "| Resolution | Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    preferred_variants = (
        "dual_bias",
        "dual_half_visible",
        "dual_visible",
        "dual_relaxed_visible",
        "dual_loose_visible",
        "dual_capped",
        "single",
        "dual_safe",
        "dual_raw",
    )
    for report in matrix_reports:
        variants = report.get("variants", {})
        ordered_variants = [variant for variant in preferred_variants if variant in variants]
        ordered_variants.extend(variant for variant in variants if variant not in ordered_variants)
        for variant in ordered_variants:
            row = _comparison_markdown_row(report, variant)
            if row is not None:
                lines.append(row)

    trend_variants = (
        "dual_bias",
        "dual_half_visible",
        "dual_visible",
        "dual_relaxed_visible",
        "dual_loose_visible",
        "dual_capped",
    )
    if matrix_reports:
        lines.extend(
            [
                "",
                "## Profile Trend Summary",
                "",
                "| Resolution | Variant | Packed | Packed bpt | Mean depth err | Vis mismatch | False lit | Leak > 1 bias |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in matrix_reports:
            for variant in trend_variants:
                row = _profile_trend_markdown_row(report, variant)
                if row is not None:
                    lines.append(row)

        if any(
            report.get("variants", {}).get(variant, {}).get("stats", {}).get("packed_abs_error_bias_cdf")
            for report in matrix_reports
            for variant in trend_variants
        ):
            lines.extend(
                [
                    "",
                    "## Error Distribution",
                    "",
                    "| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in matrix_reports:
                for variant in trend_variants:
                    row = _error_distribution_markdown_row(report, variant)
                    if row is not None:
                        lines.append(row)

        if any(
            "packed_depth_pcf3_mae_percent" in report.get("variants", {}).get(variant, {}).get("stats", {})
            for report in matrix_reports
            for variant in trend_variants
        ):
            lines.extend(
                [
                    "",
                    "## Depth Texture PCF3 Delta",
                    "",
                    "This estimates the shader-side `Compact SST PCF3` mode against the existing `Depth Texture` PCF3 path.",
                    "",
                    "| Resolution | Variant | SST PCF3 vs Depth PCF3 MAE | SST PCF3 max | SST hard vs Depth PCF3 MAE | SST hard max | @0B hard MAE | @1B hard MAE |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in matrix_reports:
                for variant in trend_variants:
                    row = _pcf3_delta_markdown_row(report, variant)
                    if row is not None:
                        lines.append(row)

        if any(
            "gpu_decompress_valid" in variant_report.get("stats", {})
            for report in sorted_reports
            for variant_report in report.get("variants", {}).values()
        ):
            lines.extend(
                [
                    "",
                    "## GPU Decompression",
                    "",
                    "This validates the paper-style Compact SST stream by decompressing it on the GPU back to an `r32_float` depth texture.",
                    "",
                    "| Resolution | Variant | GPU == CPU | GPU vs CPU mean | GPU vs CPU max | GPU vs source mean | GPU vs source RMSE | GPU vs source max | Dispatch | Readback |",
                    "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in sorted_reports:
                variants = report.get("variants", {})
                ordered_variants = [variant for variant in trend_variants if variant in variants]
                ordered_variants.extend(variant for variant in variants if variant not in ordered_variants)
                for variant in ordered_variants:
                    row = _gpu_decompress_markdown_row(report, variant)
                    if row is not None:
                        lines.append(row)

        if any(
            "dual_second_hit_percent" in report.get("variants", {}).get(variant, {}).get("stats", {})
            for report in matrix_reports
            for variant in trend_variants
        ):
            lines.extend(
                [
                    "",
                    "## Dual Layer Utilization",
                    "",
                    "| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in matrix_reports:
                for variant in trend_variants:
                    row = _dual_layer_markdown_row(report, variant)
                    if row is not None:
                        lines.append(row)

        lines.extend(
            [
                "",
                "## Memory Breakdown",
                "",
                "Decomp working set counts the persistent Compact SST stream plus one full-resolution `r32_float` decompressed depth texture.",
                "",
                "| Resolution | Variant | Packed bytes | Packed ratio | Packed stream bpt | Tile roots bpt | Packed total bpt | Decomp texture bytes | Packed+decomp bytes | Packed+decomp bpt | Packed+decomp ratio | Fixed64 bytes | Fixed64 ratio |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in matrix_reports:
            for variant in trend_variants:
                row = _memory_breakdown_markdown_row(report, variant)
                if row is not None:
                    lines.append(row)

        lines.extend(
            [
                "",
                "## Paper Stream Breakdown",
                "",
                "| Resolution | Variant | Node words | Branch bpt | 30-bit leaf bpt | 62-bit plane bpt | Tile roots bpt | Packed total bpt |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in matrix_reports:
            for variant in trend_variants:
                row = _paper_stream_breakdown_markdown_row(report, variant)
                if row is not None:
                    lines.append(row)

        lines.extend(
            [
                "",
                "## Branch Offset Packing",
                "",
                "| Resolution | Variant | 10-bit start level | 13-bit branches | Max 13-bit offset | 13-bit capacity | 10-bit branches | Max 10-bit offset | 10-bit capacity | Overflows |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in matrix_reports:
            for variant in trend_variants:
                row = _branch_offset_markdown_row(report, variant)
                if row is not None:
                    lines.append(row)

        lines.extend(
            [
                "",
                "## Forced Leaf Diagnostics",
                "",
                "| Resolution | Variant | Forced leaves | Forced pixels | Forced mean err | Forced max err |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for report in matrix_reports:
            for variant in trend_variants:
                row = _forced_leaf_markdown_row(report, variant)
                if row is not None:
                    lines.append(row)

        lines.extend(
            [
                "",
                "## Node Type Composition",
                "",
                "| Resolution | Variant | Branches | Branch % | 30-bit leaves | 30-bit % | 62-bit planes | 62-bit % | Total nodes |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in matrix_reports:
            for variant in trend_variants:
                row = _node_composition_markdown_row(report, variant)
                if row is not None:
                    lines.append(row)

        if any(
            report.get("variants", {}).get(variant, {}).get("stats", {}).get("packed_visibility_probe_details")
            for report in matrix_reports
            for variant in trend_variants
        ):
            lines.extend(
                [
                    "",
                    "## Visibility Probe Breakdown",
                    "",
                    "| Resolution | Variant | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in matrix_reports:
                for variant in trend_variants:
                    row = _visibility_probe_markdown_row(report, variant)
                    if row is not None:
                        lines.append(row)

    recommendation_reports = [report for report in sorted_reports if report.get("sweep_recommendations")]
    pcf_sweep_candidates = [
        (report, candidate)
        for report in sorted_reports
        for candidate in report.get("sweep_results", [])
        if "packed_depth_pcf3_mae_percent" in candidate.get("stats", {})
    ]
    if pcf_sweep_candidates:
        lines.extend(
            [
                "",
                "## PCF-Aware Sweep Candidates",
                "",
                "| Source | Resolution | Variant | Min leaf | Packed | Packed bpt | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias | <=1 bias |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report, candidate in sorted(
            pcf_sweep_candidates,
            key=lambda item: (
                int(item[0].get("shadow_resolution", 0)),
                str(item[0].get("_source_path", "")),
                str(item[1].get("variant", "")),
                int(item[1].get("min_leaf_size", 0)),
            ),
        ):
            lines.append(_pcf_sweep_candidate_markdown_row(report, candidate))

    if recommendation_reports:
        lines.extend(
            [
                "",
                "## Sweep Recommendations",
                "",
                "| Source | Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Packed bpt | Packed+decomp ratio | Vis mismatch | PCF3 MAE | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in recommendation_reports:
            source = report.get("_source_path", f"res {report.get('shadow_resolution')}")
            default_tile_size = int(report.get("settings", {}).get("tile_size", 0))
            default_min_leaf_size = int(report.get("settings", {}).get("min_leaf_size", 0))
            for recommendation in report.get("sweep_recommendations", []):
                lines.append(
                    f"| {_markdown_escape(source)} "
                    f"| {_markdown_escape(recommendation['label'])} "
                    f"| {_markdown_escape(recommendation['variant'])} "
                    f"| {_candidate_tile_size(recommendation, default_tile_size)} "
                    f"| {_candidate_min_leaf_size(recommendation, default_min_leaf_size)} "
                    f"| {_format_optional_float(recommendation.get('forced_leaf_error_cap'))} "
                    f"| {_format_bool(recommendation.get('forced_split_bias_fit', False))} "
                    f"| {float(recommendation['plane_error_threshold']):.6f} "
                    f"| {_markdown_escape(recommendation['dual_depth_slack'])} "
                    f"| {int(recommendation.get('plane_quantization_search_radius', 0))} "
                    f"| {_format_ratio(recommendation['packed_compression_ratio'])} "
                    f"| {_format_bpt(_recommendation_packed_bpt(recommendation))} "
                    f"| {_format_ratio(_recommendation_decomp_ratio(recommendation))} "
                    f"| {_format_percent(recommendation['packed_visibility_mismatch_percent'])} "
                    f"| {_format_optional_percent(recommendation.get('packed_depth_pcf3_mae_percent'))} "
                    f"| {_format_percent(recommendation['packed_false_lit_percent'])} "
                    f"| {_format_percent(recommendation['packed_false_shadow_percent'])} "
                    f"| {_format_percent(recommendation['packed_leak_over_full_bias_percent'])} "
                    f"| {_format_percent(recommendation.get('packed_conservative_over_full_bias_percent', 0.0))} "
                    f"| {_format_percent(recommendation.get('packed_abs_error_within_1_bias_percent', 0.0))} |"
                )

    pareto_reports = [report for report in sorted_reports if report.get("sweep_pareto")]
    if pareto_reports:
        lines.extend(
            [
                "",
                "## Pareto Fronts",
                "",
                "| Source | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in pareto_reports:
            source = report.get("_source_path", f"res {report.get('shadow_resolution')}")
            default_tile_size = int(report.get("settings", {}).get("tile_size", 0))
            default_min_leaf_size = int(report.get("settings", {}).get("min_leaf_size", 0))
            for candidate in report.get("sweep_pareto", [])[:16]:
                lines.append(
                    f"| {_markdown_escape(source)} | "
                    + _sweep_candidate_markdown_row(candidate, default_tile_size, default_min_leaf_size).lstrip("| ")
                )

        if any(
            candidate.get("stats", {}).get("packed_visibility_probe_details")
            for report in pareto_reports
            for candidate in report.get("sweep_pareto", [])[:16]
        ):
            lines.extend(
                [
                    "",
                    "## Pareto Visibility Probes",
                    "",
                    "| Source | Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for report in pareto_reports:
                source = report.get("_source_path", f"res {report.get('shadow_resolution')}")
                default_tile_size = int(report.get("settings", {}).get("tile_size", 0))
                default_min_leaf_size = int(report.get("settings", {}).get("min_leaf_size", 0))
                for candidate in report.get("sweep_pareto", [])[:16]:
                    if not candidate.get("stats", {}).get("packed_visibility_probe_details"):
                        continue
                    lines.append(
                        f"| {_markdown_escape(source)} | "
                        + _sweep_probe_markdown_row(candidate, default_tile_size, default_min_leaf_size).lstrip("| ")
                    )

    return "\n".join(lines) + "\n"


def _load_rgb_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for static shadow image comparison") from exc

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _image_diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"Image shapes do not match: {reference.shape} vs {candidate.shape}")

    diff = candidate - reference
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = math.inf if mse <= 1e-16 else 20.0 * math.log10(1.0 / rmse)
    return {
        "mean_abs": float(np.mean(abs_diff)),
        "mean_abs_percent": float(np.mean(abs_diff) * 100.0),
        "rmse": float(rmse),
        "rmse_percent": float(rmse * 100.0),
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "max_abs_percent": float(np.max(abs_diff) * 100.0) if abs_diff.size else 0.0,
        "changed_pixel_percent": float(np.mean(np.any(abs_diff > (1.0 / 255.0), axis=2)) * 100.0),
        "psnr_db": float(psnr) if math.isfinite(psnr) else "inf",
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
    }


def _save_rgb_preview(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for saving static shadow compare previews") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    preview = np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))
    Image.fromarray(preview, mode="RGB").save(path)


def _save_abs_diff_preview(reference: np.ndarray, candidate: np.ndarray, path: Path, scale: float) -> None:
    abs_diff = np.abs(candidate - reference)
    _save_rgb_preview(path, abs_diff * max(0.0, float(scale)))


def _format_metric_percent(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}%"


def build_static_shadow_compare_markdown(report: dict) -> str:
    sst_stats = report.get("sst_stats")
    sst_profile = _compare_report_profile_label(report)
    lines = [
        "# Static Shadow Runtime Compare",
        "",
        f"- Scene: `{report.get('scene')}`",
        f"- Resolution: `{report.get('width')}x{report.get('height')}`",
        f"- Frames: `{report.get('frames')}`",
        f"- Shadow resolution: `{report.get('shadow_resolution')}`",
        f"- Reference mode: `{report.get('reference_mode')}`",
        f"- Bake time: `{float(report.get('bake_seconds', 0.0)):.3f}s`",
        f"- Encode time: `{float(report.get('encode_seconds', 0.0)):.3f}s`",
        f"- SST encoded: `{bool(report.get('sst_encoded', False))}`",
        f"- SST preset: `{report.get('sst_preset_name', 'unknown')}`",
        f"- SST profile: `{sst_profile}`",
        f"- SST tile/leaf: `{_compare_report_tile_size(report)}/{_compare_report_leaf_size(report)}`",
    ]
    if sst_stats:
        lines.extend(
            [
            f"- SST packed ratio: `{_format_ratio(float(sst_stats.get('packed_compression_ratio', 0.0)))}`",
            f"- SST packed bpt: `{_format_bpt(_bits_per_texel(sst_stats, 'packed_encoded_bytes'))}`",
            f"- SST packed+decomp ratio: `{_format_ratio(_packed_decompressed_working_set_ratio(sst_stats))}`",
            "",
            "## SST Encoding",
            "",
            "| Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {int(sst_stats.get('node_count', 0))} "
                f"| {int(sst_stats.get('packed_encoded_bytes', 0)):,} "
                f"| {_format_bpt(_bits_per_texel(sst_stats, 'packed_encoded_bytes'))} "
                f"| {_format_ratio(float(sst_stats.get('packed_compression_ratio', 0.0)))} "
                f"| {_packed_decompressed_working_set_bytes(sst_stats):,} "
                f"| {_format_ratio(_packed_decompressed_working_set_ratio(sst_stats))} "
                f"| {_format_percent(float(sst_stats.get('packed_depth_pcf3_mae_percent', 0.0)))} "
                f"| {_format_percent(float(sst_stats.get('packed_visibility_mismatch_percent', 0.0)))} "
                f"| {_format_percent(float(sst_stats.get('packed_false_lit_percent', 0.0)))} "
                f"| {_format_percent(float(sst_stats.get('packed_conservative_over_full_bias_percent', 0.0)))} |"
            ),
            ]
        )
    lines.extend(
        [
            "",
            "## Images",
            "",
            "| Mode | Output | Render seconds |",
            "|---|---|---:|",
        ]
    )
    for render in report.get("renders", []):
        lines.append(
            f"| {_markdown_escape(render['mode'])} "
            f"| `{render['output']}` "
            f"| {float(render.get('seconds', 0.0)):.3f} |"
        )

    comparisons = report.get("comparisons", [])
    if comparisons:
        lines.extend(
            [
                "",
                "## Diffs",
                "",
                "| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            metrics = comparison["metrics"]
            psnr = metrics.get("psnr_db", "inf")
            psnr_text = psnr if isinstance(psnr, str) else f"{float(psnr):.2f} dB"
            lines.append(
                f"| {_markdown_escape(comparison['candidate_mode'])} "
                f"| `{comparison.get('diff_preview', '')}` "
                f"| {_format_metric_percent(metrics['mean_abs_percent'])} "
                f"| {_format_metric_percent(metrics['rmse_percent'])} "
                f"| {_format_metric_percent(metrics['max_abs_percent'])} "
                f"| {_format_metric_percent(metrics['changed_pixel_percent'])} "
                f"| {psnr_text} "
                f"| {float(metrics['candidate_mean']):.4f} |"
            )

    pairwise = report.get("pairwise_comparisons", [])
    if pairwise:
        lines.extend(
            [
                "",
                "## Pairwise Diffs",
                "",
                "| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in pairwise:
            metrics = comparison["metrics"]
            psnr = metrics.get("psnr_db", "inf")
            psnr_text = psnr if isinstance(psnr, str) else f"{float(psnr):.2f} dB"
            lines.append(
                f"| {_markdown_escape(comparison['mode_a'])} "
                f"| {_markdown_escape(comparison['mode_b'])} "
                f"| `{comparison.get('diff_preview', '')}` "
                f"| {_format_metric_percent(metrics['mean_abs_percent'])} "
                f"| {_format_metric_percent(metrics['rmse_percent'])} "
                f"| {_format_metric_percent(metrics['max_abs_percent'])} "
                f"| {_format_metric_percent(metrics['changed_pixel_percent'])} "
                f"| {psnr_text} |"
            )

    return "\n".join(lines) + "\n"


def _load_static_shadow_compare_reports(paths: list[Path]) -> list[dict]:
    reports: list[dict] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_source_path"] = str(path)
        reports.append(report)
    return reports


def _compare_report_profile_label(report: dict) -> str:
    profile_name = report.get("sst_fit_profile_name")
    if profile_name:
        return str(profile_name)
    source = str(report.get("_source_path", "")).lower()
    if "relaxed" in source:
        return "Dual Relaxed Visible"
    if "loose" in source:
        return "Dual Loose Visible"
    if "half" in source:
        return "Dual Half Visible"
    if "bias" in source:
        return "Dual Bias"
    if "visible" in source or "compact_pcf" in source:
        return "Dual Visible"
    return "unknown"


def _compare_report_preset_label(report: dict) -> str:
    preset_name = report.get("sst_preset_name")
    if preset_name:
        return str(preset_name)
    profile = _compare_report_profile_label(report)
    leaf = _compare_report_leaf_size(report)
    tile = _compare_report_tile_size(report)
    if profile == "Dual Visible" and tile == 128 and leaf == 2:
        return "Quality"
    if profile == "Dual Relaxed Visible" and tile == 128 and leaf == 2:
        return "High Compression"
    return "Manual"


def _compare_report_leaf_size(report: dict) -> int:
    if "sst_min_leaf_size" in report:
        return int(report.get("sst_min_leaf_size", 0))
    source = str(report.get("_source_path", "")).lower()
    for leaf_size in (1, 2, 4, 8):
        if f"leaf{leaf_size}" in source:
            return leaf_size
    return 0


def _compare_report_tile_size(report: dict) -> int:
    if "sst_tile_size" in report:
        return int(report.get("sst_tile_size", 0))
    sst_stats = report.get("sst_stats") or {}
    resolution = int(report.get("shadow_resolution", 0))
    tile_count = int(sst_stats.get("tile_count", 0))
    if resolution > 0 and tile_count > 0:
        grid = int(round(math.sqrt(float(tile_count))))
        if grid > 0:
            return max(1, resolution // grid)
    return 0


def _render_seconds_for_mode(report: dict, mode: str) -> float:
    for render in report.get("renders", []):
        if render.get("mode") == mode:
            return float(render.get("seconds", 0.0))
    return 0.0


def _format_psnr(value) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.2f} dB"


def _psnr_numeric(value) -> float:
    if isinstance(value, str):
        return math.inf if value.lower() == "inf" else 0.0
    return float(value)


def _runtime_compare_candidates(reports: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for report in reports:
        sst_stats = report.get("sst_stats") or {}
        if not sst_stats:
            continue
        source = report.get("_source_path", "<memory>")
        profile_label = _compare_report_profile_label(report)
        tile_size = _compare_report_tile_size(report)
        leaf_size = _compare_report_leaf_size(report)
        shadow_res = int(report.get("shadow_resolution", 0))
        image_res = f"{int(report.get('width', 0))}x{int(report.get('height', 0))}"
        for comparison in report.get("comparisons", []):
            metrics = comparison.get("metrics", {})
            candidate_mode = str(comparison.get("candidate_mode", "unknown"))
            candidates.append(
                {
                    "source": source,
                    "image_res": image_res,
                    "shadow_res": shadow_res,
                    "preset": _compare_report_preset_label(report),
                    "profile": profile_label,
                    "tile": tile_size,
                    "leaf": leaf_size,
                    "mode": candidate_mode,
                    "packed_ratio": float(sst_stats.get("packed_compression_ratio", 0.0)),
                    "packed_bpt": _bits_per_texel(sst_stats, "packed_encoded_bytes"),
                    "decomp_ratio": _packed_decompressed_working_set_ratio(sst_stats),
                    "pcf3_mae_percent": float(sst_stats.get("packed_depth_pcf3_mae_percent", 0.0)),
                    "vis_mismatch_percent": float(sst_stats.get("packed_visibility_mismatch_percent", 0.0)),
                    "false_lit_percent": float(sst_stats.get("packed_false_lit_percent", 0.0)),
                    "mean_abs_percent": float(metrics.get("mean_abs_percent", 0.0)),
                    "rmse_percent": float(metrics.get("rmse_percent", 0.0)),
                    "changed_pixel_percent": float(metrics.get("changed_pixel_percent", 0.0)),
                    "psnr_db": metrics.get("psnr_db", "inf"),
                    "candidate_render_seconds": _render_seconds_for_mode(report, candidate_mode),
                    "encode_seconds": float(report.get("encode_seconds", 0.0)),
                }
            )
    return candidates


def _best_runtime_candidate(candidates: list[dict], label: str, predicate, key) -> dict | None:
    filtered = [candidate for candidate in candidates if predicate(candidate)]
    if not filtered:
        return None
    best = max(filtered, key=key)
    result = dict(best)
    result["label"] = label
    return result


def _compute_runtime_compare_recommendations(candidates: list[dict]) -> list[dict]:
    specs = (
        (
            "best_quality",
            lambda c: c["false_lit_percent"] <= 1e-7,
            lambda c: (-c["mean_abs_percent"], -c["changed_pixel_percent"], c["packed_ratio"]),
        ),
        (
            "max_compression_mean_abs_le_0_02_percent",
            lambda c: c["false_lit_percent"] <= 1e-7 and c["mean_abs_percent"] <= 0.02,
            lambda c: (c["packed_ratio"], -c["mean_abs_percent"], -c["changed_pixel_percent"]),
        ),
        (
            "max_compression_changed_px_le_0_5_percent",
            lambda c: c["false_lit_percent"] <= 1e-7 and c["changed_pixel_percent"] <= 0.5,
            lambda c: (c["packed_ratio"], -c["mean_abs_percent"], -c["changed_pixel_percent"]),
        ),
        (
            "max_compression_mean_abs_le_0_05_percent",
            lambda c: c["false_lit_percent"] <= 1e-7 and c["mean_abs_percent"] <= 0.05,
            lambda c: (c["packed_ratio"], -c["mean_abs_percent"], -c["changed_pixel_percent"]),
        ),
        (
            "max_compression_changed_px_le_1_percent",
            lambda c: c["false_lit_percent"] <= 1e-7 and c["changed_pixel_percent"] <= 1.0,
            lambda c: (c["packed_ratio"], -c["mean_abs_percent"], -c["changed_pixel_percent"]),
        ),
    )
    recommendations: list[dict] = []
    seen: set[str] = set()
    for label, predicate, key in specs:
        recommendation = _best_runtime_candidate(candidates, label, predicate, key)
        if recommendation is not None and label not in seen:
            seen.add(label)
            recommendations.append(recommendation)
    return recommendations


def _runtime_recommendation_markdown_row(candidate: dict) -> str:
    return (
        f"| {_markdown_escape(candidate['label'])} "
        f"| {_markdown_escape(candidate['source'])} "
        f"| {candidate['shadow_res']} "
        f"| {_markdown_escape(candidate['preset'])} "
        f"| {_markdown_escape(candidate['profile'])} "
        f"| {candidate['tile']} "
        f"| {candidate['leaf']} "
        f"| {_markdown_escape(candidate['mode'])} "
        f"| {_format_ratio(candidate['packed_ratio'])} "
        f"| {_format_bpt(candidate['packed_bpt'])} "
        f"| {_format_percent(candidate['pcf3_mae_percent'])} "
        f"| {_format_percent(candidate['mean_abs_percent'])} "
        f"| {_format_percent(candidate['rmse_percent'])} "
        f"| {_format_percent(candidate['changed_pixel_percent'])} "
        f"| {_format_psnr(candidate['psnr_db'])} |"
    )


def build_static_shadow_compare_multi_markdown(reports: list[dict]) -> str:
    sorted_reports = sorted(
        reports,
        key=lambda report: (
            int(report.get("shadow_resolution", 0)),
            int(report.get("width", 0)),
            int(report.get("height", 0)),
            str(report.get("_source_path", "")),
        ),
    )
    lines: list[str] = [
        "# Static Shadow Runtime Compare Summary",
        "",
    ]
    if sorted_reports:
        scenes = sorted({str(report.get("scene", "unknown")) for report in sorted_reports})
        lines.append(f"- Scenes: `{', '.join(scenes)}`")
        lines.append("- Sources: " + ", ".join(f"`{report.get('_source_path', '<memory>')}`" for report in sorted_reports))

    runtime_candidates = _runtime_compare_candidates(sorted_reports)
    runtime_recommendations = _compute_runtime_compare_recommendations(runtime_candidates)
    if runtime_recommendations:
        lines.extend(
            [
                "",
                "## Runtime Recommendations",
                "",
                "| Constraint | Source | Shadow res | Preset | Profile | Tile | Leaf | Mode | Packed | Packed bpt | PCF3 MAE | Mean abs | RMSE | Changed px | PSNR |",
                "|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for recommendation in runtime_recommendations:
            lines.append(_runtime_recommendation_markdown_row(recommendation))

    lines.extend(
        [
            "",
            "## Compact Runtime Quality",
            "",
            "| Source | Image res | Shadow res | Preset | Profile | Tile | Leaf | Candidate | Packed | Packed bpt | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Mean abs | RMSE | Changed px | PSNR | Candidate render | Encode time |",
            "|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for report in sorted_reports:
        source = report.get("_source_path", "<memory>")
        sst_stats = report.get("sst_stats") or {}
        packed_ratio = float(sst_stats.get("packed_compression_ratio", 0.0))
        packed_bpt = _bits_per_texel(sst_stats, "packed_encoded_bytes") if sst_stats else 0.0
        decomp_ratio = _packed_decompressed_working_set_ratio(sst_stats) if sst_stats else 0.0
        pcf3_mae = float(sst_stats.get("packed_depth_pcf3_mae_percent", 0.0))
        vis_mismatch = float(sst_stats.get("packed_visibility_mismatch_percent", 0.0))
        false_lit = float(sst_stats.get("packed_false_lit_percent", 0.0))
        image_res = f"{int(report.get('width', 0))}x{int(report.get('height', 0))}"
        shadow_res = int(report.get("shadow_resolution", 0))
        profile_label = _compare_report_profile_label(report)
        tile_size = _compare_report_tile_size(report)
        leaf_size = _compare_report_leaf_size(report)
        for comparison in report.get("comparisons", []):
            metrics = comparison.get("metrics", {})
            candidate_mode = str(comparison.get("candidate_mode", "unknown"))
            lines.append(
                f"| {_markdown_escape(source)} "
                f"| {image_res} "
                f"| {shadow_res} "
                f"| {_markdown_escape(_compare_report_preset_label(report))} "
                f"| {_markdown_escape(profile_label)} "
                f"| {tile_size} "
                f"| {leaf_size} "
                f"| {_markdown_escape(candidate_mode)} "
                f"| {_format_ratio(packed_ratio)} "
                f"| {_format_bpt(packed_bpt)} "
                f"| {_format_ratio(decomp_ratio)} "
                f"| {_format_percent(pcf3_mae)} "
                f"| {_format_percent(vis_mismatch)} "
                f"| {_format_percent(false_lit)} "
                f"| {_format_metric_percent(metrics.get('mean_abs_percent', 0.0))} "
                f"| {_format_metric_percent(metrics.get('rmse_percent', 0.0))} "
                f"| {_format_metric_percent(metrics.get('changed_pixel_percent', 0.0))} "
                f"| {_format_psnr(metrics.get('psnr_db', 'inf'))} "
                f"| {_render_seconds_for_mode(report, candidate_mode):.3f}s "
                f"| {float(report.get('encode_seconds', 0.0)):.3f}s |"
            )

    lines.extend(
        [
            "",
            "## Storage",
            "",
            "| Source | Shadow res | Preset | Profile | Tile | Leaf | Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | Fixed64 bytes | Fixed64 ratio |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for report in sorted_reports:
        sst_stats = report.get("sst_stats") or {}
        if not sst_stats:
            continue
        lines.append(
            f"| {_markdown_escape(report.get('_source_path', '<memory>'))} "
            f"| {int(report.get('shadow_resolution', 0))} "
            f"| {_markdown_escape(_compare_report_preset_label(report))} "
            f"| {_markdown_escape(_compare_report_profile_label(report))} "
            f"| {_compare_report_tile_size(report)} "
            f"| {_compare_report_leaf_size(report)} "
            f"| {int(sst_stats.get('node_count', 0))} "
            f"| {int(sst_stats.get('packed_encoded_bytes', 0)):,} "
            f"| {_format_bpt(_bits_per_texel(sst_stats, 'packed_encoded_bytes'))} "
            f"| {_format_ratio(float(sst_stats.get('packed_compression_ratio', 0.0)))} "
            f"| {_packed_decompressed_working_set_bytes(sst_stats):,} "
            f"| {_format_ratio(_packed_decompressed_working_set_ratio(sst_stats))} "
            f"| {int(sst_stats.get('fixed64_encoded_bytes', 0)):,} "
            f"| {_format_ratio(float(sst_stats.get('fixed64_compression_ratio', 0.0)))} |"
        )

    return "\n".join(lines) + "\n"


def run_static_shadow_compare(args: argparse.Namespace) -> None:
    from app import App, AppConfig
    from path_tracing_renderer import PathTracingRenderer
    from scene import Scene

    modes = _parse_static_shadow_mode_list(args.static_shadow_compare_modes)
    output_dir = Path(args.static_shadow_compare_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    needs_sst = any(
        mode_value in (
            Scene.SHADOW_MODE_SST,
            Scene.SHADOW_MODE_PACKED_SST,
            Scene.SHADOW_MODE_COMPACT_SST,
            Scene.SHADOW_MODE_COMPACT_SST_PCF3,
            Scene.SHADOW_MODE_DECOMPRESSED_SST,
        )
        for _, mode_value in modes
    )

    config = AppConfig(
        width=max(1, args.width),
        height=max(1, args.height),
        headless=True,
        headless_frame_count=max(1, args.frames),
        headless_output=output_dir / "warmup.png",
        vsync=args.vsync,
        srgb_output=not args.no_srgb,
        camera_move_test=args.camera_move_test,
        scene_path=args.scene,
        sst_encoder_backend=args.sst_encoder,
    )
    app = App(config=config)
    app.set_renderer(PathTracingRenderer())
    app.scene.set_sst_encoder_backend(args.sst_encoder)
    app.scene.set_static_shadow_resolution(args.shadow_resolution)
    _apply_runtime_sst_options(app.scene, args)
    app.scene.static_shadow_auto_encode_sst = False

    bake_start = time.perf_counter()
    app.scene.bake_static_shadow_depth_map()
    bake_seconds = time.perf_counter() - bake_start
    encode_seconds = 0.0
    if needs_sst and not app.scene.sst_enabled:
        encode_start = time.perf_counter()
        app.scene.encode_sparse_shadow_tree()
        encode_seconds = time.perf_counter() - encode_start

    renders: list[dict] = []
    for index, (mode_name, mode_value) in enumerate(modes):
        output_path = output_dir / f"{index:02d}_{mode_name}.png"
        requested_mode = mode_value
        if requested_mode in (
            Scene.SHADOW_MODE_SST,
            Scene.SHADOW_MODE_PACKED_SST,
            Scene.SHADOW_MODE_COMPACT_SST,
            Scene.SHADOW_MODE_COMPACT_SST_PCF3,
            Scene.SHADOW_MODE_DECOMPRESSED_SST,
        ) and not app.scene.sst_enabled:
            requested_mode = Scene.SHADOW_MODE_DEPTH_TEXTURE
        app.scene.static_shadow_mode = requested_mode
        app.scene._sync_shadow_mode_ui()
        app.scene._update_static_shadow_status()
        app.scene._reset_accumulation()
        start = time.perf_counter()
        app.render_headless_to_file(output_path, max(1, args.frames))
        seconds = time.perf_counter() - start
        renders.append(
            {
                "mode": mode_name,
                "mode_value": requested_mode,
                "output": str(output_path),
                "seconds": seconds,
            }
        )

    images = {render["mode"]: _load_rgb_image(Path(render["output"])) for render in renders}
    reference = images[renders[0]["mode"]]
    comparisons: list[dict] = []
    for render in renders[1:]:
        candidate = images[render["mode"]]
        diff_preview = output_dir / f"diff_{renders[0]['mode']}_vs_{render['mode']}.png"
        _save_abs_diff_preview(reference, candidate, diff_preview, args.static_shadow_compare_diff_scale)
        comparisons.append(
            {
                "reference_mode": renders[0]["mode"],
                "candidate_mode": render["mode"],
                "diff_preview": str(diff_preview),
                "diff_preview_scale": max(0.0, float(args.static_shadow_compare_diff_scale)),
                "metrics": _image_diff_metrics(reference, candidate),
            }
        )

    pairwise_comparisons: list[dict] = []
    for i, render_a in enumerate(renders):
        for render_b in renders[i + 1:]:
            image_a = images[render_a["mode"]]
            image_b = images[render_b["mode"]]
            diff_preview = output_dir / f"diff_{render_a['mode']}_vs_{render_b['mode']}.png"
            _save_abs_diff_preview(image_a, image_b, diff_preview, args.static_shadow_compare_diff_scale)
            pairwise_comparisons.append(
                {
                    "mode_a": render_a["mode"],
                    "mode_b": render_b["mode"],
                    "diff_preview": str(diff_preview),
                    "diff_preview_scale": max(0.0, float(args.static_shadow_compare_diff_scale)),
                    "metrics": _image_diff_metrics(image_a, image_b),
                }
            )

    report = {
        "scene": args.scene or "<demo>",
        "width": max(1, args.width),
        "height": max(1, args.height),
        "frames": max(1, args.frames),
        "shadow_resolution": _shadow_resolution_budget(args.shadow_resolution),
        "shadow_size": list(app.scene.static_shadow_size),
        "shadow_resolution_input": _shadow_resolution_label(args.shadow_resolution),
        "bake_seconds": bake_seconds,
        "encode_seconds": encode_seconds,
        "sst_encoded": bool(app.scene.sst_enabled),
        "sst_preset": int(app.scene.sst_preset),
        "sst_preset_name": app.scene._sst_preset_name(),
        "sst_fit_profile": int(app.scene.sst_fit_profile),
        "sst_fit_profile_name": app.scene._sst_fit_profile_name(),
        "sst_tile_size": int(app.scene.sst_tile_size),
        "sst_min_leaf_size": int(app.scene.sst_min_leaf_size),
        "sst_stats": asdict(app.scene.sst_stats) if app.scene.sst_stats is not None else None,
        "diff_preview_scale": max(0.0, float(args.static_shadow_compare_diff_scale)),
        "reference_mode": renders[0]["mode"],
        "renders": renders,
        "comparisons": comparisons,
        "pairwise_comparisons": pairwise_comparisons,
    }

    json_path = Path(args.static_shadow_compare_report) if args.static_shadow_compare_report else output_dir / "compare_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = (
        Path(args.static_shadow_compare_markdown_output)
        if args.static_shadow_compare_markdown_output
        else output_dir / "compare_report.md"
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_static_shadow_compare_markdown(report), encoding="utf-8")

    print(f"Static shadow compare JSON saved to {json_path}")
    print(f"Static shadow compare Markdown saved to {markdown_path}")
    for comparison in comparisons:
        metrics = comparison["metrics"]
        print(
            f"{comparison['reference_mode']} vs {comparison['candidate_mode']}: "
            f"mean={metrics['mean_abs_percent']:.4f}% "
            f"rmse={metrics['rmse_percent']:.4f}% "
            f"max={metrics['max_abs_percent']:.4f}% "
            f"changed={metrics['changed_pixel_percent']:.4f}%"
        )


def run_sst_benchmark(args: argparse.Namespace) -> None:
    from app import App, AppConfig
    from sst_decompress import SSTDecompressor

    config = AppConfig(
        width=max(1, args.width),
        height=max(1, args.height),
        headless=True,
        headless_frame_count=0,
        scene_path=args.scene,
        sst_encoder_backend=args.sst_encoder,
    )
    app = App(config=config)
    scene = app.scene
    scene.set_sst_encoder_backend(args.sst_encoder)
    scene.set_static_shadow_resolution(args.shadow_resolution)
    scene.static_shadow_auto_encode_sst = False
    shadow_bias = scene.static_shadow_depth_bias if args.sst_shadow_bias is None else max(0.0, args.sst_shadow_bias)

    bake_start = time.perf_counter()
    scene.bake_static_shadow_depth_map()
    bake_seconds = time.perf_counter() - bake_start

    readback_start = time.perf_counter()
    front_depth = scene.static_shadow_depth_texture.to_numpy()
    second_depth = scene.static_shadow_second_depth_texture.to_numpy()
    readback_seconds = time.perf_counter() - readback_start

    gpu_decompressor = SSTDecompressor(app.device) if args.sst_gpu_decompress else None

    variants: dict[str, dict] = {}
    for variant in _selected_benchmark_variants(args.sst_benchmark_variant):
        encoded_report = _encode_sst_variant(
            args,
            front_depth,
            second_depth,
            variant,
            shadow_bias,
            max(0.0, args.sst_plane_error_threshold),
            max(0.0, args.sst_dual_depth_slack),
            device=app.device if gpu_decompressor is not None else None,
            gpu_decompressor=gpu_decompressor,
        )
        variants[variant] = {
            "stats": encoded_report["stats"],
            "timing_seconds": encoded_report["timing_seconds"],
        }

    sweep_results: list[dict] = []
    sweep_pareto: list[dict] = []
    sweep_recommendations: list[dict] = []
    if args.sst_sweep:
        plane_errors = _parse_float_list(args.sst_sweep_plane_error_thresholds, (max(0.0, args.sst_plane_error_threshold),))
        dual_slacks = _parse_float_list(args.sst_sweep_dual_depth_slacks, (max(0.0, args.sst_dual_depth_slack),))
        quantization_radii = _parse_int_list(
            args.sst_sweep_plane_quantization_search_radii,
            (max(0, args.sst_plane_quantization_search_radius),),
        )
        tile_sizes = _parse_int_list(args.sst_sweep_tile_sizes, (max(1, args.sst_tile_size),))
        min_leaf_sizes = _parse_int_list(args.sst_sweep_min_leaf_sizes, (max(1, args.sst_min_leaf_size),))
        forced_leaf_error_caps = _parse_optional_float_list(args.sst_sweep_forced_leaf_error_caps, (args.sst_forced_leaf_error_cap,))
        forced_split_bias_modes = _parse_bool_list(args.sst_sweep_forced_split_bias_fit, (args.sst_forced_split_bias_fit,))
        sweep_variants = _parse_variant_list(args.sst_sweep_variants)
        seen_configs: set[tuple[str, int, int, float | None, bool, float, float, int]] = set()
        for tile_size in tile_sizes:
            for min_leaf_size in min_leaf_sizes:
                for forced_leaf_error_cap in forced_leaf_error_caps:
                    for forced_split_bias_fit in forced_split_bias_modes:
                        for plane_error in plane_errors:
                            for variant in sweep_variants:
                                variant_slacks = dual_slacks if variant in DUAL_VARIANTS and variant != "dual_raw" else (0.0,)
                                for dual_slack in variant_slacks:
                                    for quantization_radius in quantization_radii:
                                        key = (
                                            variant,
                                            int(tile_size),
                                            int(min_leaf_size),
                                            None if forced_leaf_error_cap is None else float(forced_leaf_error_cap),
                                            bool(forced_split_bias_fit),
                                            float(plane_error),
                                            float(dual_slack),
                                            int(quantization_radius),
                                        )
                                        if key in seen_configs:
                                            continue
                                        seen_configs.add(key)
                                        sweep_results.append(
                                            _encode_sst_variant(
                                                args,
                                                front_depth,
                                                second_depth,
                                                variant,
                                                shadow_bias,
                                                plane_error,
                                                dual_slack,
                                                quantization_radius,
                                                tile_size,
                                                min_leaf_size,
                                                forced_leaf_error_cap,
                                                forced_split_bias_fit,
                                            )
                                        )
        sweep_pareto = _compute_sweep_pareto(sweep_results)
        sweep_recommendations = _compute_sweep_recommendations(sweep_results)

    report = {
        "scene": args.scene or "demo",
        "shadow_resolution": scene.static_shadow_resolution,
        "shadow_size": list(scene.static_shadow_size),
        "shadow_resolution_input": _shadow_resolution_label(args.shadow_resolution),
        "variants": variants,
        "settings": {
            "tile_size": max(1, args.sst_tile_size),
            "min_leaf_size": max(1, args.sst_min_leaf_size),
            "plane_error_threshold": max(0.0, args.sst_plane_error_threshold),
            "constant_epsilon": max(0.0, args.sst_constant_epsilon),
            "plane_quantization_search_radius": max(0, args.sst_plane_quantization_search_radius),
            "dual_depth_slack": max(0.0, args.sst_dual_depth_slack),
            "forced_leaf_error_cap": args.sst_forced_leaf_error_cap,
            "forced_split_bias_fit": args.sst_forced_split_bias_fit,
            "shadow_bias": shadow_bias,
            "gpu_decompress": bool(args.sst_gpu_decompress),
        },
        "sweep_results": sweep_results,
        "sweep_pareto": sweep_pareto,
        "sweep_recommendations": sweep_recommendations,
        "timing_seconds": {
            "bake": bake_seconds,
            "readback": readback_seconds,
            "total_without_encode": bake_seconds + readback_seconds,
        },
    }

    if args.sst_debug_output_dir:
        debug_start = time.perf_counter()
        debug_output_dir = Path(args.sst_debug_output_dir)
        debug_metadata = _export_sst_debug_maps(args, debug_output_dir, front_depth, second_depth, shadow_bias)
        report["debug_output"] = {
            "directory": str(debug_output_dir),
            "variants": list(debug_metadata.get("variants", {}).keys()),
            "png_enabled": bool(debug_metadata.get("png_enabled", False)),
            "time_seconds": time.perf_counter() - debug_start,
        }

    print("\n[SST Benchmark]")
    print(f"Scene: {report['scene']}")
    print(f"Bake time:      {bake_seconds:.3f}s")
    print(f"Readback time:  {readback_seconds:.3f}s")
    if report.get("debug_output"):
        debug_output = report["debug_output"]
        print(
            f"Debug maps:     {debug_output['directory']} "
            f"variants={','.join(debug_output['variants'])} "
            f"png={debug_output['png_enabled']} "
            f"time={debug_output['time_seconds']:.3f}s"
        )
    for name, variant_report in variants.items():
        stats = variant_report["stats"]
        print(f"\n[{name}]")
        print(f"Resolution: {stats['width']}x{stats['height']}")
        print(f"Tree depth: {stats['max_tree_depth']}  Traversal steps: {stats['max_traversal_steps']} "
              f"10-bit branches start at level {stats['branch_10bit_start_level']}")
        print(f"Tiles: {stats['tile_count']}, Nodes: {stats['node_count']} "
              f"(branches={stats['branch_node_count']} "
              f"13bit={stats['branch_13bit_node_count']} 10bit={stats['branch_10bit_node_count']}, "
              f"planes={stats['plane_node_count']}, uniform={stats['uniform_plane_node_count']})")
        print(f"Original bytes: {stats['original_bytes']:,}")
        print(f"Unpacked bytes: {stats['encoded_bytes']:,}  Compression: {stats['compression_ratio']:.2f}x")
        print(f"Packed bytes:   {stats['packed_encoded_bytes']:,}  Packed compression: {stats['packed_compression_ratio']:.2f}x "
              f"Decode valid: {stats['packed_decode_valid']}")
        print(
            f"Decomp working: {_packed_decompressed_working_set_bytes(stats):,} bytes  "
            f"Effective ratio: {_packed_decompressed_working_set_ratio(stats):.2f}x "
            f"(includes {_decompressed_depth_bytes(stats):,} byte r32 depth texture)"
        )
        branch_words, plane30_words, plane62_words, total_words = _compact_word_breakdown(stats)
        print(
            f"Packed words:   total={total_words:,} "
            f"branch={branch_words:,} 30bit={plane30_words:,} 62bit={plane62_words:,} "
            f"roots={int(stats.get('compact_tile_root_bytes', _tile_root_bytes(stats))):,} bytes"
        )
        print(f"Packed morton:  valid={stats.get('packed_morton_decode_valid', False)}  "
              f"max delta={stats.get('packed_morton_max_delta_percent', 0.0):.6f}%")
        print(f"Compact/Fixed64 parity: valid={stats.get('compact_fixed64_decode_valid', False)}  "
              f"max delta={stats.get('compact_fixed64_max_delta_percent', 0.0):.6f}%")
        print(f"Fixed64 bytes:  {stats['fixed64_encoded_bytes']:,}  Fixed64 compression: {stats['fixed64_compression_ratio']:.2f}x")
        print(f"Branch offset overflow: compact={stats['compact_branch_offset_overflow_count']} "
              f"fixed64={stats['fixed64_branch_offset_overflow_count']} "
              f"(max compact={stats['max_compact_branch_offset']}, max fixed64={stats['max_fixed64_branch_offset']})")
        print(f"Mean loss:      {stats['mean_error_percent']:.4f}% normalized depth")
        print(f"RMSE loss:      {stats['rmse_error_percent']:.4f}% normalized depth")
        print(f"Max loss:       {stats['max_error_percent']:.4f}% normalized depth")
        print(f"Packed mean:    {stats['packed_mean_error_percent']:.4f}%  "
              f"rmse={stats['packed_rmse_error_percent']:.4f}%  "
              f"max={stats['packed_max_error_percent']:.4f}%  "
              f"leak={stats['packed_leak_pixel_percent']:.4f}%")
        print(f"Packed bias risk (bias={stats['shadow_bias']:.6f}): "
              f"leak>0.5b={stats['packed_leak_over_half_bias_percent']:.4f}%  "
              f"leak>1b={stats['packed_leak_over_full_bias_percent']:.4f}%  "
              f"shadow>0.5b={stats['packed_conservative_over_half_bias_percent']:.4f}%  "
              f"shadow>1b={stats['packed_conservative_over_full_bias_percent']:.4f}%")
        print(f"Packed visibility probes {stats.get('visibility_probe_offsets_in_bias', [])}: "
              f"mismatch={stats.get('packed_visibility_mismatch_percent', 0.0):.4f}%  "
              f"falseLit={stats.get('packed_false_lit_percent', 0.0):.4f}%  "
              f"falseShadow={stats.get('packed_false_shadow_percent', 0.0):.4f}%")
        if "packed_depth_pcf3_mae_percent" in stats:
            print(
                "DepthTexture PCF3 delta: "
                f"SST-PCF3-MAE={stats['packed_depth_pcf3_mae_percent']:.4f}%  "
                f"SST-hard-MAE={stats['packed_hard_vs_depth_pcf3_mae_percent']:.4f}%  "
                f"hard@0B={_pcf3_probe_metric(stats, 0.0, 'hard_vs_pcf3_mae_percent'):.4f}%  "
                f"hard@1B={_pcf3_probe_metric(stats, 1.0, 'hard_vs_pcf3_mae_percent'):.4f}%"
            )
        if "gpu_decompress_valid" in stats:
            print(
                "GPU decompress: "
                f"valid={stats['gpu_decompress_valid']}  "
                f"cpuMaxDelta={stats['gpu_decompress_vs_cpu_max_delta_percent']:.6f}%  "
                f"srcMean={stats['gpu_decompress_vs_source_mean_error_percent']:.4f}%  "
                f"dispatch={variant_report['timing_seconds'].get('gpu_decompress_dispatch', 0.0):.3f}s  "
                f"readback={variant_report['timing_seconds'].get('gpu_decompress_readback', 0.0):.3f}s"
            )
        print(f"Fixed64 mean:   {stats['fixed64_mean_error_percent']:.4f}%  "
              f"rmse={stats['fixed64_rmse_error_percent']:.4f}%  "
              f"max={stats['fixed64_max_error_percent']:.4f}%")
        print(f"Fixed64 leak pixels: {stats['fixed64_leak_pixel_percent']:.4f}%  "
              f"mean leak={stats['fixed64_mean_leak_error_percent']:.4f}%  "
              f"max leak={stats['fixed64_max_leak_error_percent']:.4f}%  "
              f"interval violation={stats['fixed64_dual_interval_violation_percent']:.4f}%")
        print(f"Fixed64 bias risk: leak>0.5b={stats['fixed64_leak_over_half_bias_percent']:.4f}%  "
              f"leak>1b={stats['fixed64_leak_over_full_bias_percent']:.4f}%  "
              f"shadow>0.5b={stats['fixed64_conservative_over_half_bias_percent']:.4f}%  "
              f"shadow>1b={stats['fixed64_conservative_over_full_bias_percent']:.4f}%")
        print(f"Fixed64 visibility: mismatch={stats.get('fixed64_visibility_mismatch_percent', 0.0):.4f}%  "
              f"falseLit={stats.get('fixed64_false_lit_percent', 0.0):.4f}%  "
              f"falseShadow={stats.get('fixed64_false_shadow_percent', 0.0):.4f}%")
        print(f"Lossy pixels:   {stats['lossy_pixel_percent']:.4f}%")
        print(f"Leak pixels:    {stats['leak_pixel_percent']:.4f}%")
        print(f"Conservative pixels: {stats['conservative_pixel_percent']:.4f}%")
        print(f"Dual interval violation: {stats['dual_interval_violation_percent']:.4f}% "
              f"(mean={stats['mean_dual_interval_violation_percent']:.4f}%, "
              f"max={stats['max_dual_interval_violation_percent']:.4f}%)")
        tile_diag = stats.get("tile_diagnostics", {})
        if tile_diag:
            ratio = tile_diag.get("compression_ratio_percentiles", {})
            max_error = tile_diag.get("max_error_percentiles", {})
            leak_bias = tile_diag.get("leak_over_full_bias_percentiles", {})
            print(
                "Tile diag: "
                f"ratio min={ratio.get('min', 0.0):.2f}x p10={ratio.get('p10', 0.0):.2f}x p50={ratio.get('p50', 0.0):.2f}x; "
                f"maxErr p95={max_error.get('p95', 0.0):.4f}% max={max_error.get('max', 0.0):.4f}%; "
                f"leak>bias maxTile={leak_bias.get('max', 0.0):.4f}%"
            )
        print(f"Encode time:    {variant_report['timing_seconds']['encode']:.3f}s")

    if sweep_results:
        print("\n[SST Sweep Recommendations]")
        for recommendation in sweep_recommendations:
            print(
                f"{recommendation['label']}: {recommendation['variant']} "
                f"tile={_candidate_tile_size(recommendation, args.sst_tile_size)} "
                f"leaf={_candidate_min_leaf_size(recommendation, args.sst_min_leaf_size)} "
                f"forceCap={_format_optional_float(recommendation.get('forced_leaf_error_cap'))} "
                f"biasSplit={_format_bool(recommendation.get('forced_split_bias_fit', False))} "
                f"err={recommendation['plane_error_threshold']:.6f} "
                f"slack={recommendation['dual_depth_slack']} "
                f"qRadius={recommendation.get('plane_quantization_search_radius', 0)} "
                f"packed={recommendation['packed_compression_ratio']:.2f}x "
                f"visMis={recommendation['packed_visibility_mismatch_percent']:.4f}% "
                f"pcf3={_format_optional_percent(recommendation.get('packed_depth_pcf3_mae_percent'))} "
                f"falseLit={recommendation['packed_false_lit_percent']:.4f}% "
                f"leak>1b={recommendation['packed_leak_over_full_bias_percent']:.4f}% "
                f"shadow>1b={recommendation.get('packed_conservative_over_full_bias_percent', 0.0):.4f}% "
                f"within1b={recommendation.get('packed_abs_error_within_1_bias_percent', 0.0):.4f}%"
            )

        print("\n[SST Sweep Pareto]")
        for index, result in enumerate(sweep_pareto[:16], start=1):
            stats = result["stats"]
            print(
                f"{index:02d}. {result['variant']} "
                f"tile={_candidate_tile_size(result, args.sst_tile_size)} "
                f"leaf={_candidate_min_leaf_size(result, args.sst_min_leaf_size)} "
                f"forceCap={_format_optional_float(result.get('forced_leaf_error_cap'))} "
                f"biasSplit={_format_bool(result.get('forced_split_bias_fit', False))} "
                f"err={result['plane_error_threshold']:.6f} "
                f"slack={result['dual_depth_slack']} "
                f"qRadius={result.get('plane_quantization_search_radius', 0)} "
                f"packed={stats['packed_compression_ratio']:.2f}x "
                f"mean={stats['packed_mean_error_percent']:.4f}% "
                f"rmse={stats['packed_rmse_error_percent']:.4f}% "
                f"max={stats['packed_max_error_percent']:.4f}% "
                f"visMis={stats.get('packed_visibility_mismatch_percent', 0.0):.4f}% "
                f"leak>1b={stats['packed_leak_over_full_bias_percent']:.4f}% "
                f"shadow>1b={stats['packed_conservative_over_full_bias_percent']:.4f}%"
            )

    if args.benchmark_output:
        output_path = Path(args.benchmark_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Benchmark JSON saved to {output_path}")

    if args.benchmark_markdown_output:
        markdown_path = Path(args.benchmark_markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(build_sst_markdown_report(report), encoding="utf-8")
        print(f"Benchmark Markdown saved to {markdown_path}")


def main() -> None:
    args = parse_args()

    if args.sst_report_inputs:
        reports = _load_sst_reports(_parse_path_list(args.sst_report_inputs))
        markdown = build_sst_multi_markdown_report(reports)
        if args.benchmark_markdown_output:
            output_path = Path(args.benchmark_markdown_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"SST comparison Markdown saved to {output_path}")
        else:
            print(markdown)
        return

    if args.benchmark_sst:
        run_sst_benchmark(args)
        return

    if args.static_shadow_compare_report_inputs:
        reports = _load_static_shadow_compare_reports(_parse_path_list(args.static_shadow_compare_report_inputs))
        markdown = build_static_shadow_compare_multi_markdown(reports)
        if args.static_shadow_compare_markdown_output:
            output_path = Path(args.static_shadow_compare_markdown_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"Static shadow compare summary Markdown saved to {output_path}")
        else:
            print(markdown)
        return

    if args.static_shadow_compare_modes:
        run_static_shadow_compare(args)
        return

    from app import App, AppConfig
    from path_tracing_renderer import PathTracingRenderer

    static_shadow_mode = _parse_static_shadow_mode(args.static_shadow_mode)
    enabled_extensions = _enabled_extensions_from_args(args, static_shadow_mode)
    renderer = PathTracingRenderer()

    base_config = AppConfig()
    output_path = Path(args.output) if args.output else base_config.headless_output

    config = AppConfig(
        width=args.width,
        height=args.height,
        headless=args.headless,
        headless_frame_count=args.frames,
        headless_output=output_path,
        vsync=args.vsync,
        srgb_output=not args.no_srgb,
        camera_move_test=args.camera_move_test,
        scene_path=args.scene,
        enabled_extensions=enabled_extensions,
        static_shadow_mode=static_shadow_mode,
        static_shadow_resolution=args.shadow_resolution,
        sst_encoder_backend=args.sst_encoder,
        static_shadow_mask_mode=args.static_shadow_mask,
        static_shadow_mask_threshold=max(0.0, float(args.static_shadow_mask_threshold)),
        static_shadow_mask_bootstrap_passes=max(0, min(4, int(args.static_shadow_mask_bootstrap_passes))),
    )

    app = App(config=config)
    if app.extensions.has("static_shadow_sst"):
        _apply_runtime_sst_options(app.scene, args)
    app.set_renderer(renderer)
    app.main_loop()


if __name__ == "__main__":
    main()
