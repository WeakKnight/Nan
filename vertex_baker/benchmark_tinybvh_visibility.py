from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from model import load_gltf_model
from pmr_visibility_reference import sample_model_surface_pmr
from tinybvh_visibility_baker import flatten_model_geometry
from vertex_baking_utils import tinybvh_has_bvh8_cpu, trace_pmr_visibility_sh_tinybvh


DEFAULT_ASSET = Path(__file__).resolve().parent / "glTF" / "Lantern.gltf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PMR visibility tracing with TinyBVH and HWRT.")
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--samples-per-triangle", type=int, default=4)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--self-bias", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--layouts",
        choices=("all", "bvh", "bvh8"),
        default="bvh",
        help="BVH is the reference path; BVH8 is an experimental surface-ray comparison.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--hwrt", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="Optional machine-readable result path.")
    return parser.parse_args()


def median(values) -> float:
    return float(statistics.median(values))


def comparison(reference: np.ndarray, candidate: np.ndarray, ray_count: int) -> dict[str, float | int]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    sh0_per_visible_ray = 0.282094791774 * (4.0 * np.pi / float(ray_count))
    reference_counts = np.rint(reference[:, 0] / sh0_per_visible_ray).astype(np.int64)
    candidate_counts = np.rint(candidate[:, 0] / sh0_per_visible_ray).astype(np.int64)
    count_difference = np.abs(candidate_counts - reference_counts)
    absolute_difference = np.abs(difference)
    return {
        "mean_abs_sh": float(np.mean(absolute_difference)),
        "p99_abs_sh": float(np.quantile(absolute_difference, 0.99)),
        "p999_abs_sh": float(np.quantile(absolute_difference, 0.999)),
        "max_abs_sh": float(np.max(absolute_difference)),
        "rms_sh": float(np.sqrt(np.mean(difference * difference))),
        "samples_with_visible_count_difference": int(np.count_nonzero(count_difference)),
        "samples_with_exact_visible_count": int(np.count_nonzero(count_difference == 0)),
        "max_visible_count_difference": int(np.max(count_difference)),
    }


def run_tinybvh(
    positions,
    indices,
    sampling,
    *,
    layout: str,
    ray_count: int,
    max_distance: float,
    self_bias: float,
    thread_count: int,
    warmup: int,
    repeat: int,
):
    result = None
    for _ in range(warmup):
        trace_pmr_visibility_sh_tinybvh(
            positions,
            indices,
            sampling.samples.positions,
            sampling.sample_normals,
            ray_count=ray_count,
            max_distance=max_distance,
            self_bias=self_bias,
            thread_count=thread_count,
            layout=layout,
        )

    build_times = []
    trace_times = []
    wall_times = []
    stats = None
    for _ in range(repeat):
        begin = perf_counter()
        result, stats = trace_pmr_visibility_sh_tinybvh(
            positions,
            indices,
            sampling.samples.positions,
            sampling.sample_normals,
            ray_count=ray_count,
            max_distance=max_distance,
            self_bias=self_bias,
            thread_count=thread_count,
            layout=layout,
        )
        wall_times.append((perf_counter() - begin) * 1000.0)
        build_times.append(stats.build_milliseconds)
        trace_times.append(stats.trace_milliseconds)

    trace_milliseconds = median(trace_times)
    total_ray_count = int(sampling.samples.positions.shape[0]) * ray_count
    summary = {
        "backend": "tinybvh",
        "layout": layout,
        "thread_count": stats.thread_count,
        "build_milliseconds_median": median(build_times),
        "trace_milliseconds_median": trace_milliseconds,
        "wall_milliseconds_median": median(wall_times),
        "million_rays_per_second": total_ray_count / (trace_milliseconds * 1000.0),
        "visible_fraction": stats.visible_fraction,
    }
    return result, summary


def run_hwrt(model, sampling, args):
    import slangpy as spy

    from hwrt_visibility_baker import HWRTPMRVisibilitySHSampler, HWRTModelScene

    project_dir = Path(__file__).resolve().parents[1]
    device = spy.Device(
        enable_debug_layers=False,
        compiler_options={
            "include_paths": [project_dir],
            "defines": {"USE_RAYTRACING_PIPELINE": "0", "HEADLESS_MODE": "1"},
        },
    )
    setup_begin = perf_counter()
    scene = HWRTModelScene(device, model)
    sampler = HWRTPMRVisibilitySHSampler(device)
    setup_milliseconds = (perf_counter() - setup_begin) * 1000.0

    for _ in range(max(0, int(args.warmup))):
        sampler.sample(
            scene,
            sampling.samples.positions,
            sampling.sample_normals,
            ray_count=args.rays,
            max_distance=args.max_distance,
            self_bias=args.self_bias,
        )
    times = []
    result = None
    for _ in range(max(1, int(args.repeat))):
        begin = perf_counter()
        result = sampler.sample(
            scene,
            sampling.samples.positions,
            sampling.sample_normals,
            ray_count=args.rays,
            max_distance=args.max_distance,
            self_bias=args.self_bias,
        )
        times.append((perf_counter() - begin) * 1000.0)
    trace_milliseconds = median(times)
    total_ray_count = int(sampling.samples.positions.shape[0]) * args.rays
    return result, {
        "backend": "hwrt",
        "setup_milliseconds": setup_milliseconds,
        "trace_milliseconds_median": trace_milliseconds,
        "million_rays_per_second": total_ray_count / (trace_milliseconds * 1000.0),
    }


def main() -> None:
    args = parse_args()
    args.samples_per_triangle = max(1, int(args.samples_per_triangle))
    args.rays = max(1, int(args.rays))
    args.repeat = max(1, int(args.repeat))
    args.warmup = max(0, int(args.warmup))
    model = load_gltf_model(args.asset)
    sampling_begin = perf_counter()
    sampling = sample_model_surface_pmr(model, samples_per_triangle=args.samples_per_triangle)
    sampling_milliseconds = (perf_counter() - sampling_begin) * 1000.0
    positions, indices = flatten_model_geometry(model)
    total_ray_count = int(sampling.samples.positions.shape[0]) * args.rays
    print(
        f"scene: {positions.shape[0]} vertices, {indices.shape[0]} source triangles, "
        f"{sampling.proxy.triangles.shape[0]} proxy triangles"
    )
    print(
        f"work: {sampling.samples.positions.shape[0]} samples x {args.rays} rays = "
        f"{total_ray_count:,} rays; sampling {sampling_milliseconds:.2f} ms"
    )

    layouts = [args.layouts]
    if args.layouts == "all":
        layouts = ["bvh"]
        if tinybvh_has_bvh8_cpu():
            layouts.append("bvh8")
    elif args.layouts == "bvh8" and not tinybvh_has_bvh8_cpu():
        raise RuntimeError("BVH8_CPU is not available in this native build")

    outputs = {}
    summaries = []
    for layout in layouts:
        output, summary = run_tinybvh(
            positions,
            indices,
            sampling,
            layout=layout,
            ray_count=args.rays,
            max_distance=args.max_distance,
            self_bias=args.self_bias,
            thread_count=max(0, int(args.threads)),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        outputs[f"tinybvh_{layout}"] = output
        summaries.append(summary)
        print(
            f"TinyBVH {layout:4s}: build {summary['build_milliseconds_median']:.2f} ms, "
            f"trace {summary['trace_milliseconds_median']:.2f} ms, "
            f"{summary['million_rays_per_second']:.2f} Mray/s, "
            f"wall {summary['wall_milliseconds_median']:.2f} ms, "
            f"threads {summary['thread_count']}"
        )

    reference_name = "tinybvh_bvh" if "bvh" in layouts else f"tinybvh_{layouts[0]}"
    reference = outputs[reference_name]
    comparisons = {}
    for name, output in outputs.items():
        if name != reference_name:
            comparisons[f"{name}_vs_{reference_name}"] = comparison(reference, output, args.rays)

    if args.hwrt:
        hwrt_output, hwrt_summary = run_hwrt(model, sampling, args)
        summaries.append(hwrt_summary)
        comparisons[f"hwrt_vs_{reference_name}"] = comparison(reference, hwrt_output, args.rays)
        print(
            f"HWRT        : setup {hwrt_summary['setup_milliseconds']:.2f} ms, "
            f"trace {hwrt_summary['trace_milliseconds_median']:.2f} ms, "
            f"{hwrt_summary['million_rays_per_second']:.2f} Mray/s"
        )

    for name, metrics in comparisons.items():
        print(
            f"compare {name}: mean |SH| {metrics['mean_abs_sh']:.3e}, "
            f"max {metrics['max_abs_sh']:.3e}, "
            f"count mismatches {metrics['samples_with_visible_count_difference']}, "
            f"max ray delta {metrics['max_visible_count_difference']}"
        )

    payload = {
        "asset": str(args.asset.resolve()),
        "source_vertex_count": int(positions.shape[0]),
        "source_triangle_count": int(indices.shape[0]),
        "proxy_triangle_count": int(sampling.proxy.triangles.shape[0]),
        "sample_count": int(sampling.samples.positions.shape[0]),
        "rays_per_sample": args.rays,
        "total_ray_count": total_ray_count,
        "sampling_milliseconds": sampling_milliseconds,
        "runs": summaries,
        "comparisons": comparisons,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
