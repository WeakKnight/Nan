from __future__ import annotations

import argparse
import math
from pathlib import Path

from model import load_gltf_model
from software_renderer import render_unlit_preview
from surface_sampler import sample_model_surface
from vertex_color_baker import bake_model_vertex_colors, sample_base_color_values
from visibility_baker import (
    bake_visibility_cones_pmr_python,
    bake_visibility_cones_python,
    save_visibility_npz,
    vertex_visibility_preview_values,
)


DEFAULT_ASSET = Path(__file__).resolve().parent / "glTF" / "Lantern.gltf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "lantern_samples.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone vertex baking preview tool.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=DEFAULT_ASSET,
        help="glTF asset path. Defaults to vertex_baker/glTF/Lantern.gltf.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output preview PNG.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Preview width.")
    parser.add_argument("--height", type=int, default=720, help="Preview height.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=20_000,
        help="Global sample count for sample/vertex-color modes and legacy visibility only.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Deterministic surface sampling seed.")
    parser.add_argument("--point-radius", type=int, default=2, help="Sample point radius in pixels.")
    parser.add_argument(
        "--mode",
        choices=("samples", "texture", "vertex-color", "interactive", "visibility"),
        default="samples",
        help="Preview mode: sample overlay, texture unlit, baked vertex color PNG, native SlangPy viewer, or visibility reference bake.",
    )
    parser.add_argument(
        "--viewer-data",
        type=Path,
        default=None,
        help="Existing visibility .npz to inspect in interactive mode without rebaking.",
    )
    parser.add_argument(
        "--viewer-max-frames",
        type=int,
        default=0,
        help="Exit the native viewer after N frames; zero runs until the window is closed.",
    )
    parser.add_argument(
        "--viewer-capture-on-exit",
        action="store_true",
        help="Write the native viewer output to --output when the viewer exits.",
    )
    parser.add_argument(
        "--envmap",
        type=Path,
        default=Path(__file__).resolve().parent / "bloem_field_sunrise_2k.hdr",
        help="HDR environment map used by the native PBR viewer.",
    )
    parser.add_argument("--env-exposure", type=float, default=0.0, help="PBR display exposure in EV.")
    parser.add_argument(
        "--env-rotation-degrees",
        type=float,
        default=0.0,
        help="Environment yaw rotation in degrees.",
    )
    parser.add_argument(
        "--no-apply-visibility",
        action="store_true",
        help="Do not apply loaded vertex visibility as indirect-light AO.",
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=0.0,
        help="Native least-squares regularization weight for --mode vertex-color.",
    )
    parser.add_argument(
        "--vertex-anchor-weight",
        type=int,
        default=8,
        help="Number of one-hot UV texture samples added per vertex to stabilize local vertex colors.",
    )
    parser.add_argument(
        "--vertex-anchor-max-error",
        type=float,
        default=0.08,
        help="Maximum baked color deviation from each vertex's own UV texture sample.",
    )
    parser.add_argument(
        "--build-native",
        action="store_true",
        help="Build the native vertex_baking_utils library before baking.",
    )
    parser.add_argument(
        "--visibility-algorithm",
        choices=("pmr", "legacy"),
        default="pmr",
        help="Visibility pipeline. 'pmr' mirrors the original Unity tool; 'legacy' keeps the earlier cone approximation.",
    )
    parser.add_argument(
        "--visibility-trace-backend",
        choices=("hwrt", "tinybvh", "cpu"),
        default="hwrt",
        help="PMR visibility tracing backend: Slang HWRT, native TinyBVH CPU, or brute-force Python.",
    )
    parser.add_argument(
        "--visibility-tinybvh-layout",
        choices=("auto", "bvh", "bvh8"),
        default="auto",
        help=(
            "TinyBVH layout. Auto uses reference-safe BVH; BVH8 is experimental and requires "
            "a native build configured with VBAKE_TINYBVH_AVX2=ON."
        ),
    )
    parser.add_argument(
        "--visibility-cpu-threads",
        type=int,
        default=0,
        help="TinyBVH worker count; zero uses all logical CPU threads.",
    )
    parser.add_argument(
        "--visibility-samples-per-triangle",
        "--visibility-triangle-samples",
        dest="visibility_samples_per_triangle",
        type=int,
        default=256,
        help="Fixed PMR low-discrepancy samples generated on every proxy triangle.",
    )
    parser.add_argument(
        "--visibility-rays",
        type=int,
        default=512,
        help="Rays per surface sample. PMR samples the full sphere.",
    )
    parser.add_argument(
        "--visibility-min-samples-per-mesh",
        type=int,
        default=0,
        help="Legacy-only minimum surface samples assigned to each mesh.",
    )
    parser.add_argument(
        "--visibility-max-distance",
        type=float,
        default=0.5,
        help="Maximum visibility ray distance in model units.",
    )
    parser.add_argument(
        "--visibility-self-bias",
        type=float,
        default=0.001,
        help="Ray origin offset along the proxy triangle normal.",
    )
    parser.add_argument(
        "--visibility-edge-regularization",
        type=float,
        default=0.05,
        help="PMR edge-gradient regularization weight.",
    )
    parser.add_argument(
        "--visibility-proxy-voxel-mm",
        type=float,
        default=0.1,
        help="PMR proxy vertex hashing resolution in millimeters.",
    )
    parser.add_argument(
        "--visibility-proxy-ignore-normals",
        action="store_true",
        help="Merge PMR proxy vertices by position only.",
    )
    parser.add_argument(
        "--visibility-fit-backend",
        choices=("python", "native", "cpp", "cxx"),
        default="native",
        help="PMR SH16 solve/cone conversion or legacy cone fitting backend.",
    )
    parser.add_argument(
        "--visibility-show-samples",
        action="store_true",
        help="Overlay PMR surface sample locations on the visibility preview.",
    )
    parser.add_argument(
        "--visibility-show-cones",
        action="store_true",
        help="Overlay the fitted ray cone at every model vertex.",
    )
    parser.add_argument(
        "--visibility-cone-length",
        type=float,
        default=0.0,
        help="World-space cone ray length; zero uses --visibility-max-distance.",
    )
    parser.add_argument(
        "--visibility-cone-rim-segments",
        type=int,
        default=12,
        help="Circular resolution used for each vertex cone outline and surface.",
    )
    parser.add_argument(
        "--visibility-cone-xray",
        action="store_true",
        help="Draw cone geometry through the model instead of depth testing it.",
    )
    parser.add_argument(
        "--hide-samples",
        action="store_true",
        help="Compatibility alias for --mode texture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cone_display_length = float(args.visibility_cone_length)
    if not math.isfinite(cone_display_length) or cone_display_length <= 0.0:
        trace_distance = float(args.visibility_max_distance)
        cone_display_length = trace_distance if math.isfinite(trace_distance) and trace_distance > 0.0 else 0.0
    model = load_gltf_model(args.asset)
    mode = "texture" if args.hide_samples else args.mode
    samples = None
    vertex_colors = None
    visibility_cones = None

    if mode == "samples":
        samples = sample_model_surface(model, args.sample_count, args.seed)
    elif mode == "interactive" and args.viewer_data is not None:
        from slang_viewer import load_visibility_view_data

        vertex_colors, visibility_cones = load_visibility_view_data(model, args.viewer_data)
    elif mode in ("vertex-color", "interactive"):
        samples_for_bake = sample_model_surface(model, args.sample_count, args.seed)
        sample_values = sample_base_color_values(model, samples_for_bake)
        vertex_colors = bake_model_vertex_colors(
            model,
            samples_for_bake,
            sample_values,
            regularization_weight=max(0.0, float(args.regularization_weight)),
            build_native_first=bool(args.build_native),
            vertex_anchor_weight=max(0, int(args.vertex_anchor_weight)),
            vertex_anchor_max_error=max(0.0, float(args.vertex_anchor_max_error)),
        )
    elif mode == "visibility":
        if args.visibility_algorithm == "pmr":
            pmr_options = {
                "samples_per_triangle": max(1, int(args.visibility_samples_per_triangle)),
                "visibility_ray_count": max(1, int(args.visibility_rays)),
                "ray_length": float(args.visibility_max_distance),
                "self_bias": max(0.0, float(args.visibility_self_bias)),
                "edge_regularization": max(0.0, float(args.visibility_edge_regularization)),
                "proxy_voxel_size_mm": max(1e-6, float(args.visibility_proxy_voxel_mm)),
                "proxy_compare_normals": not bool(args.visibility_proxy_ignore_normals),
                "fit_backend": args.visibility_fit_backend,
                "build_native_first": bool(args.build_native),
            }
            if args.visibility_trace_backend == "hwrt":
                import slangpy as spy

                from hwrt_visibility_baker import bake_pmr_visibility_cones_hwrt

                project_dir = Path(__file__).resolve().parents[1]
                device = spy.Device(
                    enable_debug_layers=False,
                    compiler_options={
                        "include_paths": [project_dir],
                        "defines": {"USE_RAYTRACING_PIPELINE": "0", "HEADLESS_MODE": "1"},
                    },
                )
                visibility_result = bake_pmr_visibility_cones_hwrt(device, model, **pmr_options)
            elif args.visibility_trace_backend == "tinybvh":
                from tinybvh_visibility_baker import bake_pmr_visibility_cones_tinybvh

                visibility_result = bake_pmr_visibility_cones_tinybvh(
                    model,
                    **pmr_options,
                    thread_count=max(0, int(args.visibility_cpu_threads)),
                    layout=args.visibility_tinybvh_layout,
                )
            else:
                visibility_result = bake_visibility_cones_pmr_python(model, **pmr_options)
        else:
            visibility_result = bake_visibility_cones_python(
                model,
                sample_count=max(0, int(args.sample_count)),
                surface_seed=int(args.seed),
                min_samples_per_mesh=max(0, int(args.visibility_min_samples_per_mesh)),
                visibility_ray_count=max(1, int(args.visibility_rays)),
                max_distance=float(args.visibility_max_distance),
                self_bias=max(0.0, float(args.visibility_self_bias)),
                fit_backend=args.visibility_fit_backend,
                regularization_weight=max(0.0, float(args.regularization_weight)),
                build_native_first=bool(args.build_native),
            )
        if visibility_result.trace_statistics is not None:
            stats = visibility_result.trace_statistics
            print(
                f"TinyBVH {stats['layout']} ({stats['thread_count']} threads): "
                f"build {stats['build_milliseconds']:.2f} ms, "
                f"trace {stats['trace_milliseconds']:.2f} ms, "
                f"{stats['million_rays_per_second']:.2f} Mray/s"
            )
        if args.output.suffix.lower() == ".npz":
            save_visibility_npz(visibility_result, args.output)
            print(f"Loaded {len(model.meshes)} meshes from {args.asset}")
            print(f"Wrote visibility reference data to {args.output}")
            return
        vertex_colors = vertex_visibility_preview_values(visibility_result, model)
        visibility_cones = visibility_result.vertex_cones if args.visibility_show_cones else None
        samples = visibility_result.samples if args.visibility_show_samples else None
        npz_output = args.output.with_suffix(".visibility.npz")
        save_visibility_npz(visibility_result, npz_output)

    if mode == "interactive":
        from slang_viewer import run_slang_viewer

        print(f"Loaded {len(model.meshes)} meshes from {args.asset}")
        print("Starting native SlangPy viewer")
        run_slang_viewer(
            model,
            vertex_colors,
            visibility_cones,
            width=max(1, int(args.width)),
            height=max(1, int(args.height)),
            cone_length=cone_display_length,
            screenshot_path=args.output,
            max_frames=max(0, int(args.viewer_max_frames)),
            capture_on_exit=bool(args.viewer_capture_on_exit),
            environment_path=args.envmap,
            exposure=float(args.env_exposure),
            environment_rotation=float(args.env_rotation_degrees),
            apply_visibility=not bool(args.no_apply_visibility),
        )
        return

    render_unlit_preview(
        model,
        samples,
        args.output,
        width=max(1, int(args.width)),
        height=max(1, int(args.height)),
        point_radius=max(1, int(args.point_radius)),
        vertex_colors=vertex_colors,
        visibility_cones=visibility_cones,
        visibility_cone_length=cone_display_length,
        visibility_cone_rim_segments=max(3, int(args.visibility_cone_rim_segments)),
        visibility_cone_xray=bool(args.visibility_cone_xray),
    )
    sample_text = 0 if samples is None else samples.positions.shape[0]
    print(f"Loaded {len(model.meshes)} meshes from {args.asset}")
    if mode == "visibility":
        print(f"Wrote {args.output} in visibility mode")
        if visibility_result.samples_per_triangle is not None and visibility_result.proxy_triangle_count is not None:
            print(
                "Baked "
                f"{visibility_result.proxy_triangle_count} triangles x "
                f"{visibility_result.samples_per_triangle} samples/triangle = "
                f"{visibility_result.samples.positions.shape[0]} visibility surface samples"
            )
        else:
            print(f"Baked {visibility_result.samples.positions.shape[0]} legacy visibility surface samples")
        if visibility_cones is not None:
            print(f"Drew {sum(int(cones.shape[0]) for cones in visibility_cones)} fitted vertex ray cones")
        print(f"Wrote visibility reference data to {npz_output}")
    else:
        print(f"Wrote {args.output} in {mode} mode with {sample_text} visible samples")


if __name__ == "__main__":
    main()
