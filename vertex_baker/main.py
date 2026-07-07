from __future__ import annotations

import argparse
from pathlib import Path

from interactive_viewer import export_vertex_color_viewer
from model import load_gltf_model
from software_renderer import render_unlit_preview
from surface_sampler import sample_model_surface
from vertex_color_baker import bake_model_vertex_colors, sample_base_color_values


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
    parser.add_argument("--sample-count", type=int, default=20_000, help="Surface sample count.")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic surface sampling seed.")
    parser.add_argument("--point-radius", type=int, default=2, help="Sample point radius in pixels.")
    parser.add_argument(
        "--mode",
        choices=("samples", "texture", "vertex-color", "interactive"),
        default="samples",
        help="Preview mode: sample overlay, texture unlit, baked vertex color PNG, or interactive vertex color HTML.",
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
        "--hide-samples",
        action="store_true",
        help="Compatibility alias for --mode texture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_gltf_model(args.asset)
    mode = "texture" if args.hide_samples else args.mode
    samples = None
    vertex_colors = None

    if mode == "samples":
        samples = sample_model_surface(model, args.sample_count, args.seed)
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

    if mode == "interactive":
        output = args.output
        if output.suffix.lower() not in (".html", ".htm"):
            output = output.with_suffix(".html")
        export_vertex_color_viewer(model, vertex_colors, output)
        print(f"Loaded {len(model.meshes)} meshes from {args.asset}")
        print(f"Wrote interactive vertex color viewer to {output}")
        return

    render_unlit_preview(
        model,
        samples,
        args.output,
        width=max(1, int(args.width)),
        height=max(1, int(args.height)),
        point_radius=max(1, int(args.point_radius)),
        vertex_colors=vertex_colors,
    )
    sample_text = 0 if samples is None else samples.positions.shape[0]
    print(f"Loaded {len(model.meshes)} meshes from {args.asset}")
    print(f"Wrote {args.output} in {mode} mode with {sample_text} visible samples")


if __name__ == "__main__":
    main()
