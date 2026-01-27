import argparse
from pathlib import Path

from path_tracing_renderer import PathTracingRenderer
from app import App, AppConfig


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    )

    app = App(config=config)
    app.set_renderer(renderer)
    app.main_loop()


if __name__ == "__main__":
    main()
