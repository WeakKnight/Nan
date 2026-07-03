import slangpy as spy
from pathlib import Path

from render_data import RenderData


class ScreenShadowMaskPass:
    MODES = ("off", "full", "adaptive", "adaptive-wave")

    def __init__(self, device: spy.Device):
        self.device = device
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, entry_point: str):
        pipeline = self._pipelines.get(entry_point)
        if pipeline is None:
            shader_path = Path(__file__).parent / "shaders" / "screen_shadow_mask.slang"
            program = self.device.load_program(str(shader_path), [entry_point])
            pipeline = self.device.create_compute_pipeline(program)
            self._pipelines[entry_point] = pipeline
        return pipeline

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        scene,
        render_data: RenderData,
        width: int,
        height: int,
        mode: str,
        threshold: float,
        bootstrap_passes: int,
    ) -> spy.Texture | None:
        mode = mode.strip().lower()
        if mode == "off":
            return None
        if mode not in self.MODES:
            raise ValueError(f"Unsupported static shadow mask mode '{mode}'")

        width = max(1, int(width))
        height = max(1, int(height))
        grid_w = (width + 3) // 4
        grid_h = (height + 3) // 4
        super_grid_w = (grid_w + 1) // 2
        super_grid_h = (grid_h + 1) // 2

        shadow_mask = render_data.get_texture(
            "path_tracing_renderer.static_shadow_mask",
            width=width,
            height=height,
            format=spy.Format.r32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="static_shadow_mask",
        )

        with command_encoder.begin_compute_pass() as pass_encoder:
            if mode == "full":
                self._dispatch(
                    pass_encoder,
                    "full_main",
                    scene,
                    shadow_mask,
                    width,
                    height,
                    grid_w,
                    grid_h,
                    super_grid_w,
                    super_grid_h,
                    threshold,
                    bootstrap_passes,
                    [width, height, 1],
                )
                return shadow_mask

            pass_prefix = "wave_" if mode == "adaptive-wave" else ""
            self._dispatch(
                pass_encoder,
                "pass0",
                scene,
                shadow_mask,
                width,
                height,
                grid_w,
                grid_h,
                super_grid_w,
                super_grid_h,
                threshold,
                bootstrap_passes,
                [grid_w, grid_h, 1],
            )
            self._dispatch(
                pass_encoder,
                f"{pass_prefix}pass1",
                scene,
                shadow_mask,
                width,
                height,
                grid_w,
                grid_h,
                super_grid_w,
                super_grid_h,
                threshold,
                bootstrap_passes,
                [super_grid_w * super_grid_h, 1, 1] if pass_prefix else [super_grid_w, super_grid_h, 1],
            )
            self._dispatch(
                pass_encoder,
                f"{pass_prefix}pass2",
                scene,
                shadow_mask,
                width,
                height,
                grid_w,
                grid_h,
                super_grid_w,
                super_grid_h,
                threshold,
                bootstrap_passes,
                [super_grid_w * super_grid_h, 1, 1] if pass_prefix else [super_grid_w, super_grid_h, 1],
            )
            self._dispatch(
                pass_encoder,
                f"{pass_prefix}pass3",
                scene,
                shadow_mask,
                width,
                height,
                grid_w,
                grid_h,
                super_grid_w,
                super_grid_h,
                threshold,
                bootstrap_passes,
                [grid_w * grid_h, 1, 1] if pass_prefix else [grid_w, grid_h, 1],
            )
            self._dispatch(
                pass_encoder,
                f"{pass_prefix}pass4",
                scene,
                shadow_mask,
                width,
                height,
                grid_w,
                grid_h,
                super_grid_w,
                super_grid_h,
                threshold,
                bootstrap_passes,
                [grid_w * grid_h, 1, 1] if pass_prefix else [grid_w, grid_h, 1],
            )

        return shadow_mask

    def _dispatch(
        self,
        pass_encoder,
        entry_point: str,
        scene,
        shadow_mask: spy.Texture,
        width: int,
        height: int,
        grid_w: int,
        grid_h: int,
        super_grid_w: int,
        super_grid_h: int,
        threshold: float,
        bootstrap_passes: int,
        thread_count: list[int],
    ) -> None:
        shader_object = pass_encoder.bind_pipeline(self._pipeline(entry_point))
        cursor = spy.ShaderCursor(shader_object)
        cursor.g_shadow_mask = shadow_mask
        cursor.g_screen_size = spy.uint2(width, height)
        cursor.g_screen_grid_dim = spy.uint2(grid_w, grid_h)
        cursor.g_super_grid_dim = spy.uint2(super_grid_w, super_grid_h)
        cursor.g_adaptive_threshold = max(0.0, float(threshold))
        cursor.g_adaptive_bootstrap_passes = max(0, min(4, int(bootstrap_passes)))
        scene.bind(cursor.g_scene)
        pass_encoder.dispatch(thread_count=thread_count)
