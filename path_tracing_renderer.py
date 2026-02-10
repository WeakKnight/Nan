from slangpy import Device


import slangpy as spy
from scene import Scene
from tone_mapper import ToneMapper
from accumulator import Accumulator
from temporal_accumulator import TemporalAccumulator
from path_tracer import PathTracer
from render_data import RenderData

class PathTracingRenderer:
    def initialize(self, device: spy.Device, scene: Scene):
        self.device: Device = device
        self.scene: Scene = scene
        self.path_tracer: PathTracer = PathTracer(device, scene)
        self.accumulator: Accumulator = Accumulator(device, resource_key="path_tracing_renderer.accumulator_history")
        self.temporal_accumulator: TemporalAccumulator = TemporalAccumulator(device, resource_key="path_tracing_renderer.taa")
        self.tone_mapper: ToneMapper = ToneMapper(device)

        self.render_texture: spy.Texture | None = None  # type: ignore (assigned during render)
        self.accum_texture: spy.Texture | None = None  # type: ignore (assigned during render)

        scene.event_distpacher.subscribe("camera_move", self.on_camera_move)

        self.reset_accumulator = True
        self.use_accum_check_box: spy.ui.CheckBox | None = None  # Default: use accumulation (can be overridden by UI)
        self.use_taa_check_box: spy.ui.CheckBox | None = None    # Default: use TAA (can be overridden by UI)

    def on_camera_move(self, data):
        self.reset_accumulator = True

    def render(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
        frame: int,
        device: spy.Device,
        scene: Scene,
        render_data: RenderData,
    ):
        w = output.width
        h = output.height
        tex_usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access

        render_texture = render_data.get_texture(
            "path_tracing_renderer.render_texture",
            width=w, height=h,
            format=spy.Format.rgba32_float,
            usage=tex_usage,
            label="render_texture",
        )
        accum_texture = render_data.get_texture(
            "path_tracing_renderer.accum_texture",
            width=w, height=h,
            format=spy.Format.rgba32_float,
            usage=tex_usage,
            label="accum_texture",
        )
        motion_texture = render_data.get_texture(
            "path_tracing_renderer.motion_texture",
            width=w, height=h,
            format=spy.Format.rgba32_float,
            usage=tex_usage,
            label="motion_texture",
        )
        viewz_texture = render_data.get_texture(
            "path_tracing_renderer.viewz_texture",
            width=w, height=h,
            format=spy.Format.r32_float,
            usage=tex_usage,
            label="viewz_texture",
        )

        self.render_texture = render_texture
        self.accum_texture = accum_texture

        # Path trace: outputs color, motion, viewZ
        self.path_tracer.execute(command_encoder, render_texture, frame,
                                 motion=motion_texture, viewz=viewz_texture)

        use_taa = self._get_use_taa()
        use_accum = self._get_use_accum()

        if use_taa:
            # TAA handles camera motion via motion vectors — never reset on camera move.
            # Only reset on the very first frame (initialization).
            taa_output = self.temporal_accumulator.execute(
                command_encoder,
                render_data,
                render_texture,
                motion_texture,
                reset=False,
            )
            self.tone_mapper.execute(command_encoder, taa_output, output)
        elif use_accum:
            # Simple frame-count accumulation
            self.accumulator.execute(
                command_encoder,
                render_data,
                render_texture,
                accum_texture,
                self.reset_accumulator,
            )
            self.tone_mapper.execute(command_encoder, accum_texture, output)
        else:
            # No accumulation
            self.tone_mapper.execute(command_encoder, render_texture, output)

        self.reset_accumulator = False

    def _get_use_accum(self) -> bool:
        """Get use_accum value, handling both bool and UI checkbox."""
        if self.use_accum_check_box is None:
            return True
        return self.use_accum_check_box.value

    def _get_use_taa(self) -> bool:
        """Get use_taa value, handling both bool and UI checkbox."""
        if self.use_taa_check_box is None:
            return False  # Default off; simple accum is default
        return self.use_taa_check_box.value

    def setup_ui(self, ui_context: spy.ui.Context, ui_window: spy.ui.Window):
        self.use_accum_check_box = spy.ui.CheckBox(ui_window, 'Use Accum')
        self.use_taa_check_box = spy.ui.CheckBox(ui_window, 'Use TAA')

        taa = self.temporal_accumulator

        taa_clamp_cb = spy.ui.CheckBox(ui_window, 'TAA AABB Clamp')
        taa_clamp_cb.value = taa.enable_clamp
        taa_clamp_cb.callback = lambda v: setattr(taa, 'enable_clamp', v)

        spy.ui.SliderFloat(ui_window, 'TAA Min Alpha', min=0.01, max=1.0, value=taa.min_alpha,
                           callback=lambda v: setattr(taa, 'min_alpha', v))
        spy.ui.SliderFloat(ui_window, 'TAA Sigma Scale', min=0.1, max=20.0, value=taa.sigma_scale,
                           callback=lambda v: setattr(taa, 'sigma_scale', v))
        spy.ui.SliderFloat(ui_window, 'TAA History Sharpness', min=0.0, max=1.0, value=taa.history_sharpness,
                           callback=lambda v: setattr(taa, 'history_sharpness', v))
        spy.ui.SliderFloat(ui_window, 'TAA Max Convergence', min=1.0, max=100.0, value=taa.max_convergence,
                           callback=lambda v: setattr(taa, 'max_convergence', v))

        def on_exposure_changed(value):
            self.tone_mapper.exposure = value
        spy.ui.SliderFloat(ui_window, 'Exposure', min=-4.0, max=4.0, value=1.0, callback=on_exposure_changed)
