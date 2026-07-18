import slangpy as spy

from accumulator import Accumulator
from mesh_colors import MESH_COLORS_PAYLOAD_SIZE, MeshColorsLayout
from mesh_colors_resolve import MeshColorsResolve
from render_data import RenderData
from scene import Scene
from texture_space_path_tracer import TextureSpacePathTracer
from tone_mapper import ToneMapper


class TextureSpacePathTracingRenderer:
    def __init__(
        self,
        *,
        texels_per_unit: float = 16.0,
        min_resolution: int = 4,
        max_resolution: int = 64,
        max_texels: int = 16_777_216,
        samples_per_texel: int = 1,
        max_bounces: int = 3,
    ):
        self.texels_per_unit = texels_per_unit
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_texels = max_texels
        self.samples_per_texel = samples_per_texel
        self.max_bounces = max_bounces

        self.reset_accumulator = True
        self.reset_texture_space = True
        self.texture_iteration = 0
        self.use_accum_check_box: spy.ui.CheckBox | None = None
        self.pause_check_box: spy.ui.CheckBox | None = None
        self.status_text: spy.ui.Text | None = None
        self._output_size: tuple[int, int] | None = None

    def initialize(self, device: spy.Device, scene: Scene) -> None:
        self.device = device
        self.scene = scene
        self.layout = MeshColorsLayout.build(
            scene.scene_node,
            texels_per_unit=self.texels_per_unit,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            max_total_texels=self.max_texels,
        )
        self.texture_space_path_tracer = TextureSpacePathTracer(
            device,
            scene,
            self.layout,
            samples_per_texel=self.samples_per_texel,
            max_bounces=self.max_bounces,
        )
        self.resolve = MeshColorsResolve(
            device, scene, self.layout, self.texture_space_path_tracer
        )
        self.accumulator = Accumulator(
            device, resource_key="texture_space_renderer.accumulator_history"
        )
        self.tone_mapper = ToneMapper(device)
        scene.event_distpacher.subscribe("camera_move", self.on_camera_move)
        print(
            "[TextureSpace] "
            f"{len(self.layout.instance_infos)} instances, "
            f"{len(self.layout.face_infos)} faces, "
            f"{self.layout.total_surface_texels:,} surface texels, "
            f"{self.layout.total_payload_count - self.layout.total_surface_texels:,} back texels "
            f"({self.layout.total_payload_count * MESH_COLORS_PAYLOAD_SIZE / (1024 * 1024):.1f} MiB)"
        )

    def on_camera_move(self, data) -> None:
        self.reset_accumulator = True
        if isinstance(data, dict) and data.get("texture_space"):
            self.reset_texture_space = True
            self.texture_iteration = 0

    def _use_accumulation(self) -> bool:
        return (
            True
            if self.use_accum_check_box is None
            else bool(self.use_accum_check_box.value)
        )

    def _is_paused(self) -> bool:
        return (
            False
            if self.pause_check_box is None
            else bool(self.pause_check_box.value)
        )

    def _request_texture_reset(self) -> None:
        self.reset_texture_space = True
        self.reset_accumulator = True
        self.texture_iteration = 0

    def render(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
        frame: int,
        device: spy.Device,
        scene: Scene,
        render_data: RenderData,
    ) -> None:
        output_size = (output.width, output.height)
        if output_size != self._output_size:
            self._output_size = output_size
            self.reset_accumulator = True

        mesh_colors = render_data.get_buffer(
            "texture_space_renderer.mesh_colors",
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            struct_size=MESH_COLORS_PAYLOAD_SIZE,
            element_count=self.layout.total_payload_count,
            label="texture_space_mesh_colors",
        )
        resolve_texture = render_data.get_texture(
            "texture_space_renderer.resolve_texture",
            width=output.width,
            height=output.height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access,
            label="texture_space_resolve",
        )
        accum_texture = render_data.get_texture(
            "texture_space_renderer.accum_texture",
            width=output.width,
            height=output.height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access,
            label="texture_space_accum",
        )

        if not self._is_paused() or self.reset_texture_space:
            resetting_texture_space = self.reset_texture_space
            self.texture_space_path_tracer.execute(
                command_encoder,
                mesh_colors,
                self.texture_iteration,
                reset=resetting_texture_space,
            )
            self.texture_iteration += 1
            self.reset_texture_space = False
            if resetting_texture_space:
                self.reset_accumulator = True

        self.resolve.execute(command_encoder, mesh_colors, resolve_texture)
        self.accumulator.execute(
            command_encoder,
            render_data,
            resolve_texture,
            accum_texture,
            self.reset_accumulator,
        )
        self.tone_mapper.execute(
            command_encoder,
            accum_texture if self._use_accumulation() else resolve_texture,
            output,
        )
        self.reset_accumulator = False
        if self.status_text is not None:
            self.status_text.text = (
                f"Texture iterations: {self.texture_iteration}; "
                f"surface: {self.layout.total_surface_texels:,}; "
                f"back: {self.layout.total_payload_count - self.layout.total_surface_texels:,}; "
                f"cache: {self.layout.total_payload_count * MESH_COLORS_PAYLOAD_SIZE / (1024 * 1024):.1f} MiB"
            )

    def setup_ui(
        self, ui_context: spy.ui.Context, ui_window: spy.ui.Window
    ) -> None:
        self.use_accum_check_box = spy.ui.CheckBox(ui_window, "Use Screen Accum")
        self.pause_check_box = spy.ui.CheckBox(ui_window, "Pause Texture Tracing")
        spy.ui.Button(
            ui_window, "Reset Texture Irradiance", callback=self._request_texture_reset
        )
        self.status_text = spy.ui.Text(
            ui_window,
            f"Texture iterations: {self.texture_iteration}; "
            f"surface: {self.layout.total_surface_texels:,}; "
            f"back: {self.layout.total_payload_count - self.layout.total_surface_texels:,}; "
            f"cache: {self.layout.total_payload_count * MESH_COLORS_PAYLOAD_SIZE / (1024 * 1024):.1f} MiB",
        )

        def on_exposure_changed(value):
            self.tone_mapper.exposure = value

        spy.ui.SliderFloat(
            ui_window,
            "Exposure",
            min=-4.0,
            max=4.0,
            value=1.0,
            callback=on_exposure_changed,
        )
