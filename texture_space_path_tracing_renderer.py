import math

import slangpy as spy

from accumulator import Accumulator
from mesh_colors import MESH_COLORS_PAYLOAD_SIZE, MeshColorsLayout
from mesh_colors_resolve import MeshColorsResolve
from mesh_colors_rgb9e5 import MeshColorsRGB9E5Packer
from mesh_colors_surface_filter import MeshColorsSurfaceFilter
from render_data import RenderData
from scene import Scene
from texture_space_path_tracer import TextureSpacePathTracer
from tone_mapper import ToneMapper


MESH_COLORS_BUFFER_KEY = "texture_space_renderer.mesh_colors"
MESH_COLORS_FILTER_SCRATCH_KEY = "texture_space_renderer.mesh_colors_filter_scratch"
MESH_COLORS_RGB9E5_KEY = "texture_space_renderer.mesh_colors_rgb9e5"


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
        self.filter_before_save_check_box: spy.ui.CheckBox | None = None
        self.filter_pass_count_slider: spy.ui.SliderInt | None = None
        self.filter_spatial_sigma_slider: spy.ui.SliderFloat | None = None
        self.filter_normal_angle_slider: spy.ui.SliderFloat | None = None
        self.save_rgb9e5_button: spy.ui.Button | None = None
        self.status_text: spy.ui.Text | None = None
        self._output_size: tuple[int, int] | None = None

        self._save_requested = False
        self._frozen_rgb9e5: spy.Buffer | None = None
        self._release_frozen_buffer = False
        self._save_error: str | None = None

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
        self.surface_filter = MeshColorsSurfaceFilter(
            device,
            scene,
            self.layout,
            face_infos_buffer=self.texture_space_path_tracer.face_infos_buffer,
            instance_infos_buffer=self.texture_space_path_tracer.instance_infos_buffer,
            side_infos_buffer=self.texture_space_path_tracer.side_infos_buffer,
            adjacency_infos_buffer=self.texture_space_path_tracer.adjacency_infos_buffer,
        )
        self.rgb9e5_packer = MeshColorsRGB9E5Packer(device)
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
            f"({self.layout.total_payload_count * MESH_COLORS_PAYLOAD_SIZE / (1024 * 1024):.1f} MiB writable; "
            f"{self.layout.total_payload_count * 4 / (1024 * 1024):.1f} MiB RGB9E5)"
        )

    def on_camera_move(self, data) -> None:
        self.reset_accumulator = True
        if isinstance(data, dict) and data.get("texture_space"):
            self._discard_rgb9e5()
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

    def _filter_before_save(self) -> bool:
        return (
            True
            if self.filter_before_save_check_box is None
            else bool(self.filter_before_save_check_box.value)
        )

    def _filter_pass_count(self) -> int:
        return (
            5
            if self.filter_pass_count_slider is None
            else max(0, int(self.filter_pass_count_slider.value))
        )

    def _filter_spatial_sigma(self) -> float:
        return (
            1.0
            if self.filter_spatial_sigma_slider is None
            else max(0.05, float(self.filter_spatial_sigma_slider.value))
        )

    def _filter_normal_sigma_radians(self) -> float:
        degrees = (
            30.0
            if self.filter_normal_angle_slider is None
            else max(0.0, float(self.filter_normal_angle_slider.value))
        )
        return math.radians(degrees)

    def _request_texture_reset(self) -> None:
        self._discard_rgb9e5()
        self.reset_texture_space = True
        self.reset_accumulator = True
        self.texture_iteration = 0
        if self.pause_check_box is not None:
            self.pause_check_box.value = False

    def _discard_rgb9e5(self) -> None:
        self._save_requested = False
        self._frozen_rgb9e5 = None
        self._release_frozen_buffer = True
        self._save_error = None
        if self.save_rgb9e5_button is not None:
            self.save_rgb9e5_button.enabled = True

    def _request_rgb9e5_save(self) -> None:
        if self._save_requested or self._frozen_rgb9e5 is not None:
            return
        if self.texture_iteration < 1:
            self._save_error = "RGB9E5 save requires at least one texture iteration"
            return
        self._save_requested = True
        self._save_error = None
        if self.pause_check_box is not None:
            self.pause_check_box.value = True
        if self.save_rgb9e5_button is not None:
            self.save_rgb9e5_button.enabled = False

    def _execute_rgb9e5_save(
        self,
        command_encoder: spy.CommandEncoder,
        mesh_colors: spy.Buffer,
        filter_scratch: spy.Buffer | None,
        packed: spy.Buffer,
    ) -> bool:
        self._save_requested = False
        try:
            source = mesh_colors
            pass_count = self._filter_pass_count()
            if self._filter_before_save() and pass_count > 0:
                if filter_scratch is None:
                    raise RuntimeError(
                        "Mesh Colors filter scratch buffer is unavailable"
                    )
                destination = filter_scratch
                for _ in range(pass_count):
                    self.surface_filter.execute(
                        command_encoder,
                        source,
                        destination,
                        spatial_sigma=self._filter_spatial_sigma(),
                        normal_sigma_radians=self._filter_normal_sigma_radians(),
                    )
                    source, destination = destination, source

            self.rgb9e5_packer.execute(
                command_encoder,
                source,
                packed,
                self.layout.total_payload_count,
            )
            self._frozen_rgb9e5 = packed
            self._save_error = None
            self.reset_accumulator = True
            print(
                "[TextureSpace] RGB9E5 frozen cache: "
                f"{self.layout.total_payload_count:,} texels, "
                f"{self.layout.total_payload_count * 4 / (1024 * 1024):.1f} MiB"
            )
            return True
        except Exception as exc:
            self._frozen_rgb9e5 = None
            self._release_frozen_buffer = True
            self._save_error = str(exc)
            self.reset_texture_space = True
            self.texture_iteration = 0
            if self.save_rgb9e5_button is not None:
                self.save_rgb9e5_button.enabled = True
            return False

    def _status(self) -> str:
        common = (
            f"iterations: {self.texture_iteration}; "
            f"surface: {self.layout.total_surface_texels:,}; "
            f"back: "
            f"{self.layout.total_payload_count - self.layout.total_surface_texels:,}"
        )
        if self._frozen_rgb9e5 is not None:
            size = self.layout.total_payload_count * 4 / (1024 * 1024)
            return f"RGB9E5 frozen; {common}; cache: {size:.1f} MiB"
        if self._save_requested:
            return f"RGB9E5 save requested; {common}"
        working_size = (
            self.layout.total_payload_count * MESH_COLORS_PAYLOAD_SIZE
            / (1024 * 1024)
        )
        prefix = (
            f"RGB9E5 save failed: {self._save_error}; "
            if self._save_error
            else "Iterating; "
        )
        return f"{prefix}{common}; cache: {working_size:.1f} MiB"

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

        if self._release_frozen_buffer:
            render_data.release_buffer(MESH_COLORS_RGB9E5_KEY)
            self._release_frozen_buffer = False

        mesh_colors: spy.Buffer | None = None
        if self._frozen_rgb9e5 is None:
            mesh_colors = render_data.get_buffer(
                MESH_COLORS_BUFFER_KEY,
                usage=spy.BufferUsage.unordered_access
                | spy.BufferUsage.shader_resource,
                struct_size=MESH_COLORS_PAYLOAD_SIZE,
                element_count=self.layout.total_payload_count,
                label="texture_space_mesh_colors",
            )

        if self._save_requested and mesh_colors is not None:
            filter_scratch: spy.Buffer | None = None
            if self._filter_before_save() and self._filter_pass_count() > 0:
                filter_scratch = render_data.get_buffer(
                    MESH_COLORS_FILTER_SCRATCH_KEY,
                    usage=spy.BufferUsage.unordered_access
                    | spy.BufferUsage.shader_resource,
                    struct_size=MESH_COLORS_PAYLOAD_SIZE,
                    element_count=self.layout.total_payload_count,
                    label="texture_space_mesh_colors_filter_scratch",
                )
            packed = render_data.get_buffer(
                MESH_COLORS_RGB9E5_KEY,
                usage=spy.BufferUsage.unordered_access
                | spy.BufferUsage.shader_resource,
                struct_size=4,
                element_count=self.layout.total_payload_count,
                label="texture_space_mesh_colors_rgb9e5",
            )
            if self._execute_rgb9e5_save(
                command_encoder, mesh_colors, filter_scratch, packed
            ):
                render_data.release_buffer(MESH_COLORS_BUFFER_KEY)
                render_data.release_buffer(MESH_COLORS_FILTER_SCRATCH_KEY)

        if (
            self._frozen_rgb9e5 is None
            and not self._save_requested
            and (not self._is_paused() or self.reset_texture_space)
        ):
            assert mesh_colors is not None
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

        if self._frozen_rgb9e5 is not None:
            self.resolve.execute_frozen(
                command_encoder, self._frozen_rgb9e5, resolve_texture
            )
        else:
            assert mesh_colors is not None
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
            self.status_text.text = self._status()
        if self.save_rgb9e5_button is not None:
            self.save_rgb9e5_button.enabled = (
                self._frozen_rgb9e5 is None and not self._save_requested
            )

    def setup_ui(
        self, ui_context: spy.ui.Context, ui_window: spy.ui.Window
    ) -> None:
        self.use_accum_check_box = spy.ui.CheckBox(ui_window, "Use Screen Accum")
        self.pause_check_box = spy.ui.CheckBox(ui_window, "Pause Texture Tracing")
        self.filter_before_save_check_box = spy.ui.CheckBox(
            ui_window, "Filter Before RGB9E5", value=True
        )
        self.filter_pass_count_slider = spy.ui.SliderInt(
            ui_window, "Filter Passes", min=0, max=32, value=5
        )
        self.filter_spatial_sigma_slider = spy.ui.SliderFloat(
            ui_window,
            "Filter Spatial Sigma (texel hops)",
            min=0.05,
            max=4.0,
            value=1.0,
        )
        self.filter_normal_angle_slider = spy.ui.SliderFloat(
            ui_window,
            "Filter Normal Sigma (degrees)",
            min=0.0,
            max=180.0,
            value=30.0,
        )
        spy.ui.Button(
            ui_window, "Reset Texture Irradiance", callback=self._request_texture_reset
        )
        self.save_rgb9e5_button = spy.ui.Button(
            ui_window, "Save RGB9E5", callback=self._request_rgb9e5_save
        )
        self.status_text = spy.ui.Text(ui_window, self._status())

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
