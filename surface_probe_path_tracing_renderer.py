import time

import slangpy as spy

from accumulator import Accumulator
from render_data import RenderData
from scene import Scene
from surface_probe_path_tracer import SurfaceProbePathTracer
from surface_probe_resolve import SurfaceProbeResolve
from surface_probe_vertex_lighting import SurfaceProbeVertexLighting
from surface_probes import SURFACE_PROBE_PAYLOAD_SIZE, SurfaceProbeLayout
from tone_mapper import ToneMapper


SURFACE_PROBE_BUFFER_KEY = "surface_probe_renderer.probe_irradiance"
SURFACE_PROBE_SELF_HIT_BUFFER_KEY = (
    "surface_probe_renderer.probe_self_hit_counters"
)
SURFACE_PROBE_SELF_HIT_SIZE = 4
SURFACE_PROBE_DEBUG_VIEWS = (
    "Beauty",
    "Gather Count",
    "Support f(x)",
    "Density m(x)",
    "Vertex Fallback Weight",
    "Probe Self-hit Rate",
    "Vertex Lighting",
    "Vertex Confidence",
)


class SurfaceProbePathTracingRenderer:
    def __init__(
        self,
        *,
        target_probe_count: int = 20_480,
        oversample_factor: int = 5,
        seed: int = 1,
        samples_per_probe: int = 1,
        max_bounces: int = 3,
        leaf_capacity: int = 8,
        kernel_radius_scale: float = 2.5,
        normal_angle_degrees: float = 45.0,
        repair_budget_ratio: float = 0.30,
        repair_min_gather: int = 4,
        max_density_multiplier: float = 8.0,
        adaptive_wse: bool = True,
        vertex_fallback: bool = False,
        profile_build: bool = False,
        debug_view: int = 0,
        show_gather_count: bool = False,
        use_screen_accumulation: bool = False,
        sampler_backend: str = "auto",
        build_vertex_lighting: bool = False,
        vertex_lighting_build_iteration: int = 0,
        vertex_lighting_smoothing_passes: int = 64,
        vertex_lighting_smoothing_strength: float = 1.0,
    ):
        self.target_probe_count = max(1, int(target_probe_count))
        self.oversample_factor = max(1, int(oversample_factor))
        self.seed = int(seed)
        self.samples_per_probe = max(1, int(samples_per_probe))
        self.max_bounces = max(1, int(max_bounces))
        self.leaf_capacity = max(1, int(leaf_capacity))
        self.kernel_radius_scale = max(0.5, float(kernel_radius_scale))
        self.normal_angle_degrees = float(normal_angle_degrees)
        self.repair_budget_ratio = max(0.0, float(repair_budget_ratio))
        self.repair_min_gather = max(1, min(32, int(repair_min_gather)))
        self.max_density_multiplier = max(
            1.0, float(max_density_multiplier)
        )
        self.adaptive_wse = bool(adaptive_wse)
        self.vertex_fallback = bool(vertex_fallback)
        self.profile_build = bool(profile_build)
        self.debug_view = max(0, min(7, int(debug_view)))
        if show_gather_count:
            self.debug_view = 1
        self.use_screen_accumulation = bool(use_screen_accumulation)
        self.sampler_backend = sampler_backend
        self.use_vertex_lighting = bool(build_vertex_lighting)
        self.vertex_lighting_build_iteration = max(
            0, int(vertex_lighting_build_iteration)
        )
        self.vertex_lighting_smoothing_passes = max(
            0, int(vertex_lighting_smoothing_passes)
        )
        self.vertex_lighting_smoothing_strength = max(
            0.0, float(vertex_lighting_smoothing_strength)
        )

        self.reset_accumulator = True
        self.reset_probes = True
        self.probe_iteration = 0
        self.use_accum_check_box: spy.ui.CheckBox | None = None
        self.pause_check_box: spy.ui.CheckBox | None = None
        self.vertex_fallback_check_box: spy.ui.CheckBox | None = None
        self.use_vertex_lighting_check_box: spy.ui.CheckBox | None = None
        self.vertex_lighting_passes_slider: spy.ui.SliderInt | None = None
        self.vertex_lighting_strength_slider: spy.ui.SliderFloat | None = None
        self.vertex_lighting_rgbm_range_slider: spy.ui.SliderFloat | None = None
        self.build_vertex_lighting_button: spy.ui.Button | None = None
        self.debug_view_combo: spy.ui.ComboBox | None = None
        self.status_text: spy.ui.Text | None = None
        self._output_size: tuple[int, int] | None = None
        self._previous_debug_view = self.debug_view
        self._previous_vertex_fallback = self.vertex_fallback
        self._previous_use_vertex_lighting = self.use_vertex_lighting
        self._build_vertex_lighting_requested = bool(build_vertex_lighting)

    def initialize(self, device: spy.Device, scene: Scene) -> None:
        initialize_start = time.perf_counter()
        self.device = device
        self.scene = scene
        renderer_profile_samples: list[tuple[str, float]] = []
        profile_sink = (
            renderer_profile_samples if self.profile_build else None
        )
        layout_start = time.perf_counter()
        self.layout = SurfaceProbeLayout.build(
            scene.scene_node,
            target_probe_count=self.target_probe_count,
            oversample_factor=self.oversample_factor,
            seed=self.seed,
            leaf_capacity=self.leaf_capacity,
            kernel_radius_scale=self.kernel_radius_scale,
            normal_angle_degrees=self.normal_angle_degrees,
            repair_budget_ratio=self.repair_budget_ratio,
            repair_min_gather=self.repair_min_gather,
            max_density_multiplier=self.max_density_multiplier,
            adaptive_wse=self.adaptive_wse,
            build_vertex_anchors=self.vertex_fallback,
            profile_build=self.profile_build,
            sampler_backend=self.sampler_backend,
        )
        layout_elapsed = time.perf_counter() - layout_start
        if profile_sink is not None:
            profile_sink.append(("layout_build", layout_elapsed))
        self.path_tracer = SurfaceProbePathTracer(
            device,
            scene,
            self.layout,
            samples_per_probe=self.samples_per_probe,
            max_bounces=self.max_bounces,
            profile_sink=profile_sink,
        )
        self.vertex_lighting = SurfaceProbeVertexLighting(
            device,
            scene,
            self.path_tracer,
            profile_sink=profile_sink,
        )
        print(
            "[VertexLightingCoverage] "
            f"vertices={self.vertex_lighting.layout.vertex_count:,}; "
            f"zero_projection="
            f"{self.vertex_lighting.layout.zero_projection_vertex_count:,}; "
            f"partial_projection="
            f"{self.vertex_lighting.layout.partial_projection_vertex_count:,}; "
            "zero/partial vertices use same-instance spatial gather fallback"
        )
        self.resolve = SurfaceProbeResolve(
            device,
            scene,
            self.layout,
            self.path_tracer,
            profile_sink=profile_sink,
        )
        self.accumulator = Accumulator(
            device,
            resource_key="surface_probe_renderer.accumulator_history",
            profile_sink=profile_sink,
        )
        self.tone_mapper = ToneMapper(device, profile_sink=profile_sink)
        subscribe_start = time.perf_counter()
        scene.event_distpacher.subscribe("camera_move", self.on_camera_move)
        if profile_sink is not None:
            profile_sink.append(
                ("event_subscribe", time.perf_counter() - subscribe_start)
            )
        core_end = time.perf_counter()
        if self.profile_build:
            core_total = core_end - initialize_start
            accounted = sum(elapsed for _, elapsed in renderer_profile_samples)
            layout_internal = sum(
                elapsed for _, elapsed in self.layout.build_profile
            )
            stage_text = "; ".join(
                f"{label}={elapsed:.3f}s/"
                f"{100.0 * elapsed / max(core_total, 1.0e-12):.1f}%"
                for label, elapsed in renderer_profile_samples
            )
            profile_report_start = time.perf_counter()
            print(
                f"[SurfaceProbeProfile] layout_total: "
                f"{layout_elapsed:.3f}s",
                flush=True,
            )
            print(
                "[SurfaceProbeLayoutOuterProfile] "
                f"internal_stages={layout_internal:.3f}s; "
                f"return_and_reporting="
                f"{max(layout_elapsed - layout_internal, 0.0):.3f}s",
                flush=True,
            )
            print(
                "[SurfaceProbeRendererCoreProfile] "
                f"core_total={core_total:.3f}s; "
                f"accounted={accounted:.3f}s; "
                f"other={max(core_total - accounted, 0.0):.3f}s; "
                f"stages=[{stage_text}]",
                flush=True,
            )
            print(
                f"[SurfaceProbeProfile] renderer_core_initialize_total: "
                f"{core_total:.3f}s",
                flush=True,
            )
            profile_report_elapsed = (
                time.perf_counter() - profile_report_start
            )
        status_logging_start = time.perf_counter()
        print(
            "[SurfaceProbe] "
            f"{len(self.layout.instance_infos)} instances, "
            f"{self.layout.total_candidate_count:,} candidates, "
            f"{self.layout.total_audit_point_count:,} audit points, "
            f"{self.layout.total_base_surface_site_count:,} base sites + "
            f"{self.layout.total_repair_surface_site_count:,}/"
            f"{self.layout.repair_surface_site_budget:,} repair sites + "
            f"{self.layout.total_protected_surface_site_count:,} protected = "
            f"{self.layout.total_surface_site_count:,} surface sites, "
            f"{self.layout.total_vertex_anchor_site_count:,} vertex anchors, "
            f"{self.layout.total_probe_count:,} total probes "
            f"(sampler={self.layout.sampler_backend}; "
            f"adaptive_wse={self.layout.adaptive_wse}; "
            f"{self.layout.total_probe_count * SURFACE_PROBE_PAYLOAD_SIZE / (1024 * 1024):.2f} MiB irradiance; "
            f"{self.layout.total_probe_count * SURFACE_PROBE_SELF_HIT_SIZE / (1024 * 1024):.2f} MiB self-hit counters; "
            f"{self.layout.probes.nbytes / (1024 * 1024):.2f} MiB metadata)"
        )
        for index, info in enumerate(self.layout.instance_infos):
            print(
                f"  instance {index}: sites={info.surface_site_count}, "
                f"base={info.base_surface_site_count}, "
                f"repair={info.repair_surface_site_count}, "
                f"protected={info.protected_surface_site_count}, "
                f"vertex_anchors={info.vertex_anchor_site_count}, "
                f"probes={info.reconstruction_probe_count}+"
                f"{info.vertex_anchor_probe_count}, nodes={info.node_count}, "
                f"radius={info.kernel_radius:.4f}, "
                f"audit={info.audit_point_count}, "
                f"zero={info.zero_gather_before}->"
                f"{info.zero_gather_after_repair}->"
                f"{info.zero_gather_after}, "
                f"deficit={info.deficit_point_count_before}->"
                f"{info.deficit_point_count_after_repair}->"
                f"{info.deficit_point_count_after}, "
                f"ess_p50={info.ess_p50_before:.2f}->"
                f"{info.ess_p50_after:.2f}, "
                f"adaptive_m_mean={info.adaptive_density_mean:.2f}, "
                f"adaptive_m_p95={info.adaptive_density_p95:.2f}"
            )
        print(
            "[SurfaceProbeRepair] "
            f"zero={self.layout.zero_gather_before}->"
            f"{self.layout.zero_gather_after_repair}->"
            f"{self.layout.zero_gather_after}; "
            f"count<{self.repair_min_gather}="
            f"{self.layout.deficit_point_count_before}->"
            f"{self.layout.deficit_point_count_after_repair}->"
            f"{self.layout.deficit_point_count_after}; "
            f"stop={self.layout.repair_stop_reason}; "
            f"closure={self.layout.protected_stop_reason} "
            f"({self.layout.total_protected_surface_site_count:,} sites); "
            f"irreparable={self.layout.irreparable_audit_point_count}; "
            f"ess_p50={self.layout.ess_p50_before:.2f}->"
            f"{self.layout.ess_p50_after:.2f}; "
            f"f_p10={self.layout.support_f_p10:.3f}; "
            f"f_p50={self.layout.support_f_p50:.3f}; "
            f"m_p95={self.layout.density_m_p95:.2f}"
        )
        print(
            "[VertexLighting] "
            f"{self.vertex_lighting.layout.vertex_count:,} instance vertices, "
            f"{self.vertex_lighting.layout.projection_sample_count:,} "
            "area-projection contributions, "
            f"{self.vertex_lighting.layout.edge_count:,} directed topology "
            f"edges, {self.vertex_lighting.layout.weld_edge_count:,} seam "
            f"welds, condition mean="
            f"{self.vertex_lighting.layout.condition_mean:.2f}, "
            f"p95={self.vertex_lighting.layout.condition_p95:.2f}, "
            f"RGBM={self.vertex_lighting.layout.vertex_count * 4 / (1024 * 1024):.2f} MiB"
        )
        if self.profile_build:
            status_logging_elapsed = (
                time.perf_counter() - status_logging_start
            )
            initialize_total = time.perf_counter() - initialize_start
            print(
                "[SurfaceProbeRendererOverheadProfile] "
                f"profile_reporting={profile_report_elapsed:.3f}s; "
                f"status_logging={status_logging_elapsed:.3f}s",
                flush=True,
            )
            print(
                f"[SurfaceProbeProfile] renderer_initialize_total: "
                f"{initialize_total:.3f}s",
                flush=True,
            )

    def on_camera_move(self, data) -> None:
        self.reset_accumulator = True
        if isinstance(data, dict) and data.get("surface_probe"):
            self.reset_probes = True
            self.probe_iteration = 0

    def _use_accumulation(self) -> bool:
        return (
            self.use_screen_accumulation
            if self.use_accum_check_box is None
            else bool(self.use_accum_check_box.value)
        )

    def _is_paused(self) -> bool:
        return (
            False
            if self.pause_check_box is None
            else bool(self.pause_check_box.value)
        )

    def _use_vertex_fallback(self) -> bool:
        return (
            self.vertex_fallback
            if self.vertex_fallback_check_box is None
            else bool(self.vertex_fallback_check_box.value)
        )

    def _debug_view(self) -> int:
        return (
            self.debug_view
            if self.debug_view_combo is None
            else max(0, min(7, int(self.debug_view_combo.value)))
        )

    def _use_vertex_lighting(self) -> bool:
        requested = (
            self.use_vertex_lighting
            if self.use_vertex_lighting_check_box is None
            else bool(self.use_vertex_lighting_check_box.value)
        )
        return requested and self.vertex_lighting.built

    def _request_vertex_lighting_build(self) -> None:
        self._build_vertex_lighting_requested = True
        self.reset_accumulator = True
        if self.status_text is not None:
            self.status_text.text = "Vertex Lighting build queued..."

    def _request_probe_reset(self) -> None:
        self.reset_probes = True
        self.reset_accumulator = True
        self.probe_iteration = 0
        if self.pause_check_box is not None:
            self.pause_check_box.value = False

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

        debug_view = self._debug_view()
        if debug_view != self._previous_debug_view:
            self.reset_accumulator = True
            self._previous_debug_view = debug_view

        use_vertex_fallback = self._use_vertex_fallback()
        if use_vertex_fallback != self._previous_vertex_fallback:
            self.reset_accumulator = True
            self._previous_vertex_fallback = use_vertex_fallback

        use_vertex_lighting = self._use_vertex_lighting()
        if use_vertex_lighting != self._previous_use_vertex_lighting:
            self.reset_accumulator = True
            self._previous_use_vertex_lighting = use_vertex_lighting

        probe_irradiance = render_data.get_buffer(
            SURFACE_PROBE_BUFFER_KEY,
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            struct_size=SURFACE_PROBE_PAYLOAD_SIZE,
            element_count=self.layout.total_probe_count,
            label="surface_probe_irradiance",
        )
        probe_self_hit_counters = render_data.get_buffer(
            SURFACE_PROBE_SELF_HIT_BUFFER_KEY,
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            struct_size=SURFACE_PROBE_SELF_HIT_SIZE,
            element_count=self.layout.total_probe_count,
            label="surface_probe_self_hit_counters",
        )

        if not self._is_paused() or self.reset_probes:
            resetting = self.reset_probes
            self.path_tracer.execute(
                command_encoder,
                probe_irradiance,
                probe_self_hit_counters,
                self.probe_iteration,
                reset=resetting,
            )
            if resetting:
                self.reset_probes = False
            self.probe_iteration += 1

        vertex_lighting_rgbm = self.vertex_lighting.packed_buffer(render_data)
        vertex_lighting_target = self.vertex_lighting.target_buffer(render_data)
        if (
            self._build_vertex_lighting_requested
            and self.probe_iteration >= self.vertex_lighting_build_iteration
        ):
            smoothing_passes = (
                self.vertex_lighting_smoothing_passes
                if self.vertex_lighting_passes_slider is None
                else int(self.vertex_lighting_passes_slider.value)
            )
            regularization_strength = (
                self.vertex_lighting_smoothing_strength
                if self.vertex_lighting_strength_slider is None
                else float(self.vertex_lighting_strength_slider.value)
            )
            rgbm_range = (
                32.0
                if self.vertex_lighting_rgbm_range_slider is None
                else float(self.vertex_lighting_rgbm_range_slider.value)
            )
            build_start = time.perf_counter()
            vertex_lighting_rgbm = self.vertex_lighting.execute(
                command_encoder,
                render_data,
                probe_irradiance,
                min_gather_count=self.repair_min_gather,
                smoothing_passes=smoothing_passes,
                regularization_strength=regularization_strength,
                rgbm_range=rgbm_range,
            )
            self._build_vertex_lighting_requested = False
            self.reset_accumulator = True
            if self.use_vertex_lighting_check_box is not None:
                self.use_vertex_lighting_check_box.value = True
            self.use_vertex_lighting = True
            use_vertex_lighting = True
            self._previous_use_vertex_lighting = True
            print(
                "[VertexLighting] submitted "
                f"{self.vertex_lighting.layout.vertex_count:,} vertices, "
                f"{self.vertex_lighting.layout.zero_projection_vertex_count:,} "
                "zero-projection fallback vertices, "
                f"{self.vertex_lighting.layout.partial_projection_vertex_count:,} "
                "partial-blend vertices, "
                f"{smoothing_passes} smoothing passes, "
                f"strength={regularization_strength:.3f}, "
                f"RGBM range={rgbm_range:.1f} in "
                f"{time.perf_counter() - build_start:.3f}s CPU"
            )

        resolve_texture = render_data.get_texture(
            "surface_probe_renderer.resolve",
            width=output.width,
            height=output.height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.unordered_access
            | spy.TextureUsage.shader_resource,
            label="surface_probe_resolve",
        )
        accum_texture = render_data.get_texture(
            "surface_probe_renderer.accumulated",
            width=output.width,
            height=output.height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.unordered_access
            | spy.TextureUsage.shader_resource,
            label="surface_probe_accumulated",
        )
        self.resolve.execute(
            command_encoder,
            probe_irradiance,
            probe_self_hit_counters,
            resolve_texture,
            debug_view=debug_view,
            min_gather_count=self.repair_min_gather,
            use_vertex_fallback=use_vertex_fallback,
            vertex_lighting=self.vertex_lighting,
            vertex_lighting_rgbm=vertex_lighting_rgbm,
            vertex_lighting_target=vertex_lighting_target,
            use_vertex_lighting=use_vertex_lighting,
        )
        if debug_view not in (0, 6):
            command_encoder.blit(output, resolve_texture)
            if self.status_text is not None:
                if debug_view == 1:
                    self.status_text.text = (
                        "Gather count: magenta=0; red/orange=1-3; "
                        "yellow=4-7; green=8-15; cyan=16-23; "
                        "blue/white=24-32"
                    )
                elif debug_view == 2:
                    self.status_text.text = (
                        "Support f(x): magenta=0; red=0.25; "
                        "yellow=0.5; green=0.75; blue=1"
                    )
                elif debug_view == 3:
                    self.status_text.text = (
                        "Density m(x)=clamp(1/f, 1, "
                        f"{self.layout.max_density_multiplier:g}); "
                        "blue=1; magenta=max (log scale)"
                    )
                elif debug_view == 4:
                    self.status_text.text = (
                        "Vertex fallback weight: blue=0; green/yellow=blend; "
                        "red/magenta=triangle-local fallback dominates"
                    )
                elif debug_view == 5:
                    self.status_text.text = (
                        "Probe path self-hit suspects/update (log): "
                        "dark blue=0; cyan~1e-5; yellow~1e-3; red>=1e-1"
                    )
                else:
                    self.status_text.text = (
                        "Vertex gather confidence before topology solve: "
                        "magenta=0; red=0.25; yellow=0.5; "
                        "green=0.75; blue=1"
                    )
            return
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
                f"Iterating; iterations: {self.probe_iteration}; "
                f"sites: {self.layout.total_base_surface_site_count:,} + "
                f"{self.layout.total_repair_surface_site_count:,} repair; "
                f"{self.layout.total_protected_surface_site_count:,} protected; "
                f"vertex anchors: "
                f"{self.layout.total_vertex_anchor_site_count:,}; "
                f"probes: {self.layout.total_probe_count:,}"
                "; probe cache: indirect + sky, sun direct: realtime"
                + (
                    f"; vertex lighting: RGBM, "
                    f"{self.vertex_lighting.last_pass_count} passes"
                    if self.vertex_lighting.built
                    else "; vertex lighting: not built"
                )
            )

    def setup_ui(
        self, ui_context: spy.ui.Context, ui_window: spy.ui.Window
    ) -> None:
        self.use_accum_check_box = spy.ui.CheckBox(
            ui_window,
            "Use Screen Accum",
            value=self.use_screen_accumulation,
        )
        self.pause_check_box = spy.ui.CheckBox(
            ui_window, "Pause Probe Tracing", value=False
        )
        if self.layout.total_vertex_anchor_probe_count > 0:
            self.vertex_fallback_check_box = spy.ui.CheckBox(
                ui_window,
                "Vertex Fallback",
                value=self.vertex_fallback,
            )
        self.use_vertex_lighting_check_box = spy.ui.CheckBox(
            ui_window,
            "Use Vertex Lighting",
            value=self.use_vertex_lighting,
        )
        self.debug_view_combo = spy.ui.ComboBox(
            ui_window,
            "Debug View",
            items=list(SURFACE_PROBE_DEBUG_VIEWS),
            value=self.debug_view,
        )
        self.vertex_lighting_passes_slider = spy.ui.SliderInt(
            ui_window,
            "Vertex Smooth Passes",
            min=0,
            max=128,
            value=64,
        )
        self.vertex_lighting_strength_slider = spy.ui.SliderFloat(
            ui_window,
            "Vertex Smooth Strength",
            min=0.0,
            max=8.0,
            value=1.0,
        )
        self.vertex_lighting_rgbm_range_slider = spy.ui.SliderFloat(
            ui_window,
            "Vertex RGBM Range",
            min=1.0,
            max=128.0,
            value=32.0,
        )
        self.build_vertex_lighting_button = spy.ui.Button(
            ui_window,
            "Build Vertex Lighting",
            callback=self._request_vertex_lighting_build,
        )
        spy.ui.Button(
            ui_window,
            "Reset Probe Irradiance",
            callback=self._request_probe_reset,
        )
        self.status_text = spy.ui.Text(ui_window, "")

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
