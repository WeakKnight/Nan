import time

import slangpy as spy

from scene import Scene
from surface_probe_path_tracer import SurfaceProbePathTracer
from surface_probes import SurfaceProbeLayout
from surface_probe_vertex_lighting import SurfaceProbeVertexLighting


class SurfaceProbeResolve:
    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        layout: SurfaceProbeLayout,
        path_tracer: SurfaceProbePathTracer,
        *,
        profile_sink: list[tuple[str, float]] | None = None,
    ):
        self.device = device
        self.scene = scene
        self.layout = layout
        self.path_tracer = path_tracer
        stage_start = time.perf_counter() if profile_sink is not None else 0.0
        self.program = device.load_program(
            "surface_probe_resolve.slang", ["compute_main"]
        )
        if profile_sink is not None:
            now = time.perf_counter()
            profile_sink.append(("resolve_program_load", now - stage_start))
            stage_start = now
        self.pipeline = device.create_compute_pipeline(self.program)
        if profile_sink is not None:
            profile_sink.append(
                ("resolve_pipeline_create", time.perf_counter() - stage_start)
            )

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        probe_irradiance: spy.Buffer,
        probe_self_hit_counters: spy.Buffer,
        output: spy.Texture,
        *,
        debug_view: int = 0,
        min_gather_count: int = 4,
        use_vertex_fallback: bool = False,
        vertex_lighting: SurfaceProbeVertexLighting | None = None,
        vertex_lighting_rgbm: spy.Buffer | None = None,
        vertex_lighting_target: spy.Buffer | None = None,
        use_vertex_lighting: bool = False,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_probe_irradiance = probe_irradiance
            cursor.g_probe_self_hit_counters = probe_self_hit_counters
            cursor.g_surface_probes = self.path_tracer.probe_buffer
            cursor.g_surface_probe_nodes = self.path_tracer.node_buffer
            cursor.g_surface_probe_instances = self.path_tracer.instance_buffer
            cursor.g_triangle_vertex_probes = (
                self.path_tracer.triangle_vertex_probe_buffer
            )
            if (
                vertex_lighting is None
                or vertex_lighting_rgbm is None
                or vertex_lighting_target is None
            ):
                raise ValueError(
                    "SurfaceProbeResolve requires vertex-lighting resources"
                )
            cursor.g_vertex_lighting_rgbm = vertex_lighting_rgbm
            cursor.g_vertex_lighting_target = vertex_lighting_target
            cursor.g_vertex_lighting_triangle_map = (
                vertex_lighting.triangle_map_buffer
            )
            cursor.g_vertex_lighting_instance_offsets = (
                vertex_lighting.instance_offsets_buffer
            )
            cursor.g_debug_view = max(0, min(7, int(debug_view)))
            cursor.g_max_density_multiplier = float(
                self.layout.max_density_multiplier
            )
            cursor.g_min_gather_count = max(
                1, min(32, int(min_gather_count))
            )
            cursor.g_use_vertex_fallback = int(bool(use_vertex_fallback))
            cursor.g_use_vertex_lighting = int(bool(use_vertex_lighting))
            cursor.g_vertex_lighting_built = int(vertex_lighting.built)
            cursor.g_vertex_lighting_rgbm_range = max(
                vertex_lighting.last_rgbm_range, 1.0
            )
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[output.width, output.height, 1])
