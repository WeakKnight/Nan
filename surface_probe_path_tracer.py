import time

import slangpy as spy

from scene import Scene
from surface_probes import SurfaceProbeLayout


class SurfaceProbePathTracer:
    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        layout: SurfaceProbeLayout,
        *,
        samples_per_probe: int = 1,
        max_bounces: int = 3,
        profile_sink: list[tuple[str, float]] | None = None,
    ):
        self.device = device
        self.scene = scene
        self.layout = layout
        self.samples_per_probe = max(1, int(samples_per_probe))
        self.max_bounces = max(1, int(max_bounces))
        stage_start = time.perf_counter() if profile_sink is not None else 0.0

        def profile_mark(label: str) -> None:
            nonlocal stage_start
            if profile_sink is None:
                return
            now = time.perf_counter()
            profile_sink.append((label, now - stage_start))
            stage_start = now

        self.program = device.load_program(
            "surface_probe_path_tracer.slang", ["compute_main"]
        )
        profile_mark("trace_program_load")
        self.pipeline = device.create_compute_pipeline(self.program)
        profile_mark("trace_pipeline_create")
        (
            self.probe_buffer,
            self.node_buffer,
            self.instance_buffer,
            self.triangle_vertex_probe_buffer,
        ) = layout.create_gpu_buffers(device, profile_sink=profile_sink)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Buffer,
        self_hit_counters: spy.Buffer,
        radial_moments: spy.Buffer,
        iteration: int,
        *,
        reset: bool = False,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_probe_irradiance = output
            cursor.g_probe_self_hit_counters = self_hit_counters
            cursor.g_probe_radial_moments = radial_moments
            cursor.g_surface_probes = self.probe_buffer
            cursor.g_surface_probe_instances = self.instance_buffer
            cursor.g_triangle_vertex_probes = (
                self.triangle_vertex_probe_buffer
            )
            cursor.g_probe_count = self.layout.total_probe_count
            cursor.g_iteration = max(0, int(iteration))
            cursor.g_samples_per_probe = self.samples_per_probe
            cursor.g_max_bounces = self.max_bounces
            cursor.g_reset = 1 if reset else 0
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(
                thread_count=[self.layout.total_probe_count, 1, 1]
            )
