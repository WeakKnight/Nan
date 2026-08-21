import time

import slangpy as spy

from scene import Scene
from surface_probe_fields import (
    DIFFUSE_IRRADIANCE_RGB_FIELD,
    SurfaceProbeAttachments,
    SurfaceProbeFieldBuffers,
)
from surface_probe_resources import SurfaceProbeGpuGeometry


class SurfaceProbeDiffuseIrradianceBaker:
    field_desc = DIFFUSE_IRRADIANCE_RGB_FIELD

    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        geometry: SurfaceProbeGpuGeometry,
        *,
        samples_per_probe: int = 1,
        max_bounces: int = 3,
        profile_sink: list[tuple[str, float]] | None = None,
    ):
        self.device = device
        self.scene = scene
        self.geometry = geometry
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

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        field: SurfaceProbeFieldBuffers,
        attachments: SurfaceProbeAttachments,
        iteration: int,
        *,
        reset: bool = False,
    ) -> None:
        if field.desc != self.field_desc:
            raise ValueError(
                "SurfaceProbeDiffuseIrradianceBaker requires the diffuse "
                "irradiance RGB field"
            )
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_probe_field_values = field.values
            cursor.g_probe_sample_counts = field.sample_counts
            cursor.g_probe_self_hit_counters = attachments.self_hit_counts
            cursor.g_probe_radial_moments = attachments.radial_moments
            cursor.g_surface_probes = self.geometry.probes
            cursor.g_surface_probe_instances = self.geometry.instances
            cursor.g_triangle_vertex_probes = (
                self.geometry.triangle_vertex_probes
            )
            cursor.g_probe_count = self.geometry.probe_count
            cursor.g_iteration = max(0, int(iteration))
            cursor.g_samples_per_probe = self.samples_per_probe
            cursor.g_max_bounces = self.max_bounces
            cursor.g_reset = 1 if reset else 0
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(
                thread_count=[self.geometry.probe_count, 1, 1]
            )


# Compatibility for callers that still import the historical wrapper name.
# New orchestration should treat the implementation as one field baker among
# several possible bakers sharing SurfaceProbeGpuGeometry.
SurfaceProbePathTracer = SurfaceProbeDiffuseIrradianceBaker
