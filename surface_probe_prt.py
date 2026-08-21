from __future__ import annotations

import time

import numpy as np
import slangpy as spy

from render_data import RenderData
from scene import Scene
from surface_probe_fields import (
    DIFFUSE_IRRADIANCE_RGB_FIELD,
    DIFFUSE_PRT_L2_RGB_FIELD,
    SurfaceProbeAttachments,
    SurfaceProbeFieldBuffers,
    SurfaceProbeRuntimeBuffers,
)
from surface_probe_resources import SurfaceProbeGpuGeometry


def surface_probe_sh_l2_basis(direction) -> np.ndarray:
    """CPU reference for the shader's real, orthonormal SH L2 basis."""
    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        raise ValueError("SH direction must be non-zero")
    x, y, z = direction / length
    return np.asarray(
        (
            0.2820947918,
            0.4886025119 * y,
            0.4886025119 * z,
            0.4886025119 * x,
            1.0925484306 * x * y,
            1.0925484306 * y * z,
            0.3153915653 * (3.0 * y * y - 1.0),
            1.0925484306 * x * z,
            0.5462742153 * (x * x - z * z),
        ),
        dtype=np.float64,
    )


def evaluate_prt_l2_rgb(
    transport,
    lighting_sh,
    static_source=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """CPU reference for the GPU PRT-to-irradiance evaluation pass."""
    transport_array = np.asarray(transport, dtype=np.float64)
    lighting_array = np.asarray(lighting_sh, dtype=np.float64)
    if transport_array.shape != (9, 3):
        raise ValueError("PRT transport must have shape (9, 3)")
    if lighting_array.shape != (9, 3):
        raise ValueError("PRT lighting SH must have shape (9, 3)")
    result = np.asarray(static_source, dtype=np.float64) + np.sum(
        transport_array * lighting_array,
        axis=0,
    )
    return np.maximum(result, 0.0)


class SurfaceProbePrtBaker:
    field_desc = DIFFUSE_PRT_L2_RGB_FIELD

    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        geometry: SurfaceProbeGpuGeometry,
        *,
        samples_per_probe: int = 1,
        max_bounces: int = 3,
        profile_sink: list[tuple[str, float]] | None = None,
    ) -> None:
        self.scene = scene
        self.geometry = geometry
        self.samples_per_probe = max(1, int(samples_per_probe))
        self.max_bounces = max(1, int(max_bounces))
        stage_start = time.perf_counter()
        self.program = device.load_program(
            "surface_probe_prt_baker.slang", ["compute_main"]
        )
        if profile_sink is not None:
            now = time.perf_counter()
            profile_sink.append(("prt_baker_program_load", now - stage_start))
            stage_start = now
        self.pipeline = device.create_compute_pipeline(self.program)
        if profile_sink is not None:
            profile_sink.append(
                ("prt_baker_pipeline_create", time.perf_counter() - stage_start)
            )

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
                "SurfaceProbePrtBaker requires the L2 RGB PRT field"
            )
        with command_encoder.begin_compute_pass() as pass_encoder:
            cursor = spy.ShaderCursor(
                pass_encoder.bind_pipeline(self.pipeline)
            )
            cursor.g_probe_field_values = field.values
            cursor.g_probe_sample_counts = field.sample_counts
            cursor.g_probe_self_hit_counters = attachments.self_hit_counts
            cursor.g_probe_radial_moments = attachments.radial_moments
            cursor.g_probe_static_source_rgb = attachments.static_source_rgb
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


class SurfaceProbePrtEvaluator:
    LIGHTING_PROJECTION_SAMPLE_COUNT = 2048

    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        geometry: SurfaceProbeGpuGeometry,
        *,
        profile_sink: list[tuple[str, float]] | None = None,
    ) -> None:
        self.scene = scene
        self.geometry = geometry
        stage_start = time.perf_counter()
        self.project_program = device.load_program(
            "surface_probe_prt_evaluate.slang",
            ["project_lighting_main"],
        )
        self.evaluate_program = device.load_program(
            "surface_probe_prt_evaluate.slang",
            ["evaluate_field_main"],
        )
        if profile_sink is not None:
            now = time.perf_counter()
            profile_sink.append(("prt_evaluator_program_load", now - stage_start))
            stage_start = now
        self.project_pipeline = device.create_compute_pipeline(
            self.project_program
        )
        self.evaluate_pipeline = device.create_compute_pipeline(
            self.evaluate_program
        )
        if profile_sink is not None:
            profile_sink.append(
                (
                    "prt_evaluator_pipeline_create",
                    time.perf_counter() - stage_start,
                )
            )

    def lighting_signature(self) -> tuple[float, ...]:
        direction = self.scene.sun_direction
        return (
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
            float(self.scene.directional_light_intensity),
        )

    def acquire_evaluated_field(
        self,
        render_data: RenderData,
    ) -> SurfaceProbeFieldBuffers:
        return SurfaceProbeRuntimeBuffers.acquire(
            render_data,
            self.geometry.probe_count,
            field_desc=DIFFUSE_IRRADIANCE_RGB_FIELD,
            attachment_descs=(),
            resource_prefix="surface_probe_renderer.prt_evaluated",
        ).field

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        render_data: RenderData,
        source_field: SurfaceProbeFieldBuffers,
        source_attachments: SurfaceProbeAttachments,
        *,
        project_lighting: bool = True,
    ) -> SurfaceProbeFieldBuffers:
        if source_field.desc != DIFFUSE_PRT_L2_RGB_FIELD:
            raise ValueError(
                "SurfaceProbePrtEvaluator requires the L2 RGB PRT field"
            )
        usage = (
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource
        )
        lighting_sh = render_data.get_buffer(
            "surface_probe_renderer.prt.lighting_sh",
            usage=usage,
            struct_size=16,
            element_count=9,
            label="surface_probe_prt_lighting_sh",
        )
        evaluated_field = self.acquire_evaluated_field(render_data)
        if project_lighting:
            with command_encoder.begin_compute_pass() as pass_encoder:
                cursor = spy.ShaderCursor(
                    pass_encoder.bind_pipeline(self.project_pipeline)
                )
                cursor.g_lighting_sh = lighting_sh
                cursor.g_lighting_projection_sample_count = (
                    self.LIGHTING_PROJECTION_SAMPLE_COUNT
                )
                self.scene.bind(cursor.g_scene)
                pass_encoder.dispatch(thread_count=[9, 1, 1])
        with command_encoder.begin_compute_pass() as pass_encoder:
            cursor = spy.ShaderCursor(
                pass_encoder.bind_pipeline(self.evaluate_pipeline)
            )
            cursor.g_lighting_sh = lighting_sh
            cursor.g_prt_field_values = source_field.values
            cursor.g_prt_sample_counts = source_field.sample_counts
            cursor.g_prt_static_source_rgb = (
                source_attachments.static_source_rgb
            )
            cursor.g_evaluated_field_values = (
                evaluated_field.values
            )
            cursor.g_evaluated_sample_counts = (
                evaluated_field.sample_counts
            )
            cursor.g_probe_count = self.geometry.probe_count
            pass_encoder.dispatch(
                thread_count=[self.geometry.probe_count, 1, 1]
            )
        return evaluated_field
