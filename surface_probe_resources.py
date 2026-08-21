from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy

if TYPE_CHECKING:
    from surface_probes import SurfaceProbeLayout


SURFACE_PROBE_METADATA_SIZE = 48
SURFACE_PROBE_NODE_SIZE = 16
SURFACE_PROBE_INSTANCE_SIZE = 48
SURFACE_PROBE_RADIAL_DEPTH_DIM = 4
SURFACE_PROBE_RADIAL_MOMENT_SIZE = (
    SURFACE_PROBE_RADIAL_DEPTH_DIM * SURFACE_PROBE_RADIAL_DEPTH_DIM * 4
)


@dataclass(frozen=True)
class SurfaceProbeGpuGeometry:
    """Immutable GPU representation of a Surface Probe spatial domain.

    Probe fields deliberately do not live here.  A single placement, octree,
    and triangle map can therefore back irradiance, PRT, or other fields.
    """

    probes: spy.Buffer
    nodes: spy.Buffer
    instances: spy.Buffer
    triangle_vertex_probes: spy.Buffer
    probe_count: int
    instance_count: int

    @classmethod
    def create(
        cls,
        device: spy.Device,
        layout: "SurfaceProbeLayout",
        *,
        profile_sink: list[tuple[str, float]] | None = None,
    ) -> "SurfaceProbeGpuGeometry":
        stage_start = (
            time.perf_counter() if profile_sink is not None else 0.0
        )

        def profile_mark(label: str) -> None:
            nonlocal stage_start
            if profile_sink is None:
                return
            now = time.perf_counter()
            profile_sink.append((label, now - stage_start))
            stage_start = now

        probe_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_metadata",
            data=np.ascontiguousarray(layout.probes).view(np.uint8),
        )
        profile_mark("gpu_probe_metadata_buffer")
        node_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_nodes",
            data=np.ascontiguousarray(layout.nodes),
        )
        profile_mark("gpu_octree_buffer")
        instance_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_instances",
            data=np.ascontiguousarray(layout.instance_gpu_data).view(np.uint8),
        )
        profile_mark("gpu_instance_buffer")
        triangle_vertex_probe_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_triangle_vertex_map",
            data=np.ascontiguousarray(layout.triangle_vertex_probes),
        )
        profile_mark("gpu_triangle_vertex_map_buffer")
        return cls(
            probes=probe_buffer,
            nodes=node_buffer,
            instances=instance_buffer,
            triangle_vertex_probes=triangle_vertex_probe_buffer,
            probe_count=int(layout.total_probe_count),
            instance_count=len(layout.instance_infos),
        )
