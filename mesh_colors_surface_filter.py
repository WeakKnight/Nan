from __future__ import annotations

import math

import numpy as np
import slangpy as spy

from mesh_colors import MeshColorsLayout
from scene import Scene


class MeshColorsSurfaceFilter:
    """One-ring, normal-aware Mesh Colors diffusion pass."""

    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        layout: MeshColorsLayout,
        *,
        face_infos_buffer: spy.Buffer | None = None,
        instance_infos_buffer: spy.Buffer | None = None,
        side_infos_buffer: spy.Buffer | None = None,
        adjacency_infos_buffer: spy.Buffer | None = None,
        face_normals_buffer: spy.Buffer | None = None,
    ):
        self.layout = layout
        self.program = device.load_program(
            "mesh_colors_surface_filter.slang", ["compute_main"]
        )
        self.pipeline = device.create_compute_pipeline(self.program)

        if (
            face_infos_buffer is None
            or instance_infos_buffer is None
            or side_infos_buffer is None
        ):
            (
                self.face_infos_buffer,
                self.instance_infos_buffer,
                self.side_infos_buffer,
            ) = layout.create_gpu_buffers(device)
        else:
            self.face_infos_buffer = face_infos_buffer
            self.instance_infos_buffer = instance_infos_buffer
            self.side_infos_buffer = side_infos_buffer
        self.adjacency_infos_buffer = (
            layout.create_adjacency_gpu_buffer(device)
            if adjacency_infos_buffer is None
            else adjacency_infos_buffer
        )
        if face_normals_buffer is None:
            normals: list[tuple[float, float, float, float]] = []
            scene_node = scene.scene_node
            for mesh_id, _, transform_id in scene_node.instances:
                mesh = scene_node.meshes[mesh_id]
                transform = scene_node.transforms[transform_id].matrix
                for triangle in mesh.indices:
                    positions = [
                        MeshColorsLayout._world_position(
                            transform, mesh.vertices[int(index), 0:3]
                        )
                        for index in triangle
                    ]
                    normal = np.cross(
                        positions[1] - positions[0],
                        positions[2] - positions[0],
                    )
                    length = float(np.linalg.norm(normal))
                    if length > 1e-20:
                        normal /= length
                    else:
                        normal[:] = 0.0
                    normals.append(
                        (
                            float(normal[0]),
                            float(normal[1]),
                            float(normal[2]),
                            0.0,
                        )
                    )
            if len(normals) != len(layout.face_infos):
                raise ValueError("Mesh Colors face-normal count does not match layout")
            self.face_normals_buffer = device.create_buffer(
                usage=spy.BufferUsage.shader_resource,
                label="mesh_colors_face_normals",
                data=np.asarray(normals, dtype=np.float32),
            )
        else:
            self.face_normals_buffer = face_normals_buffer

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        source: spy.Buffer,
        destination: spy.Buffer,
        *,
        spatial_sigma: float = 1.0,
        normal_sigma_radians: float = math.radians(30.0),
    ) -> None:
        if source is destination:
            raise ValueError("Mesh Colors surface filtering requires ping-pong buffers")

        spatial_sigma = max(float(spatial_sigma), 1e-6)
        neighbor_weight = math.exp(-0.5 / (spatial_sigma * spatial_sigma))
        normal_sigma_radians = max(float(normal_sigma_radians), 0.0)

        with command_encoder.begin_compute_pass() as pass_encoder:
            for instance_index, instance_info in enumerate(
                self.layout.instance_infos
            ):
                if instance_info.texel_count <= 0:
                    continue
                shader_object = pass_encoder.bind_pipeline(self.pipeline)
                cursor = spy.ShaderCursor(shader_object)
                cursor.g_source = source
                cursor.g_destination = destination
                cursor.g_mesh_colors_face_infos = self.face_infos_buffer
                cursor.g_mesh_colors_instance_infos = self.instance_infos_buffer
                cursor.g_mesh_colors_side_infos = self.side_infos_buffer
                cursor.g_mesh_colors_adjacency_infos = self.adjacency_infos_buffer
                cursor.g_mesh_colors_face_normals = self.face_normals_buffer
                cursor.g_instance_index = instance_index
                cursor.g_neighbor_weight = neighbor_weight
                cursor.g_normal_sigma_radians = normal_sigma_radians
                pass_encoder.dispatch(
                    thread_count=[instance_info.texel_count, 1, 1]
                )
