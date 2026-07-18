from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import slangpy as spy

from scene_node import SceneNode


MESH_COLORS_PAYLOAD_SIZE = 16
MESH_COLORS_INVALID_OFFSET = 0xFFFFFFFF
MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED = 1 << 0


def colors_per_patch(resolution: int) -> int:
    """Return the number of barycentric lattice points for one triangle."""
    resolution = int(resolution)
    if resolution < 1:
        raise ValueError("resolution must be positive")
    return (resolution + 1) * (resolution + 2) // 2


def address_to_ij(address: int, resolution: int) -> tuple[int, int]:
    """Convert a triangle-local linear address to barycentric grid coordinates."""
    address = int(address)
    resolution = int(resolution)
    if address < 0 or address >= colors_per_patch(resolution):
        raise ValueError("address is outside the triangle patch")

    b = 2 * resolution + 3
    discriminant = b * b - 8 * address
    i = int((b - math.sqrt(discriminant)) * 0.5)
    cumulative = i * (resolution + 1) - i * (i - 1) // 2
    while cumulative > address:
        i -= 1
        cumulative = i * (resolution + 1) - i * (i - 1) // 2
    while i < resolution:
        next_cumulative = (i + 1) * (resolution + 1) - (i + 1) * i // 2
        if next_cumulative > address:
            break
        i += 1
        cumulative = next_cumulative
    return i, address - cumulative


def ij_to_address(i: int, j: int, resolution: int) -> int:
    i = int(i)
    j = int(j)
    resolution = int(resolution)
    if i < 0 or i > resolution or j < 0 or j > resolution - i:
        raise ValueError("(i, j) is outside the triangle patch")
    return i * (resolution + 1) - i * (i - 1) // 2 + j


def barycentric_at(i: int, j: int, resolution: int) -> tuple[float, float, float]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    u = float(i) / float(resolution)
    v = float(j) / float(resolution)
    return u, v, 1.0 - u - v


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def previous_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value.bit_length() - 1)


@dataclass(frozen=True)
class MeshColorsFaceInfo:
    address: int
    resolution: int

    def pack(self) -> bytes:
        return struct.pack("II", self.address, self.resolution)


@dataclass(frozen=True)
class MeshColorsInstanceInfo:
    face_offset: int
    face_count: int
    texel_offset: int
    texel_count: int

    def pack(self) -> bytes:
        return struct.pack(
            "IIII",
            self.face_offset,
            self.face_count,
            self.texel_offset,
            self.texel_count,
        )


@dataclass(frozen=True)
class MeshColorsSideInfo:
    back_texel_offset: int
    flags: int

    @property
    def is_double_sided(self) -> bool:
        return bool(self.flags & MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED)

    def pack(self) -> bytes:
        return struct.pack("II", self.back_texel_offset, self.flags)


@dataclass(frozen=True)
class MeshColorsLayout:
    face_infos: tuple[MeshColorsFaceInfo, ...]
    instance_infos: tuple[MeshColorsInstanceInfo, ...]
    side_infos: tuple[MeshColorsSideInfo, ...]
    total_surface_texels: int
    total_payload_count: int
    texels_per_unit: float
    min_resolution: int
    max_resolution: int

    @property
    def total_texel_count(self) -> int:
        """Compatibility alias for the number of surface lattice texels."""
        return self.total_surface_texels

    @staticmethod
    def _world_position(transform: spy.float4x4, position: np.ndarray) -> np.ndarray:
        p = spy.math.mul(
            transform,
            spy.float4(float(position[0]), float(position[1]), float(position[2]), 1.0),
        )
        return np.asarray((float(p[0]), float(p[1]), float(p[2])), dtype=np.float64)

    @classmethod
    def build(
        cls,
        scene_node: SceneNode,
        *,
        texels_per_unit: float = 16.0,
        min_resolution: int = 4,
        max_resolution: int = 64,
        max_total_texels: int = 16_777_216,
    ) -> "MeshColorsLayout":
        texels_per_unit = max(float(texels_per_unit), 1e-6)
        min_resolution = next_power_of_two(max(1, int(min_resolution)))
        max_resolution = previous_power_of_two(max(1, int(max_resolution)))
        if min_resolution > max_resolution:
            raise ValueError(
                "Mesh Colors min_resolution must not exceed max_resolution "
                "after power-of-two normalization"
            )
        max_total_texels = max(1, int(max_total_texels))

        face_infos: list[MeshColorsFaceInfo] = []
        instance_infos: list[MeshColorsInstanceInfo] = []
        side_infos: list[MeshColorsSideInfo] = []
        total_surface_texels = 0
        total_payload_count = 0

        for mesh_id, material_id, transform_id in scene_node.instances:
            mesh = scene_node.meshes[mesh_id]
            material = scene_node.materials[material_id]
            is_double_sided = bool(getattr(material, "double_sided", False))
            payload_multiplier = 2 if is_double_sided else 1
            transform = scene_node.transforms[transform_id].matrix
            face_offset = len(face_infos)
            instance_texel_offset = total_payload_count
            local_address = 0

            for triangle in mesh.indices:
                positions = [
                    cls._world_position(transform, mesh.vertices[int(index), 0:3])
                    for index in triangle
                ]
                max_edge = max(
                    float(np.linalg.norm(positions[1] - positions[0])),
                    float(np.linalg.norm(positions[2] - positions[1])),
                    float(np.linalg.norm(positions[0] - positions[2])),
                )
                resolution = next_power_of_two(
                    max(min_resolution, int(math.ceil(max_edge * texels_per_unit)))
                )
                resolution = min(resolution, max_resolution)
                patch_texels = colors_per_patch(resolution)

                if (
                    total_payload_count
                    + (local_address + patch_texels) * payload_multiplier
                    > max_total_texels
                ):
                    candidate = resolution
                    while candidate > min_resolution:
                        candidate //= 2
                        patch_texels = colors_per_patch(candidate)
                        if (
                            total_payload_count
                            + (local_address + patch_texels) * payload_multiplier
                            <= max_total_texels
                        ):
                            resolution = candidate
                            break
                    else:
                        raise ValueError(
                            "Mesh Colors texel budget exceeded; lower --texture-space-texels-per-unit "
                            "or --texture-space-min-resolution, or increase --texture-space-max-texels"
                        )

                face_infos.append(MeshColorsFaceInfo(local_address, resolution))
                local_address += patch_texels

            instance_infos.append(
                MeshColorsInstanceInfo(
                    face_offset=face_offset,
                    face_count=mesh.triangle_count,
                    texel_offset=instance_texel_offset,
                    texel_count=local_address,
                )
            )
            back_texel_offset = (
                instance_texel_offset + local_address
                if is_double_sided
                else MESH_COLORS_INVALID_OFFSET
            )
            side_infos.append(
                MeshColorsSideInfo(
                    back_texel_offset=back_texel_offset,
                    flags=(
                        MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED
                        if is_double_sided
                        else 0
                    ),
                )
            )
            total_surface_texels += local_address
            total_payload_count += local_address * payload_multiplier

        if total_surface_texels == 0:
            raise ValueError("Texture-space rendering requires at least one triangle")

        return cls(
            face_infos=tuple(face_infos),
            instance_infos=tuple(instance_infos),
            side_infos=tuple(side_infos),
            total_surface_texels=total_surface_texels,
            total_payload_count=total_payload_count,
            texels_per_unit=texels_per_unit,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
        )

    def create_gpu_buffers(
        self, device: spy.Device
    ) -> tuple[spy.Buffer, spy.Buffer, spy.Buffer]:
        face_data = np.frombuffer(
            b"".join(info.pack() for info in self.face_infos), dtype=np.uint8
        ).copy()
        instance_data = np.frombuffer(
            b"".join(info.pack() for info in self.instance_infos), dtype=np.uint8
        ).copy()
        side_data = np.frombuffer(
            b"".join(info.pack() for info in self.side_infos), dtype=np.uint8
        ).copy()
        face_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_colors_face_infos",
            data=face_data,
        )
        instance_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_colors_instance_infos",
            data=instance_data,
        )
        side_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_colors_side_infos",
            data=side_data,
        )
        return face_buffer, instance_buffer, side_buffer
