from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import slangpy as spy

from visibility_baker import VisibilitySampleCones, make_visibility_local_directions
from pmr_visibility_reference import (
    PMR_DEFAULT_RAY_COUNT,
    PMR_DEFAULT_RAY_LENGTH,
    PMR_DEFAULT_SELF_BIAS,
)


ROOT = Path(__file__).resolve().parents[1]
SHADER_PATH = ROOT / "vertex_visibility_raw.slang"


def _as_float4(values, name: str, *, required_channels: int | None = None) -> npt.NDArray[np.float32]:
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise ValueError(f"{name} must have shape (N, 3) or (N, 4)")
    if required_channels is not None and array.shape[1] != required_channels:
        raise ValueError(f"{name} must have shape (N, {required_channels})")
    result = np.zeros((array.shape[0], 4), dtype=np.float32)
    result[:, : array.shape[1]] = array
    return result


def _read_float4_buffer(buffer: spy.Buffer, count: int) -> npt.NDArray[np.float32]:
    data = np.asarray(buffer.to_numpy())
    if data.dtype != np.float32:
        data = data.view(np.float32)
    return np.ascontiguousarray(data.reshape(-1, 4)[:count], dtype=np.float32)


class HWRTVisibilityRawSampler:
    def __init__(self, device: spy.Device):
        self.device = device
        self.program = self.device.load_program(str(SHADER_PATH), ["compute_main"])
        self.pipeline = self.device.create_compute_pipeline(self.program)

    def sample(
        self,
        scene,
        sample_positions,
        sample_normals,
        sample_tangents,
        *,
        ray_count: int = 128,
        local_directions=None,
        max_distance: float = np.inf,
        self_bias: float | None = None,
    ) -> VisibilitySampleCones:
        positions = _as_float4(sample_positions, "sample_positions")
        normals = _as_float4(sample_normals, "sample_normals")
        tangents = _as_float4(sample_tangents, "sample_tangents", required_channels=4)
        sample_count = int(positions.shape[0])
        if normals.shape[0] != sample_count or tangents.shape[0] != sample_count:
            raise ValueError("sample_positions, sample_normals, and sample_tangents must have matching lengths")
        if sample_count == 0:
            return VisibilitySampleCones(
                directions=np.zeros((0, 3), dtype=np.float32),
                aperture_radians=np.zeros((0,), dtype=np.float32),
                scale=np.zeros((0,), dtype=np.float32),
                visible_fraction=np.zeros((0,), dtype=np.float32),
            )

        if local_directions is None:
            local_directions = make_visibility_local_directions(max(1, int(ray_count)))
        local_dirs = _as_float4(local_directions, "local_directions")
        if local_dirs.shape[0] == 0:
            raise ValueError("local_directions must contain at least one direction")

        if self_bias is None:
            bounds_min = np.min(positions[:, :3], axis=0)
            bounds_max = np.max(positions[:, :3], axis=0)
            self_bias = max(float(np.linalg.norm(bounds_max - bounds_min)), 1.0) * 1e-5
        ray_t_max = float(max_distance)
        if not np.isfinite(ray_t_max) or ray_t_max <= 0.0:
            ray_t_max = 1e30

        position_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="visibility_sample_positions",
            data=positions,
        )
        normal_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="visibility_sample_normals",
            data=normals,
        )
        tangent_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="visibility_sample_tangents",
            data=tangents,
        )
        local_dir_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="visibility_local_directions",
            data=local_dirs,
        )
        output_cone0 = self.device.create_buffer(
            usage=spy.BufferUsage.unordered_access,
            label="visibility_raw_cone0",
            size=sample_count * 16,
        )
        output_cone1 = self.device.create_buffer(
            usage=spy.BufferUsage.unordered_access,
            label="visibility_raw_cone1",
            size=sample_count * 16,
        )

        command_encoder = self.device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_sample_positions = position_buffer
            cursor.g_sample_normals = normal_buffer
            cursor.g_sample_tangents = tangent_buffer
            cursor.g_local_directions = local_dir_buffer
            cursor.g_output_cone0 = output_cone0
            cursor.g_output_cone1 = output_cone1
            cursor.g_sample_count = sample_count
            cursor.g_ray_count = int(local_dirs.shape[0])
            cursor.g_self_bias = float(max(0.0, self_bias))
            cursor.g_max_distance = float(ray_t_max)
            scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[sample_count, 1, 1])
        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait()

        cone0 = _read_float4_buffer(output_cone0, sample_count)
        cone1 = _read_float4_buffer(output_cone1, sample_count)
        return VisibilitySampleCones(
            directions=cone0[:, :3].astype(np.float32, copy=False),
            aperture_radians=cone0[:, 3].astype(np.float32, copy=False),
            scale=cone1[:, 0].astype(np.float32, copy=False),
            visible_fraction=cone1[:, 1].astype(np.float32, copy=False),
        )


class HWRTPMRVisibilitySHSampler:
    """HWRT equivalent of PMR's ComputeVisibilitySH.raytrace pass."""

    def __init__(self, device: spy.Device):
        self.device = device
        self.program = self.device.load_program(str(SHADER_PATH), ["compute_pmr_sh"])
        self.pipeline = self.device.create_compute_pipeline(self.program)

    def sample(
        self,
        scene,
        sample_positions,
        sample_normals,
        *,
        ray_count: int = PMR_DEFAULT_RAY_COUNT,
        max_distance: float = PMR_DEFAULT_RAY_LENGTH,
        self_bias: float = PMR_DEFAULT_SELF_BIAS,
        rays_per_dispatch: int = 64,
        max_points_per_dispatch: int = 1024 * 1024,
    ) -> npt.NDArray[np.float32]:
        positions = _as_float4(sample_positions, "sample_positions")
        normals = _as_float4(sample_normals, "sample_normals")
        sample_count = int(positions.shape[0])
        if normals.shape[0] != sample_count:
            raise ValueError("sample_positions and sample_normals must have matching lengths")
        if sample_count == 0:
            return np.zeros((0, 16), dtype=np.float32)

        max_points_per_dispatch = int(max_points_per_dispatch)
        if max_points_per_dispatch > 0 and sample_count > max_points_per_dispatch:
            chunks = []
            for begin in range(0, sample_count, max_points_per_dispatch):
                end = min(sample_count, begin + max_points_per_dispatch)
                chunks.append(
                    self.sample(
                        scene,
                        positions[begin:end, :3],
                        normals[begin:end, :3],
                        ray_count=ray_count,
                        max_distance=max_distance,
                        self_bias=self_bias,
                        rays_per_dispatch=rays_per_dispatch,
                        max_points_per_dispatch=0,
                    )
                )
            return np.concatenate(chunks, axis=0)

        ray_count = max(1, int(ray_count))
        rays_per_dispatch = max(1, int(rays_per_dispatch))
        ray_t_max = float(max_distance)
        if not np.isfinite(ray_t_max) or ray_t_max <= 0.0:
            ray_t_max = 1e30

        position_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="pmr_visibility_sample_positions",
            data=positions,
        )
        normal_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="pmr_visibility_sample_normals",
            data=normals,
        )
        outputs = [
            self.device.create_buffer(
                usage=spy.BufferUsage.unordered_access,
                label=f"pmr_visibility_sh{index}",
                size=sample_count * 16,
            )
            for index in range(4)
        ]

        for ray_offset in range(0, ray_count, rays_per_dispatch):
            command_encoder = self.device.create_command_encoder()
            with command_encoder.begin_compute_pass() as pass_encoder:
                shader_object = pass_encoder.bind_pipeline(self.pipeline)
                cursor = spy.ShaderCursor(shader_object)
                cursor.g_sample_positions = position_buffer
                cursor.g_sample_normals = normal_buffer
                cursor.g_output_sh0 = outputs[0]
                cursor.g_output_sh1 = outputs[1]
                cursor.g_output_sh2 = outputs[2]
                cursor.g_output_sh3 = outputs[3]
                cursor.g_sample_count = sample_count
                cursor.g_ray_count = ray_count
                cursor.g_ray_offset = ray_offset
                cursor.g_ray_batch_count = min(rays_per_dispatch, ray_count - ray_offset)
                cursor.g_self_bias = float(max(0.0, self_bias))
                cursor.g_max_distance = ray_t_max
                scene.bind(cursor.g_scene)
                pass_encoder.dispatch(thread_count=[sample_count, 1, 1])
            self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait()

        return np.concatenate(
            [_read_float4_buffer(output, sample_count) for output in outputs],
            axis=1,
        ).astype(np.float32, copy=False)


def sample_visibility_cones_hwrt(
    device: spy.Device,
    scene,
    sample_positions,
    sample_normals,
    sample_tangents,
    *,
    ray_count: int = 128,
    local_directions=None,
    max_distance: float = np.inf,
    self_bias: float | None = None,
) -> VisibilitySampleCones:
    sampler = HWRTVisibilityRawSampler(device)
    return sampler.sample(
        scene,
        sample_positions,
        sample_normals,
        sample_tangents,
        ray_count=ray_count,
        local_directions=local_directions,
        max_distance=max_distance,
        self_bias=self_bias,
    )


def sample_pmr_visibility_sh_hwrt(
    device: spy.Device,
    scene,
    sample_positions,
    sample_normals,
    *,
    ray_count: int = PMR_DEFAULT_RAY_COUNT,
    max_distance: float = PMR_DEFAULT_RAY_LENGTH,
    self_bias: float = PMR_DEFAULT_SELF_BIAS,
    rays_per_dispatch: int = 64,
    max_points_per_dispatch: int = 1024 * 1024,
) -> npt.NDArray[np.float32]:
    return HWRTPMRVisibilitySHSampler(device).sample(
        scene,
        sample_positions,
        sample_normals,
        ray_count=ray_count,
        max_distance=max_distance,
        self_bias=self_bias,
        rays_per_dispatch=rays_per_dispatch,
        max_points_per_dispatch=max_points_per_dispatch,
    )


class HWRTModelScene:
    """Minimal opaque RTAS for a vertex_baker.model.Model."""

    def __init__(self, device: spy.Device, model):
        self.device = device
        position_parts = []
        index_parts = []
        vertex_offset = 0
        for mesh in model.meshes:
            positions = np.ascontiguousarray(mesh.positions, dtype=np.float32)
            indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
            if positions.shape[0] == 0 or indices.shape[0] == 0:
                continue
            position_parts.append(positions)
            index_parts.append(indices + np.uint32(vertex_offset))
            vertex_offset += int(positions.shape[0])
        if not position_parts:
            raise ValueError("model contains no triangles for HWRT visibility")

        positions = np.ascontiguousarray(np.concatenate(position_parts, axis=0), dtype=np.float32)
        indices = np.ascontiguousarray(np.concatenate(index_parts, axis=0).reshape(-1), dtype=np.uint32)
        self.vertex_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="pmr_visibility_rtas_vertices",
            data=positions,
        )
        self.index_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="pmr_visibility_rtas_indices",
            data=indices,
        )

        triangle_input = spy.AccelerationStructureBuildInputTriangles(
            {
                "vertex_buffers": [{"buffer": self.vertex_buffer, "offset": 0}],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": int(positions.shape[0]),
                "vertex_stride": 12,
                "index_buffer": {"buffer": self.index_buffer, "offset": 0},
                "index_format": spy.IndexFormat.uint32,
                "index_count": int(indices.shape[0]),
                # PMR's mesh visibility rays treat assembled geometry as opaque blockers.
                "flags": spy.AccelerationStructureGeometryFlags.opaque,
            }
        )
        blas_desc = spy.AccelerationStructureBuildDesc({"inputs": [triangle_input]})
        blas_sizes = device.get_acceleration_structure_sizes(blas_desc)
        self.blas_scratch = device.create_buffer(
            size=blas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="pmr_visibility_blas_scratch",
        )
        self.blas = device.create_acceleration_structure(
            size=blas_sizes.acceleration_structure_size,
            label="pmr_visibility_blas",
        )
        command_encoder = device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=blas_desc,
            dst=self.blas,
            src=None,
            scratch_buffer=self.blas_scratch,
        )
        device.submit_command_buffer(command_encoder.finish())

        self.instance_list = device.create_acceleration_structure_instance_list(size=1)
        self.instance_list.write(
            0,
            {
                "transform": spy.float3x4.identity(),
                "instance_id": 0,
                "instance_mask": 0xFF,
                "instance_contribution_to_hit_group_index": 0,
                "flags": spy.AccelerationStructureInstanceFlags.none,
                "acceleration_structure": self.blas.handle,
            },
        )
        tlas_desc = spy.AccelerationStructureBuildDesc(
            {"inputs": [self.instance_list.build_input_instances()]}
        )
        tlas_sizes = device.get_acceleration_structure_sizes(tlas_desc)
        self.tlas_scratch = device.create_buffer(
            size=tlas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="pmr_visibility_tlas_scratch",
        )
        self.tlas = device.create_acceleration_structure(
            size=tlas_sizes.acceleration_structure_size,
            label="pmr_visibility_tlas",
        )
        command_encoder = device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=tlas_desc,
            dst=self.tlas,
            src=None,
            scratch_buffer=self.tlas_scratch,
        )
        device.submit_command_buffer(command_encoder.finish())
        device.wait()

    def bind(self, cursor: spy.ShaderCursor) -> None:
        cursor.tlas = self.tlas


def bake_pmr_visibility_cones_hwrt(
    device: spy.Device,
    model,
    *,
    scene=None,
    samples_per_triangle: int = 256,
    visibility_ray_count: int = PMR_DEFAULT_RAY_COUNT,
    ray_length: float = PMR_DEFAULT_RAY_LENGTH,
    self_bias: float = PMR_DEFAULT_SELF_BIAS,
    edge_regularization: float = 0.05,
    proxy_voxel_size_mm: float = 0.1,
    proxy_compare_normals: bool = True,
    fit_backend: str = "native",
    build_native_first: bool = False,
    rays_per_dispatch: int = 64,
    max_points_per_dispatch: int = 1024 * 1024,
):
    from pmr_visibility_reference import sample_model_surface_pmr
    from visibility_baker import fit_visibility_sh_pmr

    sampling = sample_model_surface_pmr(
        model,
        samples_per_triangle=max(1, int(samples_per_triangle)),
        voxel_size_mm=float(proxy_voxel_size_mm),
        compare_normals=bool(proxy_compare_normals),
    )
    trace_scene = scene if scene is not None else HWRTModelScene(device, model)
    sample_sh = sample_pmr_visibility_sh_hwrt(
        device,
        trace_scene,
        sampling.samples.positions,
        sampling.sample_normals,
        ray_count=max(1, int(visibility_ray_count)),
        max_distance=float(ray_length),
        self_bias=max(0.0, float(self_bias)),
        rays_per_dispatch=max(1, int(rays_per_dispatch)),
        max_points_per_dispatch=max(0, int(max_points_per_dispatch)),
    )
    return fit_visibility_sh_pmr(
        model,
        sampling,
        sample_sh,
        edge_regularization=max(0.0, float(edge_regularization)),
        fit_backend=fit_backend,
        build_native_first=build_native_first,
    )
