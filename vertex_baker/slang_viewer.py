from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import slangpy as spy


ROOT = Path(__file__).resolve().parents[1]
VERTEX_BAKER_DIR = Path(__file__).resolve().parent
SHADER_PATH = VERTEX_BAKER_DIR / "slang_viewer.slang"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VERTEX_BAKER_DIR) not in sys.path:
    sys.path.insert(0, str(VERTEX_BAKER_DIR))

from camera import Camera
from ibl_precompute import EnvironmentIBL
from model import Model, load_gltf_model
from visibility_baker import vertex_visibility_preview_values


VIEW_MODE_NAMES = (
    "PBR",
    "Baked value",
    "Base color",
    "World normal",
    "Linear depth",
    "Mesh ID",
    "Cone aperture",
    "Cone scale",
    "Roughness",
    "Metallic",
    "Emissive",
    "Material AO",
    "PMR diffuse AO",
    "PMR specular occlusion",
)
INVALID_ID = np.uint32(0xFFFFFFFF)
DEFAULT_ENVIRONMENT = VERTEX_BAKER_DIR / "bloem_field_sunrise_2k.hdr"


def _normalize_rows(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    result = np.ascontiguousarray(values, dtype=np.float32).copy()
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > 1e-20
    result[valid] /= lengths[valid, None]
    result[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return result


def _vertex_value_float4(
    model: Model,
    vertex_values: list[npt.NDArray[np.float32]] | None,
) -> npt.NDArray[np.float32]:
    parts = []
    for mesh_index, mesh in enumerate(model.meshes):
        if vertex_values is None:
            material = model.materials[mesh.material_index]
            rgb = np.tile(np.asarray(material.base_color[:3], dtype=np.float32), (mesh.positions.shape[0], 1))
        else:
            value = np.asarray(vertex_values[mesh_index], dtype=np.float32)
            if value.ndim != 2 or value.shape[0] != mesh.positions.shape[0] or value.shape[1] not in (1, 3, 4):
                raise ValueError(
                    f"vertex values for mesh {mesh.name!r} must have shape ({mesh.positions.shape[0]}, 1|3|4)"
                )
            if value.shape[1] == 1:
                rgb = np.repeat(value, 3, axis=1)
            else:
                rgb = value[:, :3]
        packed = np.ones((mesh.positions.shape[0], 4), dtype=np.float32)
        packed[:, :3] = rgb
        parts.append(packed)
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def _vertex_cone_float4(
    model: Model,
    vertex_cones: list[npt.NDArray[np.float32]] | None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    cone0_parts = []
    cone1_parts = []
    for mesh_index, mesh in enumerate(model.meshes):
        count = mesh.positions.shape[0]
        cone0 = np.zeros((count, 4), dtype=np.float32)
        cone1 = np.zeros((count, 4), dtype=np.float32)
        if vertex_cones is None:
            cone0[:, :3] = _normalize_rows(mesh.normals)
        else:
            cones = np.asarray(vertex_cones[mesh_index], dtype=np.float32)
            if cones.shape != (count, 5):
                raise ValueError(f"vertex cones for mesh {mesh.name!r} must have shape ({count}, 5)")
            cone0[:, :3] = _normalize_rows(cones[:, :3])
            cone0[:, 3] = np.clip(cones[:, 3], 0.0, 0.5 * np.pi)
            cone1[:, 0] = np.clip(np.where(np.isfinite(cones[:, 4]), cones[:, 4], 0.0), 0.0, 1.0)
        cone0_parts.append(cone0)
        cone1_parts.append(cone1)
    return (
        np.ascontiguousarray(np.concatenate(cone0_parts, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(cone1_parts, axis=0), dtype=np.float32),
    )


class BakerRTScene:
    """GPU scene whose indices exactly match vertex_baker.Model."""

    def __init__(
        self,
        device: spy.Device,
        model: Model,
        vertex_values: list[npt.NDArray[np.float32]] | None = None,
        vertex_cones: list[npt.NDArray[np.float32]] | None = None,
        shader_session: spy.SlangSession | None = None,
        use_ray_query: bool = True,
    ) -> None:
        if vertex_values is not None and len(vertex_values) != len(model.meshes):
            raise ValueError("vertex_values must contain one array per mesh")
        if vertex_cones is not None and len(vertex_cones) != len(model.meshes):
            raise ValueError("vertex_cones must contain one array per mesh")

        self.device = device
        self.shader_session = shader_session or device.slang_session
        self.use_ray_query = bool(use_ray_query)
        self.model = model
        self.has_cones = vertex_cones is not None
        position_parts = []
        normal_parts = []
        index_parts = []
        triangle_mesh_parts = []
        triangle_material_parts = []
        vertex_mesh_parts = []
        vertex_local_parts = []
        uv_parts = []
        vertex_offset = 0
        for mesh_index, mesh in enumerate(model.meshes):
            positions = np.ascontiguousarray(mesh.positions, dtype=np.float32)
            normals = _normalize_rows(mesh.normals)
            indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
            position_parts.append(positions)
            normal_parts.append(normals)
            uv_parts.append(np.ascontiguousarray(mesh.uvs, dtype=np.float32))
            index_parts.append(indices + np.uint32(vertex_offset))
            triangle_mesh_parts.append(np.full((indices.shape[0],), mesh_index, dtype=np.uint32))
            triangle_material_parts.append(
                np.full((indices.shape[0],), mesh.material_index, dtype=np.uint32)
            )
            vertex_mesh_parts.append(np.full((positions.shape[0],), mesh_index, dtype=np.uint32))
            vertex_local_parts.append(np.arange(positions.shape[0], dtype=np.uint32))
            vertex_offset += positions.shape[0]

        self.positions = np.ascontiguousarray(np.concatenate(position_parts, axis=0), dtype=np.float32)
        self.normals = np.ascontiguousarray(np.concatenate(normal_parts, axis=0), dtype=np.float32)
        self.uvs = np.ascontiguousarray(np.concatenate(uv_parts, axis=0), dtype=np.float32)
        self.indices = np.ascontiguousarray(np.concatenate(index_parts, axis=0).reshape(-1), dtype=np.uint32)
        self.triangle_mesh_ids = np.ascontiguousarray(np.concatenate(triangle_mesh_parts), dtype=np.uint32)
        self.triangle_material_ids = np.ascontiguousarray(
            np.concatenate(triangle_material_parts), dtype=np.uint32
        )
        self.vertex_mesh_ids = np.ascontiguousarray(np.concatenate(vertex_mesh_parts), dtype=np.uint32)
        self.vertex_local_ids = np.ascontiguousarray(np.concatenate(vertex_local_parts), dtype=np.uint32)
        self.vertex_values = _vertex_value_float4(model, vertex_values)
        self.vertex_cone0, self.vertex_cone1 = _vertex_cone_float4(model, vertex_cones)
        self.bounds_min = np.asarray(model.bounds_min, dtype=np.float32)
        self.bounds_max = np.asarray(model.bounds_max, dtype=np.float32)

        raster_vertex_dtype = np.dtype(
            [
                ("position", np.float32, 3),
                ("normal", np.float32, 3),
                ("uv", np.float32, 2),
                ("baked", np.float32, 4),
                ("cone0", np.float32, 4),
                ("cone_scale", np.float32),
                ("bary", np.float32, 3),
                ("triangle_id", np.uint32),
            ],
            align=False,
        )
        raster_vertices = np.empty(self.indices.shape[0], dtype=raster_vertex_dtype)
        raster_vertices["position"] = self.positions[self.indices]
        raster_vertices["normal"] = self.normals[self.indices]
        raster_vertices["uv"] = self.uvs[self.indices]
        raster_vertices["baked"] = self.vertex_values[self.indices]
        raster_vertices["cone0"] = self.vertex_cone0[self.indices]
        raster_vertices["cone_scale"] = self.vertex_cone1[self.indices, 0]
        raster_vertices["bary"] = np.tile(np.eye(3, dtype=np.float32), (self.indices.shape[0] // 3, 1))
        raster_vertices["triangle_id"] = np.repeat(
            np.arange(self.indices.shape[0] // 3, dtype=np.uint32),
            3,
        )
        self.raster_vertex_stride = raster_vertex_dtype.itemsize
        self.raster_vertex_offsets = {
            name: int(raster_vertex_dtype.fields[name][1])
            for name in raster_vertex_dtype.names or ()
        }
        self.raster_vertex_buffer = device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer,
            label="baker_viewer_raster_vertices",
            data=np.ascontiguousarray(raster_vertices).view(np.uint8),
        )

        self.position_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_positions",
            data=self.positions,
        )
        self.normal_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_normals",
            data=self.normals,
        )
        self.uv_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_uvs",
            data=self.uvs,
        )
        self.index_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_indices",
            data=self.indices,
        )
        self.triangle_mesh_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_triangle_mesh_ids",
            data=self.triangle_mesh_ids,
        )
        self.triangle_material_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_triangle_material_ids",
            data=self.triangle_material_ids,
        )
        self.vertex_mesh_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_vertex_mesh_ids",
            data=self.vertex_mesh_ids,
        )
        self.material_sampler = device.create_sampler(
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.wrap,
            address_w=spy.TextureAddressingMode.wrap,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        self.texture_loader = spy.TextureLoader(device)
        self._material_texture_cache: dict[tuple[str, bool], spy.TextureView] = {}
        self.material_textures: list[spy.Texture] = []
        self.material_texture_views: list[spy.TextureView] = []
        self.raster_material_bindings: list[dict[str, object]] = []
        self.material_buffer = self._create_material_buffer()
        self.value_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_vertex_values",
            data=self.vertex_values,
        )
        self.cone0_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_vertex_cone0",
            data=self.vertex_cone0,
        )
        self.cone1_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_vertex_cone1",
            data=self.vertex_cone1,
        )

        triangle_input = spy.AccelerationStructureBuildInputTriangles(
            {
                "vertex_buffers": [{"buffer": self.position_buffer, "offset": 0}],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": int(self.positions.shape[0]),
                "vertex_stride": 12,
                "index_buffer": {"buffer": self.index_buffer, "offset": 0},
                "index_format": spy.IndexFormat.uint32,
                "index_count": int(self.indices.shape[0]),
                "flags": spy.AccelerationStructureGeometryFlags.no_duplicate_any_hit_invocation,
            }
        )
        blas_desc = spy.AccelerationStructureBuildDesc({"inputs": [triangle_input]})
        blas_sizes = device.get_acceleration_structure_sizes(blas_desc)
        self.blas_scratch = device.create_buffer(
            size=blas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="baker_viewer_blas_scratch",
        )
        self.blas = device.create_acceleration_structure(
            size=blas_sizes.acceleration_structure_size,
            label="baker_viewer_blas",
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
            label="baker_viewer_tlas_scratch",
        )
        self.tlas = device.create_acceleration_structure(
            size=tlas_sizes.acceleration_structure_size,
            label="baker_viewer_tlas",
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

    def _material_texture(
        self,
        path: Path | None,
        pixels: npt.NDArray[np.float32] | None,
        *,
        srgb: bool,
        fallback: tuple[int, int, int, int],
        label: str,
    ) -> spy.TextureView:
        cache_key = (str(path.resolve()) if path is not None else f"__{label}_{fallback}", srgb)
        cached = self._material_texture_cache.get(cache_key)
        if cached is not None:
            return cached
        if path is not None and path.is_file():
            texture = self.texture_loader.load_texture(
                str(path),
                options={"load_as_srgb": srgb, "generate_mips": True},
            )
        else:
            data = (
                np.clip(np.asarray(pixels) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
                if pixels is not None
                else np.asarray([[fallback]], dtype=np.uint8)
            )
            texture = self.device.create_texture(
                width=int(data.shape[1]),
                height=int(data.shape[0]),
                format=spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm,
                usage=spy.TextureUsage.shader_resource,
                data=data,
                label=label,
            )
        view = texture.create_view()
        self.material_textures.append(texture)
        self.material_texture_views.append(view)
        self._material_texture_cache[cache_key] = view
        return view

    def _create_material_buffer(self) -> spy.Buffer:
        module = self.shader_session.load_module(str(SHADER_PATH))
        buffer_type = module.layout.find_type_by_name("StructuredBuffer<ViewerMaterial>")
        if buffer_type is None:
            raise RuntimeError("Could not reflect ViewerMaterial from slang_viewer.slang")
        layout = module.layout.get_type_layout(buffer_type).element_type_layout
        buffer = self.device.create_buffer(
            size=max(1, len(self.model.materials)) * layout.stride,
            usage=spy.BufferUsage.shader_resource,
            label="baker_viewer_materials",
        )
        cursor = spy.BufferCursor(layout, buffer, load_before_write=False)
        for material_index, material in enumerate(self.model.materials):
            base_view = self._material_texture(
                material.base_color_texture_path,
                material.base_color_texture,
                srgb=True,
                fallback=(255, 255, 255, 255),
                label=f"baker_material_{material_index}_base_color",
            )
            mr_view = self._material_texture(
                material.metallic_roughness_texture_path,
                material.metallic_roughness_texture,
                srgb=False,
                fallback=(255, 255, 255, 255),
                label=f"baker_material_{material_index}_metallic_roughness",
            )
            normal_view = self._material_texture(
                material.normal_texture_path,
                material.normal_texture,
                srgb=False,
                fallback=(128, 128, 255, 255),
                label=f"baker_material_{material_index}_normal",
            )
            occlusion_view = self._material_texture(
                material.occlusion_texture_path,
                material.occlusion_texture,
                srgb=False,
                fallback=(255, 255, 255, 255),
                label=f"baker_material_{material_index}_occlusion",
            )
            emissive_view = self._material_texture(
                material.emissive_texture_path,
                material.emissive_texture,
                srgb=True,
                fallback=(0, 0, 0, 255),
                label=f"baker_material_{material_index}_emissive",
            )
            flags = 0
            flags |= 1 << 0 if material.base_color_texture is not None else 0
            flags |= 1 << 1 if material.metallic_roughness_texture is not None else 0
            flags |= 1 << 2 if material.normal_texture is not None else 0
            flags |= 1 << 3 if material.occlusion_texture is not None else 0
            flags |= 1 << 4 if material.emissive_texture is not None else 0
            flags |= 1 << 5 if material.double_sided else 0
            self.raster_material_bindings.append(
                {
                    "base_color_factor": np.asarray(material.base_color, dtype=np.float32),
                    "emissive_factor": np.asarray(
                        material.emissive if material.emissive is not None else [0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    "roughness_factor": float(material.roughness),
                    "metallic_factor": float(material.metallic),
                    "normal_scale": float(material.normal_scale),
                    "occlusion_strength": float(material.occlusion_strength),
                    "flags": flags,
                    "base_color_texture": base_view,
                    "metallic_roughness_texture": mr_view,
                    "normal_texture": normal_view,
                    "occlusion_texture": occlusion_view,
                    "emissive_texture": emissive_view,
                }
            )
            entry = cursor[material_index]
            entry.base_color_factor = spy.float4(np.asarray(material.base_color, dtype=np.float32))
            entry.emissive_factor = spy.float3(
                np.asarray(material.emissive if material.emissive is not None else [0.0, 0.0, 0.0], dtype=np.float32)
            )
            entry.roughness_factor = float(material.roughness)
            entry.metallic_factor = float(material.metallic)
            entry.normal_scale = float(material.normal_scale)
            entry.occlusion_strength = float(material.occlusion_strength)
            entry.flags = flags
            entry.base_color_texture = base_view.descriptor_handle_ro
            entry.metallic_roughness_texture = mr_view.descriptor_handle_ro
            entry.normal_texture = normal_view.descriptor_handle_ro
            entry.occlusion_texture = occlusion_view.descriptor_handle_ro
            entry.emissive_texture = emissive_view.descriptor_handle_ro
        cursor.apply()
        return buffer

    def bind_gbuffer(self, cursor: spy.ShaderCursor) -> None:
        if self.use_ray_query:
            cursor.g_tlas = self.tlas
        cursor.g_positions = self.position_buffer
        cursor.g_normals = self.normal_buffer
        cursor.g_uvs = self.uv_buffer
        cursor.g_indices = self.index_buffer
        cursor.g_triangle_mesh_ids = self.triangle_mesh_buffer
        cursor.g_triangle_material_ids = self.triangle_material_buffer
        cursor.g_materials = self.material_buffer
        cursor.g_material_sampler = self.material_sampler
        cursor.g_vertex_values = self.value_buffer
        cursor.g_vertex_cone0 = self.cone0_buffer
        cursor.g_vertex_cone1 = self.cone1_buffer

    def bind_raster_material(self, cursor: spy.ShaderCursor, material_index: int) -> None:
        material = self.raster_material_bindings[material_index]
        cursor.g_raster_base_color_factor = spy.float4(material["base_color_factor"])
        cursor.g_raster_emissive_factor = spy.float3(material["emissive_factor"])
        cursor.g_raster_roughness_factor = material["roughness_factor"]
        cursor.g_raster_metallic_factor = material["metallic_factor"]
        cursor.g_raster_normal_scale = material["normal_scale"]
        cursor.g_raster_occlusion_strength = material["occlusion_strength"]
        cursor.g_raster_material_flags = material["flags"]
        cursor.g_raster_base_color_texture = material["base_color_texture"]
        cursor.g_raster_metallic_roughness_texture = material["metallic_roughness_texture"]
        cursor.g_raster_normal_texture = material["normal_texture"]
        cursor.g_raster_occlusion_texture = material["occlusion_texture"]
        cursor.g_raster_emissive_texture = material["emissive_texture"]

    def selected_data(self, global_vertex_id: int) -> dict[str, object] | None:
        if global_vertex_id < 0 or global_vertex_id >= self.positions.shape[0]:
            return None
        mesh_id = int(self.vertex_mesh_ids[global_vertex_id])
        local_id = int(self.vertex_local_ids[global_vertex_id])
        return {
            "global_id": global_vertex_id,
            "mesh_id": mesh_id,
            "mesh_name": self.model.meshes[mesh_id].name,
            "local_id": local_id,
            "position": self.positions[global_vertex_id],
            "axis": self.vertex_cone0[global_vertex_id, :3],
            "aperture": float(self.vertex_cone0[global_vertex_id, 3]),
            "scale": float(self.vertex_cone1[global_vertex_id, 0]),
        }

    def initial_vertex(self) -> int:
        if not self.has_cones or self.positions.shape[0] == 0:
            return -1
        center = (self.bounds_min + self.bounds_max) * 0.5
        diagonal = max(float(np.linalg.norm(self.bounds_max - self.bounds_min)), 1e-6)
        distance_score = np.linalg.norm(self.positions - center[None, :], axis=1) / diagonal
        aperture_score = np.abs(self.vertex_cone0[:, 3] - np.deg2rad(65.0))
        return int(np.argmin(aperture_score + distance_score * 0.15))


@dataclass
class ViewerTextures:
    width: int
    height: int
    normal_roughness: spy.Texture
    albedo_metallic: spy.Texture
    emissive_occlusion: spy.Texture
    baked_value: spy.Texture
    visibility_cone: spy.Texture
    cone_params: spy.Texture
    depth: spy.Texture
    ids: spy.Texture
    raster_depth: spy.Texture
    output: spy.Texture


class BakerViewerPasses:
    def __init__(
        self,
        device: spy.Device,
        shader_session: spy.SlangSession | None = None,
        use_ray_query: bool = True,
    ) -> None:
        self.device = device
        self.shader_session = shader_session or device.slang_session
        self.use_ray_query = bool(use_ray_query)
        self.clear_program = self.shader_session.load_program(str(SHADER_PATH), ["clear_gbuffer_main"])
        self.clear_pipeline = device.create_compute_pipeline(self.clear_program)
        if self.use_ray_query:
            self.gbuffer_program = self.shader_session.load_program(str(SHADER_PATH), ["gbuffer_main"])
            self.gbuffer_pipeline = device.create_compute_pipeline(self.gbuffer_program)
            self.gbuffer_raster_pipeline = None
        else:
            self.gbuffer_program = self.shader_session.load_program(
                str(SHADER_PATH),
                ["gbuffer_raster_vertex", "gbuffer_raster_fragment"],
            )
            self.gbuffer_input_layout = device.create_input_layout(
                [
                    {"semantic_name": "POSITION", "format": spy.Format.rgb32_float, "offset": 0},
                    {"semantic_name": "NORMAL", "format": spy.Format.rgb32_float, "offset": 12},
                    {"semantic_name": "TEXCOORD", "semantic_index": 0, "format": spy.Format.rg32_float, "offset": 24},
                    {"semantic_name": "TEXCOORD", "semantic_index": 1, "format": spy.Format.rgba32_float, "offset": 32},
                    {"semantic_name": "TEXCOORD", "semantic_index": 2, "format": spy.Format.rgba32_float, "offset": 48},
                    {"semantic_name": "TEXCOORD", "semantic_index": 3, "format": spy.Format.r32_float, "offset": 64},
                    {"semantic_name": "TEXCOORD", "semantic_index": 4, "format": spy.Format.rgb32_float, "offset": 68},
                    {"semantic_name": "TEXCOORD", "semantic_index": 5, "format": spy.Format.r32_uint, "offset": 80},
                ],
                [{"stride": 84}],
            )
            target_formats = (
                spy.Format.rgba16_float,
                spy.Format.rgba16_float,
                spy.Format.rgba16_float,
                spy.Format.rgba16_float,
                spy.Format.rgba16_float,
                spy.Format.rg16_float,
                spy.Format.r32_float,
                spy.Format.rgba32_uint,
            )
            self.gbuffer_raster_pipeline = device.create_render_pipeline(
                self.gbuffer_program,
                self.gbuffer_input_layout,
                primitive_topology=spy.PrimitiveTopology.triangle_list,
                targets=[{"format": format} for format in target_formats],
                depth_stencil={
                    "format": spy.Format.d32_float,
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_func": spy.ComparisonFunc.less,
                },
                rasterizer={
                    "fill_mode": spy.FillMode.solid,
                    "cull_mode": spy.CullMode.none,
                    "front_face": spy.FrontFaceMode.counter_clockwise,
                    "depth_clip_enable": True,
                },
                label="baker_viewer_raster_gbuffer",
            )
            self.gbuffer_pipeline = None
        self.composite_program = self.shader_session.load_program(str(SHADER_PATH), ["composite_main"])
        self.composite_pipeline = device.create_compute_pipeline(self.composite_program)
        self.pick_program = self.shader_session.load_program(str(SHADER_PATH), ["pick_main"])
        self.pick_pipeline = device.create_compute_pipeline(self.pick_program)
        self.pick_buffer = device.create_buffer(
            size=16,
            usage=spy.BufferUsage.unordered_access,
            label="baker_viewer_pick_result",
        )

    def create_textures(self, width: int, height: int) -> ViewerTextures:
        usage = (
            spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target
        )

        def texture(format: spy.Format, label: str) -> spy.Texture:
            return self.device.create_texture(
                format=format,
                width=width,
                height=height,
                usage=usage,
                label=label,
            )

        return ViewerTextures(
            width=width,
            height=height,
            normal_roughness=texture(spy.Format.rgba16_float, "baker_gbuffer_normal_roughness"),
            albedo_metallic=texture(spy.Format.rgba16_float, "baker_gbuffer_albedo_metallic"),
            emissive_occlusion=texture(spy.Format.rgba16_float, "baker_gbuffer_emissive_occlusion"),
            baked_value=texture(spy.Format.rgba16_float, "baker_gbuffer_baked_value"),
            visibility_cone=texture(spy.Format.rgba16_float, "baker_gbuffer_visibility_cone"),
            cone_params=texture(spy.Format.rg16_float, "baker_gbuffer_cone_params"),
            depth=texture(spy.Format.r32_float, "baker_gbuffer_depth"),
            ids=texture(spy.Format.rgba32_uint, "baker_gbuffer_ids"),
            raster_depth=self.device.create_texture(
                format=spy.Format.d32_float,
                width=width,
                height=height,
                usage=spy.TextureUsage.depth_stencil,
                label="baker_gbuffer_raster_depth",
            ),
            output=texture(spy.Format.rgba32_float, "baker_viewer_output"),
        )

    @staticmethod
    def _bind_gbuffer_textures(cursor: spy.ShaderCursor, textures: ViewerTextures) -> None:
        cursor.g_normal_roughness = textures.normal_roughness
        cursor.g_albedo_metallic = textures.albedo_metallic
        cursor.g_emissive_occlusion = textures.emissive_occlusion
        cursor.g_baked_value = textures.baked_value
        cursor.g_visibility_cone = textures.visibility_cone
        cursor.g_cone_params = textures.cone_params
        cursor.g_depth = textures.depth
        cursor.g_ids = textures.ids

    def encode_gbuffer(
        self,
        command_encoder: spy.CommandEncoder,
        scene: BakerRTScene,
        camera: Camera,
        textures: ViewerTextures,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.clear_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            self._bind_gbuffer_textures(cursor, textures)
            pass_encoder.dispatch(thread_count=[textures.width, textures.height, 1])

        if self.use_ray_query:
            assert self.gbuffer_pipeline is not None
            with command_encoder.begin_compute_pass() as pass_encoder:
                shader_object = pass_encoder.bind_pipeline(self.gbuffer_pipeline)
                cursor = spy.ShaderCursor(shader_object)
                scene.bind_gbuffer(cursor)
                camera.bind(cursor.g_camera)
                self._bind_gbuffer_textures(cursor, textures)
                pass_encoder.dispatch(thread_count=[textures.width, textures.height, 1])
            return

        assert self.gbuffer_raster_pipeline is not None
        color_textures = (
            textures.normal_roughness,
            textures.albedo_metallic,
            textures.emissive_occlusion,
            textures.baked_value,
            textures.visibility_cone,
            textures.cone_params,
            textures.depth,
            textures.ids,
        )
        render_pass = {
            "color_attachments": [
                {
                    "view": texture.create_view(),
                    "load_op": spy.LoadOp.load,
                    "store_op": spy.StoreOp.store,
                }
                for texture in color_textures
            ],
            "depth_stencil_attachment": {
                "view": textures.raster_depth.create_view(),
                "depth_load_op": spy.LoadOp.clear,
                "depth_store_op": spy.StoreOp.dont_care,
                "depth_clear_value": 1.0,
            },
        }
        with command_encoder.begin_render_pass(render_pass) as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.gbuffer_raster_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            scene.bind_gbuffer(cursor)
            camera.bind(cursor.g_camera)
            pass_encoder.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(textures.width, textures.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(textures.width, textures.height)],
                    "vertex_buffers": [scene.raster_vertex_buffer],
                }
            )
            start_vertex = 0
            for mesh in scene.model.meshes:
                vertex_count = int(mesh.indices.size)
                scene.bind_raster_material(cursor, mesh.material_index)
                pass_encoder.draw(
                    {
                        "vertex_count": vertex_count,
                        "instance_count": 1,
                        "start_vertex_location": start_vertex,
                    }
                )
                start_vertex += vertex_count

    def encode_pick(
        self,
        command_encoder: spy.CommandEncoder,
        textures: ViewerTextures,
        scene: BakerRTScene,
        camera: Camera,
        pixel: tuple[int, int],
        radius: float,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pick_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_pick_ids = textures.ids
            cursor.g_pick_depth = textures.depth
            cursor.g_pick_positions = scene.position_buffer
            cursor.g_pick_vertex_mesh_ids = scene.vertex_mesh_buffer
            cursor.g_pick_result = self.pick_buffer
            cursor.g_pick_pixel = spy.uint2(pixel[0], pixel[1])
            cursor.g_pick_vertex_count = int(scene.positions.shape[0])
            cursor.g_pick_radius = max(float(radius), 0.0)
            cursor.g_pick_depth_tolerance = max(
                float(np.linalg.norm(scene.bounds_max - scene.bounds_min)) * 0.002,
                1e-5,
            )
            camera.bind(cursor.g_pick_camera)
            pass_encoder.dispatch(thread_count=[1, 1, 1])

    def encode_composite(
        self,
        command_encoder: spy.CommandEncoder,
        textures: ViewerTextures,
        camera: Camera,
        scene: BakerRTScene,
        environment: EnvironmentIBL,
        *,
        view_mode: int,
        selected_vertex: int,
        cone_length: float,
        show_cone: bool,
        show_diagram: bool,
        exposure: float,
        environment_rotation: float,
        show_environment: bool,
        apply_visibility: bool,
    ) -> None:
        selected = scene.selected_data(selected_vertex)
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.composite_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_comp_normal_roughness = textures.normal_roughness
            cursor.g_comp_albedo_metallic = textures.albedo_metallic
            cursor.g_comp_emissive_occlusion = textures.emissive_occlusion
            cursor.g_comp_baked_value = textures.baked_value
            cursor.g_comp_visibility_cone = textures.visibility_cone
            cursor.g_comp_cone_params = textures.cone_params
            cursor.g_comp_depth = textures.depth
            cursor.g_comp_ids = textures.ids
            cursor.g_comp_sky = environment.sky
            cursor.g_comp_specular = environment.specular
            cursor.g_comp_dfg = environment.dfg
            cursor.g_comp_specular_occlusion = environment.specular_occlusion
            cursor.g_comp_sh = environment.sh
            cursor.g_comp_environment_sampler = environment.cube_sampler
            camera.bind(cursor.g_comp_camera)
            cursor.g_output = textures.output
            cursor.g_view_mode = int(view_mode)
            cursor.g_depth_scale = max(float(np.linalg.norm(scene.bounds_max - scene.bounds_min)), 1e-5)
            cursor.g_view_proj = camera.view_proj_matrix_no_jitter
            cursor.g_selected_valid = 1 if selected is not None and scene.has_cones else 0
            cursor.g_selected_position = spy.float3(selected["position"] if selected is not None else [0, 0, 0])
            cursor.g_selected_axis = spy.float3(selected["axis"] if selected is not None else [0, 0, 1])
            cursor.g_selected_aperture = float(selected["aperture"] if selected is not None else 0.0)
            cursor.g_selected_scale = float(selected["scale"] if selected is not None else 0.0)
            cursor.g_selected_length = max(float(cone_length), 1e-6)
            cursor.g_show_cone = 1 if show_cone else 0
            cursor.g_show_diagram = 1 if show_diagram else 0
            cursor.g_environment_exposure = float(exposure)
            cursor.g_environment_rotation = math.radians(float(environment_rotation))
            cursor.g_specular_mip_count = float(environment.specular.mip_count)
            cursor.g_show_environment = 1 if show_environment else 0
            cursor.g_apply_visibility = 1 if apply_visibility else 0
            cursor.g_has_visibility = 1 if scene.has_cones else 0
            pass_encoder.dispatch(thread_count=[textures.width, textures.height, 1])


class OrbitCameraController:
    def __init__(self, camera: Camera, bounds_min, bounds_max) -> None:
        self.camera = camera
        self.bounds_min = np.asarray(bounds_min, dtype=np.float32)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float32)
        self.radius = max(float(np.linalg.norm(self.bounds_max - self.bounds_min) * 0.5), 1e-3)
        self.bounds_center = (self.bounds_min + self.bounds_max) * 0.5
        self.center = self.bounds_center.copy()
        self.yaw = 0.08
        self.pitch = 0.02
        self.distance = self.radius * 3.0
        self.rotating = False
        self.panning = False
        self.last_pos = np.zeros(2, dtype=np.float32)
        self.press_pos = np.zeros(2, dtype=np.float32)
        self.dragged = False
        self.update_camera()

    def reset(self) -> None:
        self.center = self.bounds_center.copy()
        self.yaw = 0.08
        self.pitch = 0.02
        self.distance = self.radius * 3.0
        self.update_camera()

    def update_camera(self) -> None:
        cp = math.cos(self.pitch)
        direction = np.array(
            [math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp],
            dtype=np.float32,
        )
        position = self.center + direction * self.distance
        self.camera.position = spy.float3(position)
        self.camera.target = spy.float3(self.center)
        self.camera.up = spy.float3(0.0, 1.0, 0.0)
        distance_to_bounds_center = float(np.linalg.norm(position - self.bounds_center))
        bounds_padding = self.radius * 1.05
        self.camera.near_clip_plane = max(
            distance_to_bounds_center - bounds_padding,
            self.radius * 1e-3,
            1e-5,
        )
        self.camera.far_clip_plane = max(
            distance_to_bounds_center + bounds_padding,
            self.camera.near_clip_plane + self.radius * 0.1,
        )
        self.camera.focal_distance = 1.0
        self.camera.fov = 45.0
        self.camera.recompute()

    @staticmethod
    def _event_position(event: spy.MouseEvent) -> npt.NDArray[np.float32]:
        return np.array([float(event.pos.x), float(event.pos.y)], dtype=np.float32)

    def on_mouse_event(self, event: spy.MouseEvent) -> tuple[int, int] | None:
        position = self._event_position(event)
        if event.is_button_down():
            self.last_pos = position
            self.press_pos = position
            self.dragged = False
            if event.button == spy.MouseButton.left:
                self.rotating = True
            elif event.button in (spy.MouseButton.right, spy.MouseButton.middle):
                self.panning = True
            return None
        if event.is_button_up():
            was_click = event.button == spy.MouseButton.left and self.rotating and not self.dragged
            if event.button == spy.MouseButton.left:
                self.rotating = False
            elif event.button in (spy.MouseButton.right, spy.MouseButton.middle):
                self.panning = False
            return (int(position[0]), int(position[1])) if was_click else None
        if event.is_move():
            delta = position - self.last_pos
            self.last_pos = position
            if np.linalg.norm(position - self.press_pos) > 3.0:
                self.dragged = True
            if self.rotating:
                self.yaw += float(delta[0]) * 0.006
                self.pitch = float(np.clip(self.pitch + float(delta[1]) * 0.006, -1.45, 1.45))
                self.update_camera()
            elif self.panning:
                forward = np.asarray(self.camera.target - self.camera.position, dtype=np.float32)
                forward /= max(float(np.linalg.norm(forward)), 1e-8)
                right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                right /= max(float(np.linalg.norm(right)), 1e-8)
                up = np.cross(right, forward)
                scale = self.distance * 0.0015
                self.center += (-right * float(delta[0]) + up * float(delta[1])) * scale
                self.update_camera()
        if event.is_scroll():
            scroll_y = float(event.scroll.y)
            self.distance *= math.exp(-scroll_y * 0.12)
            self.distance = float(np.clip(self.distance, self.radius * 0.05, self.radius * 30.0))
            self.update_camera()
        return None


class VertexBakerSlangViewer:
    def __init__(
        self,
        model: Model,
        vertex_values: list[npt.NDArray[np.float32]] | None = None,
        vertex_cones: list[npt.NDArray[np.float32]] | None = None,
        *,
        width: int = 1280,
        height: int = 720,
        cone_length: float = 0.5,
        screenshot_path: str | Path = "vertex_baker_viewer.png",
        max_frames: int = 0,
        capture_on_exit: bool = False,
        environment_path: str | Path = DEFAULT_ENVIRONMENT,
        exposure: float = 0.0,
        environment_rotation: float = 0.0,
        apply_visibility: bool = True,
    ) -> None:
        resolved_environment_path = Path(environment_path).resolve()
        resolved_screenshot_path = Path(screenshot_path).resolve()
        self.device = spy.Device(
            enable_debug_layers=False,
            enable_print=True,
            compiler_options={
                "include_paths": [ROOT, VERTEX_BAKER_DIR],
                "defines": {"USE_RAYTRACING_PIPELINE": "0", "HEADLESS_MODE": "0"},
            },
        )
        self.use_ray_query = self.device.has_feature(spy.Feature.ray_query)
        self.shader_session = self.device.create_slang_session(
            compiler_options={
                "include_paths": [ROOT, VERTEX_BAKER_DIR],
                "defines": {
                    "USE_RAYTRACING_PIPELINE": "0",
                    "HEADLESS_MODE": "0",
                    "USE_RAY_QUERY": "1" if self.use_ray_query else "0",
                },
            }
        )
        self.window = spy.Window(width=max(1, width), height=max(1, height), title="Vertex Baker", resizable=True)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height, vsync=False)
        self.ui = spy.ui.Context(self.device)
        self.environment = EnvironmentIBL(self.device, resolved_environment_path, self.shader_session)
        self.scene = BakerRTScene(
            self.device,
            model,
            vertex_values,
            vertex_cones,
            shader_session=self.shader_session,
            use_ray_query=self.use_ray_query,
        )
        self.passes = BakerViewerPasses(
            self.device,
            self.shader_session,
            use_ray_query=self.use_ray_query,
        )
        print(
            "Vertex baker GBuffer: "
            + ("inline ray query" if self.use_ray_query else "raster fallback")
        )
        self.camera = Camera()
        self.orbit = OrbitCameraController(self.camera, self.scene.bounds_min, self.scene.bounds_max)
        self.textures: ViewerTextures | None = None
        self.pending_pick: tuple[int, int] | None = None
        self.selected_vertex = self.scene.initial_vertex()
        self.last_pick_distance: float | None = None
        self.cone_length = max(float(cone_length), 1e-6)
        self.screenshot_path = resolved_screenshot_path
        self.max_frames = max(0, int(max_frames))
        self.capture_on_exit = bool(capture_on_exit)

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        ui_window = spy.ui.Window(self.ui.screen, "Vertex Baker", spy.float2(10, 10), spy.float2(330, 445))
        self.view_mode = spy.ui.ComboBox(ui_window, "View", items=list(VIEW_MODE_NAMES), value=0)
        self.exposure_slider = spy.ui.SliderFloat(ui_window, "Exposure EV", min=-8.0, max=8.0, value=float(exposure))
        self.environment_rotation_slider = spy.ui.SliderFloat(
            ui_window,
            "Environment rotation",
            min=-180.0,
            max=180.0,
            value=float(environment_rotation),
        )
        self.show_environment = spy.ui.CheckBox(ui_window, "Show environment", value=True)
        self.apply_visibility = spy.ui.CheckBox(
            ui_window,
            "Apply visibility",
            value=bool(apply_visibility and vertex_cones is not None),
        )
        self.show_cone = spy.ui.CheckBox(ui_window, "Selected cone", value=vertex_cones is not None)
        self.show_diagram = spy.ui.CheckBox(ui_window, "Cone cross-section", value=vertex_cones is not None)
        self.cone_length_slider = spy.ui.SliderFloat(
            ui_window,
            "Cone length",
            min=max(self.cone_length * 0.05, 1e-4),
            max=max(self.cone_length * 4.0, self.cone_length + 1e-4),
            value=self.cone_length,
        )
        self.pick_radius_slider = spy.ui.SliderFloat(
            ui_window,
            "Pick radius",
            min=4.0,
            max=64.0,
            value=20.0,
        )
        spy.ui.Button(ui_window, "Reset camera", callback=self.orbit.reset)
        self.selection_text = spy.ui.Text(ui_window, "")
        self._update_selection_text()

    def _ensure_textures(self, width: int, height: int) -> None:
        if self.textures is None or self.textures.width != width or self.textures.height != height:
            self.device.wait()
            self.textures = self.passes.create_textures(width, height)

    def _update_selection_text(self) -> None:
        selected = self.scene.selected_data(self.selected_vertex)
        if selected is None:
            self.selection_text.text = "Selection: none"
            return
        aperture_degrees = math.degrees(float(selected["aperture"]))
        pick_distance = (
            f"\nPick distance: {self.last_pick_distance:.2f} px"
            if self.last_pick_distance is not None
            else ""
        )
        self.selection_text.text = (
            f"{selected['mesh_name']} | vertex {selected['local_id']}\n"
            f"Aperture: {aperture_degrees:.4f} deg\n"
            f"Scale: {float(selected['scale']):.6f}\n"
            f"Global vertex: {selected['global_id']}"
            f"{pick_distance}"
        )

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
                return
            if event.key == spy.KeyCode.f2 and self.textures is not None:
                bitmap = self.textures.output.to_bitmap()
                bitmap.convert(
                    spy.Bitmap.PixelFormat.rgb,
                    spy.Bitmap.ComponentType.uint8,
                    srgb_gamma=False,
                ).write_async(str(self.screenshot_path))
        self.ui.handle_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if self.ui.handle_mouse_event(event):
            return
        clicked = self.orbit.on_mouse_event(event)
        if clicked is not None:
            self.pending_pick = clicked

    def on_resize(self, width: int, height: int) -> None:
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=False)
        else:
            self.surface.unconfigure()

    def run(self) -> None:
        frame = 0
        while not self.window.should_close():
            self.window.process_events()
            if not self.surface.config:
                continue
            surface_texture = self.surface.acquire_next_image()
            if surface_texture is None:
                continue
            self._ensure_textures(surface_texture.width, surface_texture.height)
            assert self.textures is not None
            self.camera.width = self.textures.width
            self.camera.height = self.textures.height
            self.camera.recompute()

            command_encoder = self.device.create_command_encoder()
            self.passes.encode_gbuffer(command_encoder, self.scene, self.camera, self.textures)
            pick_requested = self.pending_pick is not None
            if self.pending_pick is not None:
                window_width = max(int(self.window.width), 1)
                window_height = max(int(self.window.height), 1)
                pick_x = int(round(self.pending_pick[0] * self.textures.width / window_width))
                pick_y = int(round(self.pending_pick[1] * self.textures.height / window_height))
                pick_x = int(np.clip(pick_x, 0, self.textures.width - 1))
                pick_y = int(np.clip(pick_y, 0, self.textures.height - 1))
                self.passes.encode_pick(
                    command_encoder,
                    self.textures,
                    self.scene,
                    self.camera,
                    (pick_x, pick_y),
                    float(self.pick_radius_slider.value),
                )
                self.pending_pick = None
            self.passes.encode_composite(
                command_encoder,
                self.textures,
                self.camera,
                self.scene,
                self.environment,
                view_mode=int(self.view_mode.value),
                selected_vertex=self.selected_vertex,
                cone_length=float(self.cone_length_slider.value),
                show_cone=bool(self.show_cone.value),
                show_diagram=bool(self.show_diagram.value),
                exposure=float(self.exposure_slider.value),
                environment_rotation=float(self.environment_rotation_slider.value),
                show_environment=bool(self.show_environment.value),
                apply_visibility=bool(self.apply_visibility.value),
            )
            command_encoder.blit(surface_texture, self.textures.output)
            self.ui.begin_frame(self.window.width, self.window.height)
            self.ui.end_frame(surface_texture, command_encoder)
            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture
            self.surface.present()

            if pick_requested:
                self.device.wait()
                picked = np.asarray(self.passes.pick_buffer.to_numpy()).view(np.uint32).reshape(-1)
                if picked.size >= 4 and picked[3] != 0:
                    self.selected_vertex = int(picked[0])
                    self.last_pick_distance = float(picked[2:3].view(np.float32)[0])
                else:
                    self.selected_vertex = -1
                    self.last_pick_distance = None
                self._update_selection_text()

            self.device.flush_print()
            frame += 1
            if self.max_frames > 0 and frame >= self.max_frames:
                break
        self.device.wait()
        if self.capture_on_exit and self.textures is not None:
            self.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.textures.output.to_bitmap().convert(
                spy.Bitmap.PixelFormat.rgb,
                spy.Bitmap.ComponentType.uint8,
                srgb_gamma=False,
            ).write(str(self.screenshot_path))


def load_visibility_view_data(
    model: Model,
    path: str | Path,
) -> tuple[list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
    path = Path(path)
    with np.load(path) as data:
        cones = []
        encoded = []
        for mesh_index, mesh in enumerate(model.meshes):
            cone_key = f"mesh_{mesh_index}_raw_visibility_cone"
            encoded_key = f"mesh_{mesh_index}_texcoord2"
            if cone_key not in data or encoded_key not in data:
                raise ValueError(f"{path} does not contain {cone_key} and {encoded_key}")
            mesh_cones = np.ascontiguousarray(data[cone_key], dtype=np.float32)
            mesh_encoded = np.ascontiguousarray(data[encoded_key], dtype=np.float32)
            if mesh_cones.shape != (mesh.positions.shape[0], 5):
                raise ValueError(
                    f"{cone_key} has shape {mesh_cones.shape}, expected ({mesh.positions.shape[0]}, 5)"
                )
            cones.append(mesh_cones)
            encoded.append(mesh_encoded)
    values = vertex_visibility_preview_values(
        SimpleNamespace(vertex_cones=cones, encoded_texcoord2=encoded),
        model,
    )
    return values, cones


def run_slang_viewer(
    model: Model,
    vertex_values: list[npt.NDArray[np.float32]] | None = None,
    vertex_cones: list[npt.NDArray[np.float32]] | None = None,
    **kwargs,
) -> None:
    VertexBakerSlangViewer(model, vertex_values, vertex_cones, **kwargs).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native SlangPy vertex baker viewer.")
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, default=None, help="Optional visibility .npz to inspect.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cone-length", type=float, default=0.5)
    parser.add_argument("--screenshot", type=Path, default=Path("vertex_baker_viewer.png"))
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames; zero runs until closed.")
    parser.add_argument("--capture-on-exit", action="store_true")
    parser.add_argument("--envmap", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--env-exposure", type=float, default=0.0)
    parser.add_argument("--env-rotation-degrees", type=float, default=0.0)
    parser.add_argument("--no-apply-visibility", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = load_gltf_model(args.asset)
    values = None
    cones = None
    if args.visibility is not None:
        values, cones = load_visibility_view_data(model, args.visibility)
    run_slang_viewer(
        model,
        values,
        cones,
        width=args.width,
        height=args.height,
        cone_length=args.cone_length,
        screenshot_path=args.screenshot,
        max_frames=args.max_frames,
        capture_on_exit=args.capture_on_exit,
        environment_path=args.envmap,
        exposure=args.env_exposure,
        environment_rotation=args.env_rotation_degrees,
        apply_visibility=not args.no_apply_visibility,
    )


if __name__ == "__main__":
    main()
