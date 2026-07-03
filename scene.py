from event_dispatcher.sync_dispatcher import SyncEventDispatcher


from slangpy import Device, Sampler


from slangpy.math import float3


import slangpy as spy
import numpy as np
import struct
import math
from datetime import datetime
from camera import Camera
from dataclasses import dataclass
from scene_node import SceneNode
from material import Material, ALPHA_MODE_OPAQUE, ALPHA_MODE_MASK, ALPHA_MODE_BLEND
from mesh import Mesh
from transform import Transform
from typing import List, Optional, Tuple
from event_dispatcher import SyncEventDispatcher
from atmosphere import AtmosphereTransmittanceLUT, AtmosphereMultiScatteringLUT, AtmosphereSkyViewLUT
from sun_position import SunPosition, SunPositionData
from texture_manager import TextureManager, TextureType, TextureRecord
from static_shadow_depth_map import StaticShadowDepthMap
from sparse_shadow_tree import SparseShadowTreeEncoder, SparseShadowTreeStats
from sst_decompress import SSTDecompressor


class Scene:
    SHADOW_MODE_REALTIME = 0
    SHADOW_MODE_DEPTH_TEXTURE = 1
    SHADOW_MODE_SST = 2
    SHADOW_MODE_PACKED_SST = 3
    SHADOW_MODE_COMPACT_SST = 4
    SHADOW_MODE_COMPACT_SST_PCF3 = 5
    SHADOW_MODE_DECOMPRESSED_SST = 6
    SST_FIT_DUAL_BIAS = 0
    SST_FIT_DUAL_HALF_VISIBLE = 1
    SST_FIT_DUAL_VISIBLE = 2
    SST_FIT_DUAL_RELAXED_VISIBLE = 3
    SST_FIT_DUAL_LOOSE_VISIBLE = 4
    SST_PRESET_QUALITY = 0
    SST_PRESET_HIGH_COMPRESSION = 1
    SST_PRESET_MANUAL = 2

    # Instance flags (must match InstanceFlags in common.slang)
    INSTANCE_FLAG_NONE = 0
    INSTANCE_FLAG_ODD_NEGATIVE_TRANSFORM = 1 << 0
    INSTANCE_FLAG_TRANSMISSIVE = 1 << 1
    
    # Texture presence flags (must match TextureFlags in common.slang)
    TEXTURE_FLAG_BASE_COLOR = 1 << 0
    TEXTURE_FLAG_NORMAL = 1 << 1
    TEXTURE_FLAG_ROUGHNESS = 1 << 2
    TEXTURE_FLAG_METALLIC = 1 << 3
    TEXTURE_FLAG_EMISSIVE = 1 << 4
    TEXTURE_FLAG_SPECULAR_COLOR = 1 << 5
    
    @dataclass
    class MaterialData:
        """Internal material data for BufferCursor filling"""
        base_color: spy.float3
        flags: int
        emissive: spy.float3
        shading_model: int
        roughness: float
        metallic: float
        texture_flags: int
        alpha_mode: int
        alpha_cutoff: float
        specular_color: spy.float3
        # TextureRecords for bindless handles (None means use default)
        base_color_tex: Optional[TextureRecord]
        normal_tex: Optional[TextureRecord]
        roughness_tex: Optional[TextureRecord]
        metallic_tex: Optional[TextureRecord]
        emissive_tex: Optional[TextureRecord]
        specular_color_tex: Optional[TextureRecord]

    @dataclass
    class MeshDesc:
        vertex_count: int
        index_count: int
        vertex_offset: int
        index_offset: int

        def pack(self):
            return struct.pack(
                "IIII",
                self.vertex_count,
                self.index_count,
                self.vertex_offset,
                self.index_offset,
            )

    @dataclass
    class InstanceDesc:
        mesh_id: int
        material_id: int
        transform_id: int
        instance_flags: int

        def pack(self):
            return struct.pack("IIII", self.mesh_id, self.material_id, self.transform_id, self.instance_flags)

    def __init__(self, device: spy.Device, scene_node: SceneNode, event_distpacher: SyncEventDispatcher):
        super().__init__()
        self.device: Device = device
        self.event_distpacher: SyncEventDispatcher = event_distpacher

        self.linear_sampler: Sampler = device.create_sampler()
        self.transmittance_sampler: Sampler = device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        self.static_shadow_sampler: Sampler = device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        self.camera: Camera = scene_node.camera
        self.asset_path: str = scene_node.asset_path
        
        # Sky atmosphere parameters
        # Sun direction: (0.0, 0.5, 0.866) = 30 degrees elevation, toward +Z (illuminating surfaces facing -Z)
        self._sun_direction: float3 = spy.float3(0.0, 0.5, 0.866)  # Default: 30 degrees elevation, +Z
        self._sun_direction_dirty = True  # Flag to track if sky LUT needs regeneration
        
        # Directional light parameters
        # Default intensity: PI (≈3.14159), matching UE's default
        # Default color: white (1, 1, 1)
        # Default half angle: 0.26785 degrees (half of UE's default 0.5357° angular diameter)
        self._directional_light_intensity: float = math.pi
        self._directional_light_color: spy.float3 = spy.float3(1.0, 1.0, 1.0)
        self._directional_light_cos_half_angle: float = math.cos(math.radians(0.5357 / 2.0))
        
        # UI elements
        self._hours_slider: spy.ui.SliderFloat | None = None
        self._intensity_slider: spy.ui.SliderFloat | None = None
        self._static_shadow_status_text: spy.ui.Text | None = None
        
        # Location for sun position calculation (default: Chengdu, async update)
        self._latitude: float = SunPosition.DEFAULT_LATITUDE
        self._longitude: float = SunPosition.DEFAULT_LONGITUDE
        self._timezone = SunPosition.get_local_timezone()
        self._last_hours_value: float | None = None  # Track slider changes
        
        # Frame index for stochastic alpha
        self._frame_index: int = 0

        # Static shadow depth map state. The renderer falls back to realtime
        # ray-query shadows until a valid bake is produced.
        self.static_shadow_resolution: int = 2048
        self.static_shadow_depth_bias: float = 0.0015
        self.static_shadow_enabled: bool = False
        self.static_shadow_mode: int = Scene.SHADOW_MODE_REALTIME
        self.static_shadow_depth_texture: spy.Texture = device.create_texture(
            format=spy.Format.r32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            label="static_shadow_default_depth",
            data=np.ones((1, 1), dtype=np.float32),
        )
        self.static_shadow_second_depth_texture: spy.Texture = device.create_texture(
            format=spy.Format.r32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            label="static_shadow_default_second_depth",
            data=np.ones((1, 1), dtype=np.float32),
        )
        self.static_shadow_world_to_light: spy.float4x4 = spy.float4x4.identity()
        self.static_shadow_baked_sun_direction: tuple[float, float, float] | None = None
        self.static_shadow_baker = StaticShadowDepthMap(device)
        self.sst_decompressor = SSTDecompressor(device)
        self.sst_decompressed_depth_texture: spy.Texture = device.create_texture(
            format=spy.Format.r32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            label="sst_default_decompressed_depth",
            data=np.ones((1, 1), dtype=np.float32),
        )
        self.sst_fit_profile: int = Scene.SST_FIT_DUAL_VISIBLE
        self.sst_preset: int = Scene.SST_PRESET_QUALITY
        self.sst_tile_profile: int = 1  # 0=64, 1=128, 2=256
        self.sst_min_leaf_size: int = 2
        self.sst_quantization_search_radius: int = 0
        self.sst_dual_depth_slack_scale: float = 1.0
        self.sst_encoder = self._create_sst_encoder_for_profile(self.sst_fit_profile)
        self.sst_enabled: bool = False
        self.sst_tile_size: int = self.sst_encoder.tile_size
        self.sst_tile_grid: tuple[int, int] = (1, 1)
        self.sst_node_count: int = 1
        self.sst_max_traversal_steps: int = self.sst_encoder.max_traversal_steps
        self.sst_compact_word_count: int = 1
        self.sst_branch_10bit_start_level: int = self.sst_encoder.branch_10bit_start_level
        self.sst_compact_valid: bool = True
        self.sst_max_error: float = 0.0
        self.sst_mean_error: float = 0.0
        self.sst_compression_ratio: float = 0.0
        self.sst_stats: SparseShadowTreeStats | None = None
        # Bistro benchmark winner: Dual Visible, 128x128 tiles, 2x2 min leaves,
        # compact bit-packed traversal with 3x3 PCF. Keep the lower-level knobs
        # for headless experiments, but the interactive UI uses this path.
        self.static_shadow_auto_encode_sst: bool = True
        self.sst_node_buffer = self._create_sst_node_buffer(
            SparseShadowTreeEncoder.NODE_STRUCT.pack(1, 0, 0, 0, 0.0, 0.0, 1.0, 0.0),
            "sst_default_node_buffer",
        )
        self.sst_tile_roots_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_default_tile_roots",
            data=np.array([0], dtype=np.uint32),
        )
        self.sst_packed_node_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_default_packed_node_buffer",
            data=np.array([[1, 0]], dtype=np.uint32),
        )
        self.sst_compact_words_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_default_compact_words_buffer",
            data=np.array([1], dtype=np.uint32),
        )
        self.sst_compact_roots_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_default_compact_roots_buffer",
            data=np.array([0], dtype=np.uint32),
        )
        
        # Start async location fetch
        # SunPosition.get_current_location_async(self._on_location_received)
        
        # Initialize atmosphere LUT generators
        self.transmittance_lut_gen: AtmosphereTransmittanceLUT = AtmosphereTransmittanceLUT(device)
        self.multiscatt_lut_gen: AtmosphereMultiScatteringLUT = AtmosphereMultiScatteringLUT(device)
        self.sky_view_lut_gen: AtmosphereSkyViewLUT = AtmosphereSkyViewLUT(device)
        
        # Generate static LUTs (transmittance and multi-scattering don't depend on sun direction)
        self._generate_static_atmosphere_luts()
        
        # Generate initial sky view LUT
        self._generate_sky_view_lut()

        # Initialize TextureManager for PBR textures
        self.texture_manager = TextureManager(device)
        
        # Prepare material data with texture records
        self.material_data_list: list[Scene.MaterialData] = []
        print(f"[Scene] Starting material processing, scene_node has {len(scene_node.materials)} materials")
        for i, m in enumerate(scene_node.materials):
            # Load textures and get TextureRecords
            base_color_tex = None
            normal_tex = None
            roughness_tex = None
            metallic_tex = None
            emissive_tex = None
            specular_color_tex = None
            texture_flags = 0
            
            if m.base_color_texture:
                base_color_tex = self.texture_manager.load_texture(m.base_color_texture, TextureType.BASE_COLOR)
                texture_flags |= Scene.TEXTURE_FLAG_BASE_COLOR
            
            if m.normal_texture:
                normal_tex = self.texture_manager.load_texture(m.normal_texture, TextureType.NORMAL)
                texture_flags |= Scene.TEXTURE_FLAG_NORMAL
            
            if m.roughness_texture:
                roughness_tex = self.texture_manager.load_texture(m.roughness_texture, TextureType.ROUGHNESS)
                texture_flags |= Scene.TEXTURE_FLAG_ROUGHNESS
            
            if m.metallic_texture:
                metallic_tex = self.texture_manager.load_texture(m.metallic_texture, TextureType.METALLIC)
                texture_flags |= Scene.TEXTURE_FLAG_METALLIC
            
            if m.emissive_texture:
                emissive_tex = self.texture_manager.load_texture(m.emissive_texture, TextureType.EMISSIVE)
                texture_flags |= Scene.TEXTURE_FLAG_EMISSIVE
            
            if m.specular_color_texture:
                specular_color_tex = self.texture_manager.load_texture(m.specular_color_texture, TextureType.SPECULAR_COLOR)
                texture_flags |= Scene.TEXTURE_FLAG_SPECULAR_COLOR
            
            # Debug: print texture_flags for all materials
            print(f"[Scene] Material {i} (desc_idx={len(self.material_data_list)}): texture_flags={texture_flags}, "
                  f"base_color={m.base_color}, "
                  f"base_color_texture={m.base_color_texture}")
            
            self.material_data_list.append(Scene.MaterialData(
                base_color=m.base_color,
                flags=m.flags,
                emissive=m.emissive,
                shading_model=m.shading_model,
                roughness=m.roughness,
                metallic=m.metallic,
                texture_flags=texture_flags,
                alpha_mode=m.alpha_mode,
                alpha_cutoff=m.alpha_cutoff,
                specular_color=m.specular_color,
                base_color_tex=base_color_tex,
                normal_tex=normal_tex,
                roughness_tex=roughness_tex,
                metallic_tex=metallic_tex,
                emissive_tex=emissive_tex,
                specular_color_tex=specular_color_tex
            ))
        
        # Create material buffer using BufferCursor for proper Handle binding
        self.material_descs_buffer = self._create_material_buffer(device, self.material_data_list)

        # Prepare mesh descriptors
        vertex_count = 0
        index_count = 0
        self.mesh_descs = []
        for mesh in scene_node.meshes:
            self.mesh_descs.append(
                Scene.MeshDesc(
                    vertex_count=mesh.vertex_count,
                    index_count=mesh.index_count,
                    vertex_offset=vertex_count,
                    index_offset=index_count,
                )
            )
            vertex_count += mesh.vertex_count
            index_count += mesh.index_count

        # Prepare instance descriptors
        self.instance_descs = []

        for mesh_id, material_id, transform_id in scene_node.instances:
            # Calculate instance flags
            instance_flags = 0

            # Check if transform has odd negative scaling (determinant <= 0)
            # This causes triangle winding order to flip
            transform_matrix = scene_node.transforms[transform_id].matrix

            # Extract 3x3 rotation+scale part
            row0 = np.array([transform_matrix[0, 0], transform_matrix[0, 1], transform_matrix[0, 2]])
            row1 = np.array([transform_matrix[1, 0], transform_matrix[1, 1], transform_matrix[1, 2]])
            row2 = np.array([transform_matrix[2, 0], transform_matrix[2, 1], transform_matrix[2, 2]])

            # Compute determinant sign: dot(cross(row0, row1), row2)
            cross_product = np.cross(row0, row1)
            determinant_sign = np.dot(cross_product, row2)

            # If determinant <= 0, set OddNegativeTransform flag
            if determinant_sign <= 0:
                instance_flags |= Scene.INSTANCE_FLAG_ODD_NEGATIVE_TRANSFORM

            self.instance_descs.append(Scene.InstanceDesc(mesh_id, material_id, transform_id, instance_flags))

        # Debug: count material_id distribution
        material_id_counts = {}
        for inst in self.instance_descs:
            mid = inst.material_id
            material_id_counts[mid] = material_id_counts.get(mid, 0) + 1
        print(f"[Scene] Instance count: {len(self.instance_descs)}")
        print(f"[Scene] Unique material IDs used: {len(material_id_counts)}")
        print(f"[Scene] Material ID 0 instances: {material_id_counts.get(0, 0)}")
        # Show first 10 material IDs by count
        sorted_counts = sorted(material_id_counts.items(), key=lambda x: -x[1])[:10]
        print(f"[Scene] Top 10 material IDs by instance count: {sorted_counts}")

        # Create vertex and index buffers
        vertices = np.concatenate([mesh.vertices for mesh in scene_node.meshes], axis=0)
        indices = np.concatenate([mesh.indices for mesh in scene_node.meshes], axis=0)
        assert vertices.shape[0] == vertex_count
        assert indices.shape[0] == index_count // 3

        self.vertex_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_buffer",
            data=vertices,
        )

        self.index_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="index_buffer",
            data=indices,
        )

        mesh_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.mesh_descs), dtype=np.uint8
        ).flatten()
        self.mesh_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_descs_buffer",
            data=mesh_descs_data,
        )

        instance_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.instance_descs), dtype=np.uint8
        ).flatten()
        self.instance_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="instance_descs_buffer",
            data=instance_descs_data,
        )

        # Prepare transforms
        self.transforms = [t.matrix for t in scene_node.transforms]
        self.inverse_transpose_transforms = [
            spy.math.transpose(spy.math.inverse(t)) for t in self.transforms
        ]
        self.transform_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="transform_buffer",
            data=np.stack([t.to_numpy() for t in self.transforms]),
        )
        self.inverse_transpose_transforms_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="inverse_transpose_transforms_buffer",
            data=np.stack([t.to_numpy() for t in self.inverse_transpose_transforms]),
        )
        self.scene_bounds_min, self.scene_bounds_max = self._compute_scene_bounds(scene_node)

        # Build BLASes
        self.blases = [self.build_blas(mesh_desc) for mesh_desc in self.mesh_descs]

        # Build TLAS
        self.tlas = self.build_tlas()

    def build_blas(self, mesh_desc: MeshDesc):
        build_input = spy.AccelerationStructureBuildInputTriangles(
            {
                "vertex_buffers": [
                    {
                        "buffer": self.vertex_buffer,
                        "offset": mesh_desc.vertex_offset * 32,
                    }
                ],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": mesh_desc.vertex_count,
                "vertex_stride": 32,
                "index_buffer": {
                    "buffer": self.index_buffer,
                    "offset": mesh_desc.index_offset * 4,
                },
                "index_format": spy.IndexFormat.uint32,
                "index_count": mesh_desc.index_count,
                # Use 'none' instead of 'opaque' to allow alpha testing in any-hit shader
                "flags": spy.AccelerationStructureGeometryFlags.none,
            }
        )

        build_desc = spy.AccelerationStructureBuildDesc({"inputs": [build_input]})

        sizes = self.device.get_acceleration_structure_sizes(build_desc)

        blas_scratch_buffer = self.device.create_buffer(
            size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="blas_scratch_buffer",
        )

        blas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="blas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=build_desc, dst=blas, src=None, scratch_buffer=blas_scratch_buffer
        )
        self.device.submit_command_buffer(command_encoder.finish())

        return blas

    def build_tlas(self):
        instance_list = self.device.create_acceleration_structure_instance_list(
            size=len(self.instance_descs)
        )
        for instance_id, instance_desc in enumerate(self.instance_descs):
            instance_list.write(
                instance_id,
                {
                    "transform": spy.float3x4(self.transforms[instance_desc.transform_id]),
                    "instance_id": instance_id,
                    "instance_mask": 0xFF,
                    "instance_contribution_to_hit_group_index": 0,
                    "flags": spy.AccelerationStructureInstanceFlags.none,
                    "acceleration_structure": self.blases[instance_desc.mesh_id].handle,
                },
            )

        build_desc = spy.AccelerationStructureBuildDesc(
            {
                "inputs": [instance_list.build_input_instances()],
            }
        )

        sizes = self.device.get_acceleration_structure_sizes(build_desc)

        tlas_scratch_buffer = self.device.create_buffer(
            size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch_buffer",
        )

        tlas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="tlas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
        )
        self.device.submit_command_buffer(command_encoder.finish())

        return tlas

    def _compute_scene_bounds(self, scene_node: SceneNode) -> tuple[np.ndarray, np.ndarray]:
        bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        for mesh_id, _, transform_id in scene_node.instances:
            positions = scene_node.meshes[mesh_id].vertices[:, 0:3]
            transform = np.asarray(scene_node.transforms[transform_id].matrix.to_numpy(), dtype=np.float32)
            homogeneous = np.concatenate(
                [positions, np.ones((positions.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            world_positions = (transform @ homogeneous.T).T[:, 0:3]
            bounds_min = np.minimum(bounds_min, world_positions.min(axis=0))
            bounds_max = np.maximum(bounds_max, world_positions.max(axis=0))

        if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)):
            bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        extent = bounds_max - bounds_min
        pad = np.maximum(extent * 0.01, np.array([0.01, 0.01, 0.01], dtype=np.float32))
        return bounds_min - pad, bounds_max + pad

    def _make_static_shadow_matrices(self) -> tuple[spy.float4x4, spy.float4x4]:
        sun_direction = np.array(
            [float(self._sun_direction[0]), float(self._sun_direction[1]), float(self._sun_direction[2])],
            dtype=np.float32,
        )
        sun_norm = np.linalg.norm(sun_direction)
        if sun_norm <= 1e-6:
            sun_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            sun_direction /= sun_norm

        z_axis = -sun_direction  # Incoming light direction, from sun toward the scene.
        helper_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(helper_up, z_axis))) > 0.95:
            helper_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        x_axis = np.cross(helper_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        corners = np.array(
            [
                [x, y, z]
                for x in (self.scene_bounds_min[0], self.scene_bounds_max[0])
                for y in (self.scene_bounds_min[1], self.scene_bounds_max[1])
                for z in (self.scene_bounds_min[2], self.scene_bounds_max[2])
            ],
            dtype=np.float32,
        )
        light_space = np.stack(
            [corners @ x_axis, corners @ y_axis, corners @ z_axis],
            axis=1,
        )
        light_min = light_space.min(axis=0)
        light_max = light_space.max(axis=0)
        light_extent = np.maximum(light_max - light_min, np.array([1e-4, 1e-4, 1e-4], dtype=np.float32))

        world_to_shadow = np.array(
            [
                [x_axis[0] / light_extent[0], x_axis[1] / light_extent[0], x_axis[2] / light_extent[0], -light_min[0] / light_extent[0]],
                [y_axis[0] / light_extent[1], y_axis[1] / light_extent[1], y_axis[2] / light_extent[1], -light_min[1] / light_extent[1]],
                [z_axis[0] / light_extent[2], z_axis[1] / light_extent[2], z_axis[2] / light_extent[2], -light_min[2] / light_extent[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        shadow_to_world = np.linalg.inv(world_to_shadow).astype(np.float32)
        return spy.float4x4(world_to_shadow), spy.float4x4(shadow_to_world)

    def _reset_accumulation(self):
        self.event_distpacher.dispatch("camera_move", None)

    def _set_static_shadow_status(self, text: str):
        if self._static_shadow_status_text is not None:
            try:
                self._static_shadow_status_text.text = text
            except AttributeError:
                pass

    def _sync_shadow_mode_ui(self):
        pass

    def _create_sst_node_buffer(self, node_bytes: bytes, label: str):
        return self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label=label,
            data=np.frombuffer(node_bytes, dtype=np.uint8).copy(),
        )

    def _sst_fit_profile_name(self, profile: int | None = None) -> str:
        profile = self.sst_fit_profile if profile is None else profile
        names = ("Dual Bias", "Dual Half Visible", "Dual Visible", "Dual Relaxed Visible", "Dual Loose Visible")
        return names[min(max(int(profile), 0), len(names) - 1)]

    def _sst_preset_name(self, preset: int | None = None) -> str:
        preset = self.sst_preset if preset is None else preset
        names = ("Quality", "High Compression", "Manual")
        return names[min(max(int(preset), 0), len(names) - 1)]

    def _sst_tile_size_from_profile(self, profile: int | None = None) -> int:
        profile = self.sst_tile_profile if profile is None else profile
        tile_sizes = (64, 128, 256)
        return tile_sizes[min(max(int(profile), 0), len(tile_sizes) - 1)]

    def _create_sst_encoder_for_profile(self, profile: int) -> SparseShadowTreeEncoder:
        profile = min(max(int(profile), Scene.SST_FIT_DUAL_BIAS), Scene.SST_FIT_DUAL_LOOSE_VISIBLE)
        visibility_tolerance = 0.0
        if profile == Scene.SST_FIT_DUAL_HALF_VISIBLE:
            visibility_tolerance = self.static_shadow_depth_bias * 0.5
        elif profile == Scene.SST_FIT_DUAL_VISIBLE:
            visibility_tolerance = self.static_shadow_depth_bias
        elif profile == Scene.SST_FIT_DUAL_RELAXED_VISIBLE:
            visibility_tolerance = self.static_shadow_depth_bias * 2.0
        elif profile == Scene.SST_FIT_DUAL_LOOSE_VISIBLE:
            visibility_tolerance = self.static_shadow_depth_bias * 4.0

        return SparseShadowTreeEncoder(
            tile_size=self._sst_tile_size_from_profile(),
            min_leaf_size=max(1, int(self.sst_min_leaf_size)),
            dual_depth_slack=self.static_shadow_depth_bias * max(0.0, float(self.sst_dual_depth_slack_scale)),
            dual_max_leak=self.static_shadow_depth_bias,
            dual_visibility_tolerance=visibility_tolerance,
            shadow_bias=self.static_shadow_depth_bias,
            plane_quantization_search_radius=max(0, int(self.sst_quantization_search_radius)),
        )

    def _invalidate_sst_encoding(self):
        if self.sst_enabled:
            self.sst_enabled = False
            self.sst_stats = None
            if self.static_shadow_mode in (
                Scene.SHADOW_MODE_SST,
                Scene.SHADOW_MODE_PACKED_SST,
                Scene.SHADOW_MODE_COMPACT_SST,
                Scene.SHADOW_MODE_COMPACT_SST_PCF3,
                Scene.SHADOW_MODE_DECOMPRESSED_SST,
            ):
                self.static_shadow_mode = Scene.SHADOW_MODE_DEPTH_TEXTURE if self.static_shadow_enabled else Scene.SHADOW_MODE_REALTIME
                self._sync_shadow_mode_ui()
            self._reset_accumulation()

    def _refresh_sst_encoder(self):
        self.sst_encoder = self._create_sst_encoder_for_profile(self.sst_fit_profile)
        self.sst_tile_size = self.sst_encoder.tile_size
        self.sst_max_traversal_steps = self.sst_encoder.max_traversal_steps
        self.sst_branch_10bit_start_level = self.sst_encoder.branch_10bit_start_level

    def _sync_sst_option_ui(self):
        pass

    def _apply_sst_preset(self, preset: int):
        preset = min(max(int(preset), Scene.SST_PRESET_QUALITY), Scene.SST_PRESET_MANUAL)
        if preset == Scene.SST_PRESET_MANUAL:
            if self.sst_preset != Scene.SST_PRESET_MANUAL:
                self.sst_preset = Scene.SST_PRESET_MANUAL
                self._sync_sst_option_ui()
                self._update_static_shadow_status()
            return

        if preset == Scene.SST_PRESET_HIGH_COMPRESSION:
            target_profile = Scene.SST_FIT_DUAL_RELAXED_VISIBLE
        else:
            target_profile = Scene.SST_FIT_DUAL_VISIBLE

        changed = (
            self.sst_preset != preset or
            self.sst_fit_profile != target_profile or
            self.sst_tile_profile != 1 or
            self.sst_min_leaf_size != 2 or
            self.sst_quantization_search_radius != 0 or
            abs(self.sst_dual_depth_slack_scale - 1.0) > 1e-4
        )
        if not changed:
            return

        self.sst_preset = preset
        self.sst_fit_profile = target_profile
        self.sst_tile_profile = 1
        self.sst_min_leaf_size = 2
        self.sst_quantization_search_radius = 0
        self.sst_dual_depth_slack_scale = 1.0
        self._refresh_sst_encoder()
        self._invalidate_sst_encoding()
        self._sync_sst_option_ui()
        self._update_static_shadow_status()

    def _set_sst_fit_profile(self, profile: int):
        profile = min(max(int(profile), Scene.SST_FIT_DUAL_BIAS), Scene.SST_FIT_DUAL_LOOSE_VISIBLE)
        if profile == self.sst_fit_profile:
            return

        self.sst_preset = Scene.SST_PRESET_MANUAL
        self.sst_fit_profile = profile
        self._refresh_sst_encoder()
        self._invalidate_sst_encoding()
        self._sync_sst_option_ui()
        self._update_static_shadow_status()

    def _set_sst_encoder_options(self, tile_profile: int, min_leaf_size: int, quantization_radius: int, dual_slack_scale: float):
        tile_profile = min(max(int(tile_profile), 0), 2)
        min_leaf_size = min(max(int(min_leaf_size), 1), 8)
        quantization_radius = min(max(int(quantization_radius), 0), 2)
        dual_slack_scale = min(max(float(dual_slack_scale), 0.0), 8.0)
        if (
            tile_profile == self.sst_tile_profile and
            min_leaf_size == self.sst_min_leaf_size and
            quantization_radius == self.sst_quantization_search_radius and
            abs(dual_slack_scale - self.sst_dual_depth_slack_scale) < 1e-4
        ):
            return

        self.sst_preset = Scene.SST_PRESET_MANUAL
        self.sst_tile_profile = tile_profile
        self.sst_min_leaf_size = min_leaf_size
        self.sst_quantization_search_radius = quantization_radius
        self.sst_dual_depth_slack_scale = dual_slack_scale
        self._refresh_sst_encoder()
        self._invalidate_sst_encoding()
        self._sync_sst_option_ui()
        self._update_static_shadow_status()

    def _update_static_shadow_status(self):
        mode_names = ("Realtime", "Depth Texture", "SST", "Packed SST", "Compact SST", "Compact SST PCF3", "Decompressed SST")
        mode_name = mode_names[min(max(self.static_shadow_mode, 0), len(mode_names) - 1)]
        profile_name = self._sst_fit_profile_name()
        sst_config_text = (
            f"Tile={self._sst_tile_size_from_profile()} "
            f"Leaf={self.sst_min_leaf_size} "
        )
        if not self.static_shadow_enabled:
            self._set_static_shadow_status(f"Static shadow: not baked Strategy={profile_name} CompactPCF {sst_config_text}")
            return

        if self.sst_enabled and self.sst_stats is not None:
            self._set_static_shadow_status(
                f"Mode={mode_name} Strategy={profile_name} CompactPCF Depth={self.static_shadow_resolution} "
                f"{sst_config_text} "
                f"Tiles={self.sst_stats.tile_count} Nodes={self.sst_stats.node_count} "
                f"Steps={self.sst_stats.max_traversal_steps} "
                f"Packed={self.sst_stats.packed_compression_ratio:.2f}x "
                f"CompactOK={self.sst_stats.packed_decode_valid} "
                f"Leak>B={self.sst_stats.packed_leak_over_full_bias_percent:.3f}% "
                f"VisMis={self.sst_stats.packed_visibility_mismatch_percent:.3f}% "
                f"MeanLoss={self.sst_stats.mean_error_percent:.3f}%"
            )
        else:
            self._set_static_shadow_status(f"Mode={mode_name} Strategy={profile_name} CompactPCF Depth={self.static_shadow_resolution} {sst_config_text} SST=not encoded")

    def encode_sparse_shadow_tree(self):
        if not self.static_shadow_enabled:
            print("[Scene] Static shadow depth map must be baked before SST encoding")
            self._set_static_shadow_status("Static shadow: bake before SST encode")
            return None

        depth_data = self.static_shadow_depth_texture.to_numpy()
        depth_data = np.asarray(depth_data, dtype=np.float32).squeeze()
        if depth_data.ndim != 2:
            raise RuntimeError(f"Unexpected static shadow depth readback shape: {depth_data.shape}")
        second_depth_data = self.static_shadow_second_depth_texture.to_numpy()
        second_depth_data = np.asarray(second_depth_data, dtype=np.float32).squeeze()

        encoded = self.sst_encoder.encode(depth_data, second_depth_data)
        self.sst_node_buffer = self._create_sst_node_buffer(encoded.nodes, "sst_node_buffer")
        self.sst_packed_node_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_packed_node_buffer",
            data=encoded.fixed64_nodes,
        )
        self.sst_compact_words_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_compact_words_buffer",
            data=encoded.compact_words,
        )
        self.sst_compact_roots_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_compact_roots_buffer",
            data=encoded.compact_tile_roots,
        )
        self.sst_tile_roots_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="sst_tile_roots_buffer",
            data=encoded.tile_roots,
        )
        self.sst_decompressed_depth_texture = self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.static_shadow_resolution,
            height=self.static_shadow_resolution,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="sst_decompressed_depth_map",
        )
        self.sst_tile_grid = encoded.tile_grid
        self.sst_tile_size = encoded.tile_size
        self.sst_node_count = encoded.stats.node_count
        self.sst_max_traversal_steps = encoded.stats.max_traversal_steps
        self.sst_compact_valid = encoded.stats.packed_decode_valid
        self.sst_compact_word_count = int(encoded.compact_words.size) if self.sst_compact_valid else 0
        self.sst_branch_10bit_start_level = encoded.stats.branch_10bit_start_level
        self.sst_max_error = encoded.stats.max_error
        self.sst_mean_error = encoded.stats.mean_error
        self.sst_compression_ratio = encoded.stats.compression_ratio
        self.sst_stats = encoded.stats
        self.sst_enabled = True
        if self.sst_compact_valid:
            command_encoder = self.device.create_command_encoder()
            self.sst_decompressor.execute(
                command_encoder,
                self.sst_decompressed_depth_texture,
                self.sst_compact_words_buffer,
                self.sst_compact_roots_buffer,
                self.static_shadow_resolution,
                self.sst_tile_grid,
                self.sst_tile_size,
                self.sst_max_traversal_steps,
                self.sst_compact_word_count,
                self.sst_branch_10bit_start_level,
            )
            self.device.submit_command_buffer(command_encoder.finish())
        if self.sst_compact_valid:
            self.static_shadow_mode = Scene.SHADOW_MODE_COMPACT_SST_PCF3
        else:
            self.static_shadow_mode = Scene.SHADOW_MODE_DEPTH_TEXTURE
        self._update_static_shadow_status()
        self._reset_accumulation()
        print(
            "[Scene] SST encoded: "
            f"profile={self._sst_fit_profile_name()}, "
            f"tile={self.sst_tile_size}, leaf={self.sst_min_leaf_size}, "
            f"qRadius={self.sst_quantization_search_radius}, slack={self.sst_dual_depth_slack_scale:.2f}B, "
            f"tiles={encoded.stats.tile_count}, nodes={encoded.stats.node_count}, "
            f"ratio={encoded.stats.compression_ratio:.2f}x, "
            f"mean_loss={encoded.stats.mean_error_percent:.4f}%, "
            f"max_loss={encoded.stats.max_error_percent:.4f}%, "
            f"rmse={encoded.stats.rmse_error_percent:.4f}%, "
            f"leak_pixels={encoded.stats.leak_pixel_percent:.4f}%, "
            f"leak_gt_bias={encoded.stats.packed_leak_over_full_bias_percent:.4f}%, "
            f"vis_mismatch={encoded.stats.packed_visibility_mismatch_percent:.4f}%, "
            f"packed_ratio={encoded.stats.packed_compression_ratio:.2f}x, "
            f"compact_valid={encoded.stats.packed_decode_valid}, "
            f"decompressed={'yes' if self.sst_compact_valid else 'no'}, "
            f"steps={encoded.stats.max_traversal_steps}"
        )
        return encoded.stats

    def bake_static_shadow_depth_map(self):
        self.static_shadow_world_to_light, shadow_to_world = self._make_static_shadow_matrices()
        shadow_depth_texture = self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.static_shadow_resolution,
            height=self.static_shadow_resolution,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="static_shadow_depth_map",
        )
        shadow_second_depth_texture = self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.static_shadow_resolution,
            height=self.static_shadow_resolution,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="static_shadow_second_depth_map",
        )

        command_encoder = self.device.create_command_encoder()
        self.static_shadow_baker.execute(
            command_encoder,
            self,
            shadow_depth_texture,
            shadow_second_depth_texture,
            shadow_to_world,
        )
        self.device.submit_command_buffer(command_encoder.finish())

        self.static_shadow_depth_texture = shadow_depth_texture
        self.static_shadow_second_depth_texture = shadow_second_depth_texture
        self.static_shadow_enabled = True
        self.static_shadow_mode = Scene.SHADOW_MODE_DEPTH_TEXTURE
        self.static_shadow_baked_sun_direction = (
            float(self._sun_direction[0]),
            float(self._sun_direction[1]),
            float(self._sun_direction[2]),
        )
        self.sst_enabled = False
        self.sst_stats = None
        self._sync_shadow_mode_ui()
        self._update_static_shadow_status()
        self._reset_accumulation()
        print(f"[Scene] Static shadow depth map baked at {self.static_shadow_resolution}x{self.static_shadow_resolution}")
        if self.static_shadow_auto_encode_sst:
            self.encode_sparse_shadow_tree()

    def _generate_static_atmosphere_luts(self):
        """Generate transmittance and multi-scattering LUTs (only needed once)."""
        command_encoder = self.device.create_command_encoder()
        self.transmittance_lut_gen.generate(command_encoder)
        self.multiscatt_lut_gen.generate(command_encoder, self.transmittance_lut_gen.get_texture())
        self.device.submit_command_buffer(command_encoder.finish())

    def _generate_sky_view_lut(self):
        """Generate sky view LUT based on current sun direction."""
        command_encoder = self.device.create_command_encoder()
        self.sky_view_lut_gen.generate(
            command_encoder,
            self.transmittance_lut_gen.get_texture(),
            self.multiscatt_lut_gen.get_texture(),
            sun_direction=(float(self._sun_direction[0]), float(self._sun_direction[1]), float(self._sun_direction[2]))
        )
        self.device.submit_command_buffer(command_encoder.finish())
        self._sun_direction_dirty = False
    
    def _on_location_received(self, latitude: float, longitude: float):
        """Callback when async location fetch completes."""
        self._latitude = latitude
        self._longitude = longitude
        print(f"[Scene] Location updated: ({latitude:.4f}, {longitude:.4f})")
        # Trigger sun direction update if slider exists
        if self._hours_slider is not None:
            self._last_hours_value = None  # Force recalculation on next update

    def set_sun_from_location_time(
        self,
        latitude: float,
        longitude: float,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: int,
        timezone: float = 0.0,
        daylight_saving: bool = False
    ) -> SunPositionData:
        """
        Set sun direction based on geographic location and date/time.
        
        Args:
            latitude: Geographic latitude in degrees (-90 to 90, positive = North)
            longitude: Geographic longitude in degrees (-180 to 180, positive = East)
            year, month, day: Date components
            hour, minute, second: Time components (local time)
            timezone: Timezone offset from UTC in hours (e.g., 8.0 for UTC+8)
            daylight_saving: Whether daylight saving time is in effect
            
        Returns:
            SunPositionData containing all calculated sun position information
        """
        sun_data = SunPosition.calculate(
            latitude=latitude,
            longitude=longitude,
            year=year, month=month, day=day,
            hour=hour, minute=minute, second=second,
            timezone=timezone,
            daylight_saving=daylight_saving
        )
        
        # Update sun direction from calculated data
        self.sun_direction = spy.float3(
            sun_data.sun_direction[0],
            sun_data.sun_direction[1],
            sun_data.sun_direction[2]
        )
        
        return sun_data

    def _create_material_buffer(self, device: spy.Device, material_data_list: list['Scene.MaterialData']) -> spy.Buffer:
        """Create material buffer using BufferCursor for proper Handle binding.
        
        This method uses slangpy's BufferCursor to correctly write MaterialDesc structs
        including Texture2D.Handle fields which require special handling.
        """
        if len(material_data_list) == 0:
            # Create a dummy buffer with one default material
            material_data_list = [Scene.MaterialData(
                base_color=spy.float3(0.8, 0.8, 0.8),
                flags=0,
                emissive=spy.float3(0.0, 0.0, 0.0),
                shading_model=0,
                roughness=0.5,
                metallic=0.0,
                texture_flags=0,
                alpha_mode=ALPHA_MODE_OPAQUE,
                alpha_cutoff=0.5,
                specular_color=spy.float3(1.0, 1.0, 1.0),
                base_color_tex=None,
                normal_tex=None,
                roughness_tex=None,
                metallic_tex=None,
                emissive_tex=None,
                specular_color_tex=None
            )]
        
        # Load the common.slang module to get MaterialDesc layout
        module = device.load_module("common.slang")
        
        # Get StructuredBuffer<MaterialDesc> layout and extract element layout
        sb_type = module.layout.find_type_by_name("StructuredBuffer<MaterialDesc>")
        if sb_type is None:
            raise RuntimeError("Could not find StructuredBuffer<MaterialDesc> type in common.slang")
        
        material_desc_layout = module.layout.get_type_layout(sb_type).element_type_layout
        print(f"[Scene] MaterialDesc stride: {material_desc_layout.stride} bytes")
        
        # Create buffer with correct size
        buffer = device.create_buffer(
            size=len(material_data_list) * material_desc_layout.stride,
            usage=spy.BufferUsage.shader_resource,
            label="material_descs_buffer",
        )
        
        # Get default textures for null handles
        default_base_color = self.texture_manager.get_default_texture(TextureType.BASE_COLOR)
        default_normal = self.texture_manager.get_default_texture(TextureType.NORMAL)
        default_roughness = self.texture_manager.get_default_texture(TextureType.ROUGHNESS)
        default_metallic = self.texture_manager.get_default_texture(TextureType.METALLIC)
        default_emissive = self.texture_manager.get_default_texture(TextureType.EMISSIVE)
        default_specular_color = self.texture_manager.get_default_texture(TextureType.SPECULAR_COLOR)
        
        # Fill buffer using BufferCursor
        cursor = spy.BufferCursor(material_desc_layout, buffer, load_before_write=False)
        for i, mat in enumerate(material_data_list):
            cursor[i].base_color = mat.base_color
            cursor[i].flags = mat.flags
            cursor[i].emissive = mat.emissive
            cursor[i].shading_model = mat.shading_model
            cursor[i].roughness = mat.roughness
            cursor[i].metallic = mat.metallic
            cursor[i].texture_flags = mat.texture_flags
            cursor[i].alpha_mode = mat.alpha_mode
            cursor[i].alpha_cutoff = mat.alpha_cutoff
            cursor[i].specular_color = mat.specular_color
            
            # Set texture handles using descriptor_handle_ro from TextureView
            base_tex = mat.base_color_tex if mat.base_color_tex else default_base_color
            normal_tex = mat.normal_tex if mat.normal_tex else default_normal
            rough_tex = mat.roughness_tex if mat.roughness_tex else default_roughness
            metal_tex = mat.metallic_tex if mat.metallic_tex else default_metallic
            emiss_tex = mat.emissive_tex if mat.emissive_tex else default_emissive
            spec_color_tex = mat.specular_color_tex if mat.specular_color_tex else default_specular_color
            
            cursor[i].base_color_tex_handle = base_tex.view.descriptor_handle_ro
            cursor[i].normal_tex_handle = normal_tex.view.descriptor_handle_ro
            cursor[i].roughness_tex_handle = rough_tex.view.descriptor_handle_ro
            cursor[i].metallic_tex_handle = metal_tex.view.descriptor_handle_ro
            cursor[i].emissive_tex_handle = emiss_tex.view.descriptor_handle_ro
            cursor[i].specular_color_tex_handle = spec_color_tex.view.descriptor_handle_ro
        
        cursor.apply()
        return buffer

    def update(self):
        """Update scene state. Call this each frame before rendering."""
        # Increment frame index for stochastic alpha (once per frame, not per bind)
        self._frame_index += 1
        
        # Update sun direction from hours slider if present
        if self._hours_slider is not None:
            hours = self._hours_slider.value
            if self._last_hours_value is None or abs(hours - self._last_hours_value) > 0.001:
                self._last_hours_value = hours
                self._update_sun_from_hours(hours)
        
        # Update directional light intensity from slider if present
        if self._intensity_slider is not None:
            self._directional_light_intensity = self._intensity_slider.value

        if self._sun_direction_dirty:
            self._generate_sky_view_lut()
    
    def _update_sun_from_hours(self, hours: float):
        """Update sun direction based on hours slider value."""
        now = datetime.now()
        hour = int(hours)
        minute = int((hours - hour) * 60)
        second = int(((hours - hour) * 60 - minute) * 60)
        
        sun_data = SunPosition.calculate(
            latitude=self._latitude,
            longitude=self._longitude,
            year=now.year,
            month=now.month,
            day=now.day,
            hour=hour,
            minute=minute,
            second=second,
            timezone=self._timezone,
            daylight_saving=False
        )
        
        self.sun_direction = spy.float3(
            sun_data.sun_direction[0],
            sun_data.sun_direction[1],
            sun_data.sun_direction[2]
        )
    
    def setup_ui(self, ui_context: spy.ui.Context, ui_window: spy.ui.Window):
        """Setup scene-related UI elements."""
        now = datetime.now()
        current_hours = now.hour + now.minute / 60.0 + now.second / 3600.0
        self._hours_slider = spy.ui.SliderFloat(ui_window, 'Hours', min=0, max=23.99, value=current_hours)
        # Directional light intensity slider (0 to 20, default: PI)
        self._intensity_slider = spy.ui.SliderFloat(ui_window, 'Sun Intensity', min=0, max=20, value=self._directional_light_intensity)
        
        def bake_static_shadow_btn():
            self.bake_static_shadow_depth_map()
        
        spy.ui.Button(ui_window, 'Bake Static Shadow Depth Map', callback=bake_static_shadow_btn)
        self._static_shadow_status_text = spy.ui.Text(ui_window, 'Static shadow: not baked')
        # Initialize sun position with current time
        self._update_sun_from_hours(current_hours)
    
    @property
    def directional_light_intensity(self) -> float:
        """Get current directional light intensity."""
        return self._directional_light_intensity
    
    @directional_light_intensity.setter
    def directional_light_intensity(self, value: float):
        """Set directional light intensity."""
        self._directional_light_intensity = max(0.0, value)

    @property
    def sun_direction(self) -> spy.float3:
        return self._sun_direction
    
    @sun_direction.setter
    def sun_direction(self, value: spy.float3):
        """Set sun direction and mark sky LUT for regeneration."""
        if (self._sun_direction[0] != value[0] or 
            self._sun_direction[1] != value[1] or 
            self._sun_direction[2] != value[2]):
            self._sun_direction = value
            self._sun_direction_dirty = True
            if self.static_shadow_enabled:
                self.static_shadow_enabled = False
                self.sst_enabled = False
                self.sst_stats = None
                self.static_shadow_mode = Scene.SHADOW_MODE_REALTIME
                self._sync_shadow_mode_ui()
                self._set_static_shadow_status('Static shadow: sun changed, rebake needed')
                self._reset_accumulation()

    def bind(self, cursor: spy.ShaderCursor):
        cursor["tlas"] = self.tlas
        cursor["material_descs"] = self.material_descs_buffer  # Bind as StructuredBuffer<MaterialDesc>
        cursor["mesh_descs"] = self.mesh_descs_buffer
        cursor["instance_descs"] = self.instance_descs_buffer
        cursor["vertices"] = self.vertex_buffer
        cursor["indices"] = self.index_buffer
        cursor["transforms"] = self.transform_buffer
        cursor["inverse_transpose_transforms"] = self.inverse_transpose_transforms_buffer
        cursor["transmittance_lut"] = self.transmittance_lut_gen.get_texture()
        cursor["transmittance_sampler"] = self.transmittance_sampler
        cursor["sky_view_lut"] = self.sky_view_lut_gen.get_texture()
        cursor["linear_sampler"] = self.linear_sampler
        cursor["sun_direction"] = self._sun_direction
        cursor["instance_count"] = len(self.instance_descs)
        cursor["frame_index"] = self._frame_index
        cursor["static_shadow"]["depth_texture"] = self.static_shadow_depth_texture
        cursor["static_shadow"]["sst_decompressed_texture"] = self.sst_decompressed_depth_texture
        cursor["static_shadow"]["depth_sampler"] = self.static_shadow_sampler
        cursor["static_shadow"]["world_to_shadow"] = self.static_shadow_world_to_light
        cursor["static_shadow"]["enabled"] = 1 if self.static_shadow_enabled else 0
        cursor["static_shadow"]["resolution"] = self.static_shadow_resolution
        cursor["static_shadow"]["depth_bias"] = self.static_shadow_depth_bias
        cursor["static_shadow"]["shadow_mode"] = self.static_shadow_mode
        cursor["static_shadow"]["sst_enabled"] = 1 if self.sst_enabled else 0
        cursor["static_shadow"]["sst_tile_size"] = self.sst_tile_size
        cursor["static_shadow"]["sst_tile_grid"] = spy.uint2(self.sst_tile_grid[0], self.sst_tile_grid[1])
        cursor["static_shadow"]["sst_node_count"] = self.sst_node_count
        cursor["static_shadow"]["sst_max_traversal_steps"] = self.sst_max_traversal_steps
        cursor["static_shadow"]["sst_compact_word_count"] = self.sst_compact_word_count
        cursor["static_shadow"]["sst_branch_10bit_start_level"] = self.sst_branch_10bit_start_level
        cursor["static_shadow"]["sst_nodes"] = self.sst_node_buffer
        cursor["static_shadow"]["sst_packed_nodes"] = self.sst_packed_node_buffer
        cursor["static_shadow"]["sst_compact_words"] = self.sst_compact_words_buffer
        cursor["static_shadow"]["sst_tile_roots"] = self.sst_tile_roots_buffer
        cursor["static_shadow"]["sst_compact_roots"] = self.sst_compact_roots_buffer
        
        # Bind directional light parameters
        # Direction is same as sun_direction (pointing toward the sun)
        cursor["directional_light"]["direction"] = self._sun_direction
        cursor["directional_light"]["cos_half_angle"] = self._directional_light_cos_half_angle
        cursor["directional_light"]["color"] = self._directional_light_color
        cursor["directional_light"]["intensity"] = self._directional_light_intensity
        
        self.camera.bind(cursor["camera"])
