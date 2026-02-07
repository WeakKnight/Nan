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


class Scene:
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
        # TextureRecords for bindless handles (None means use default)
        base_color_tex: Optional[TextureRecord]
        normal_tex: Optional[TextureRecord]
        roughness_tex: Optional[TextureRecord]
        metallic_tex: Optional[TextureRecord]
        emissive_tex: Optional[TextureRecord]

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
        
        # Location for sun position calculation (default: Chengdu, async update)
        self._latitude: float = SunPosition.DEFAULT_LATITUDE
        self._longitude: float = SunPosition.DEFAULT_LONGITUDE
        self._timezone = SunPosition.get_local_timezone()
        self._last_hours_value: float | None = None  # Track slider changes
        
        # Frame index for stochastic alpha
        self._frame_index: int = 0
        
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
                base_color_tex=base_color_tex,
                normal_tex=normal_tex,
                roughness_tex=roughness_tex,
                metallic_tex=metallic_tex,
                emissive_tex=emissive_tex
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
                base_color_tex=None,
                normal_tex=None,
                roughness_tex=None,
                metallic_tex=None,
                emissive_tex=None
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
            # pad0, pad1, pad2 are automatically zero
            
            # Set texture handles using descriptor_handle_ro from TextureView
            base_tex = mat.base_color_tex if mat.base_color_tex else default_base_color
            normal_tex = mat.normal_tex if mat.normal_tex else default_normal
            rough_tex = mat.roughness_tex if mat.roughness_tex else default_roughness
            metal_tex = mat.metallic_tex if mat.metallic_tex else default_metallic
            emiss_tex = mat.emissive_tex if mat.emissive_tex else default_emissive
            
            cursor[i].base_color_tex_handle = base_tex.view.descriptor_handle_ro
            cursor[i].normal_tex_handle = normal_tex.view.descriptor_handle_ro
            cursor[i].roughness_tex_handle = rough_tex.view.descriptor_handle_ro
            cursor[i].metallic_tex_handle = metal_tex.view.descriptor_handle_ro
            cursor[i].emissive_tex_handle = emiss_tex.view.descriptor_handle_ro
        
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
        
        # Bind directional light parameters
        # Direction is same as sun_direction (pointing toward the sun)
        cursor["directional_light"]["direction"] = self._sun_direction
        cursor["directional_light"]["cos_half_angle"] = self._directional_light_cos_half_angle
        cursor["directional_light"]["color"] = self._directional_light_color
        cursor["directional_light"]["intensity"] = self._directional_light_intensity
        
        self.camera.bind(cursor["camera"])
