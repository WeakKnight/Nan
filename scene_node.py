import slangpy as spy
import numpy as np
import trimesh
import os
import json
import math
from camera import Camera
from material import Material, ALPHA_MODE_OPAQUE, ALPHA_MODE_MASK, ALPHA_MODE_BLEND
from mesh import Mesh
from transform import Transform

# Monkey patch trimesh to allow emissiveFactor > 1.0 (HDR emissive)
def _patch_trimesh_emissive():
    """Patch trimesh PBRMaterial to allow HDR emissive values > 1.0"""
    try:
        from trimesh.visual.material import PBRMaterial
        
        # Store original setter
        original_emissive_setter = PBRMaterial.emissiveFactor.fset
        
        def patched_emissive_setter(self, value):
            """Allow emissiveFactor values > 1.0 for HDR lighting."""
            if value is None:
                self._data['emissiveFactor'] = None
                return
            value = np.array(value, dtype=np.float64).flatten()
            if len(value) == 3:
                # Skip the 0-1 range validation - allow HDR values
                self._data['emissiveFactor'] = value
            else:
                raise ValueError("emissiveFactor must be a 3-element array")
        
        # Replace the setter
        PBRMaterial.emissiveFactor = property(
            PBRMaterial.emissiveFactor.fget,
            patched_emissive_setter
        )
    except Exception as e:
        print(f"Warning: Failed to patch trimesh emissiveFactor: {e}")

_patch_trimesh_emissive()

class SceneNode:
    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.materials = []
        self.meshes = []
        self.transforms = []
        self.instances = []
        self.asset_path:str = None

    def add_material(self, material: Material):
        material_id = len(self.materials)
        self.materials.append(material)
        return material_id

    def add_mesh(self, mesh: Mesh):
        mesh_id = len(self.meshes)
        self.meshes.append(mesh)
        return mesh_id

    def add_transform(self, transform: Transform):
        transform_id = len(self.transforms)
        self.transforms.append(transform)
        return transform_id

    def add_instance(self, mesh_id: int, material_id: int, transform_id: int):
        instance_id = len(self.instances)
        self.instances.append((mesh_id, material_id, transform_id))
        return instance_id

    @staticmethod
    def _convert_trimesh_to_mesh(trimesh_mesh: trimesh.Trimesh, flip_uv_y: bool = False) -> Mesh:
        """Convert Trimesh object to PathTracer Mesh.
        
        Args:
            trimesh_mesh: Trimesh mesh object
            flip_uv_y: If True, flip UV Y coordinate (1.0 - y). Used for glTF format.
        """
        
        # Extract positions
        positions = trimesh_mesh.vertices.astype(np.float32)
        
        # Extract normals (Trimesh computes automatically)
        normals = trimesh_mesh.vertex_normals.astype(np.float32)
        
        # Handle UVs
        if hasattr(trimesh_mesh.visual, 'uv') and trimesh_mesh.visual.uv is not None:
            uvs = trimesh_mesh.visual.uv.astype(np.float32)
            # Flip UV Y for glTF (glTF uses top-left origin, we use bottom-left)
            if flip_uv_y:
                uvs[:, 1] = 1.0 - uvs[:, 1]
        else:
            # Generate placeholder UVs (all zeros)
            uvs = np.zeros((len(positions), 2), dtype=np.float32)
        
        # Pack into PathTracer vertex format [position(3), normal(3), uv(2)]
        vertex_count = len(positions)
        vertices = np.zeros((vertex_count, 8), dtype=np.float32)
        vertices[:, 0:3] = positions
        vertices[:, 3:6] = normals
        vertices[:, 6:8] = uvs
        
        # Get indices
        indices = trimesh_mesh.faces.astype(np.uint32)
        
        return Mesh(vertices, indices)

    @staticmethod
    def _extract_material(trimesh_mesh: trimesh.Trimesh, asset_dir: str, default: Material = None, gltf_textures: dict = None) -> Material:
        """
        Extract material from Trimesh including PBR texture paths.
        
        Args:
            trimesh_mesh: Trimesh mesh object
            asset_dir: Directory of the asset file (for resolving relative texture paths)
            default: Default material to use if no material info found
            gltf_textures: Optional dict from _load_gltf_textures() with texture paths
            
        Returns:
            Material with color values and texture paths
        """
        # Default values
        base_color = spy.float3(0.8, 0.8, 0.8)
        emissive = spy.float3(0.0, 0.0, 0.0)
        specular_color = spy.float3(1.0, 1.0, 1.0)
        roughness = 0.5
        metallic = 0.0
        
        # Texture paths
        base_color_texture = None
        normal_texture = None
        roughness_texture = None
        metallic_texture = None
        emissive_texture = None
        specular_color_texture = None
        
        # Alpha related
        alpha_mode = ALPHA_MODE_OPAQUE
        alpha_cutoff = 0.5
        double_sided = False
        
        def resolve_texture_path(texture_source) -> str | None:
            """Resolve texture path from trimesh texture source."""
            if texture_source is None:
                return None
            
            # Handle PIL Image - need to get file path
            if hasattr(texture_source, 'filename') and texture_source.filename:
                return texture_source.filename
            
            # Handle string path
            if isinstance(texture_source, str):
                # Make absolute if relative
                if not os.path.isabs(texture_source):
                    return os.path.join(asset_dir, texture_source)
                return texture_source
            
            # Handle trimesh texture with file attribute
            if hasattr(texture_source, 'file_path') and texture_source.file_path:
                path = texture_source.file_path
                if not os.path.isabs(path):
                    return os.path.join(asset_dir, path)
                return path
            
            return None
        
        # Try to extract from visual material
        if hasattr(trimesh_mesh.visual, 'material'):
            material = trimesh_mesh.visual.material
            
            # Debug: print material type and attributes
            print(f"[SceneNode] Material type: {type(material).__name__}")
            # List all public attributes of material
            mat_attrs = [a for a in dir(material) if not a.startswith('_')]
            print(f"[SceneNode]   All attributes: {mat_attrs}")
            tex_attrs = ['baseColorTexture', 'normalTexture', 'metallicRoughnessTexture', 'emissiveTexture']
            for attr in tex_attrs:
                if hasattr(material, attr):
                    val = getattr(material, attr)
                    print(f"[SceneNode]   {attr}: {type(val).__name__ if val else None} = {val}")
            
            # Extract base color
            if hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                color = np.array(material.baseColorFactor[:3], dtype=np.float32)
                if color.max() > 1.0:
                    color = color / 255.0
                base_color = spy.float3(color)
            elif hasattr(material, 'diffuse') and material.diffuse is not None:
                color = material.diffuse
                if isinstance(color, (list, np.ndarray)) and len(color) >= 3:
                    color = np.array(color[:3], dtype=np.float32)
                    if color.max() > 1.0:
                        color = color / 255.0
                    base_color = spy.float3(color)
            elif hasattr(material, 'main_color') and material.main_color is not None:
                color = np.array(material.main_color[:3], dtype=np.float32) / 255.0
                base_color = spy.float3(color)
            
            # Extract emissive
            if hasattr(material, 'emissiveFactor') and material.emissiveFactor is not None:
                emissive_val = np.array(material.emissiveFactor[:3], dtype=np.float32)
                emissive = spy.float3(emissive_val)
            
            # Extract roughness
            if hasattr(material, 'roughnessFactor') and material.roughnessFactor is not None:
                roughness = float(material.roughnessFactor)
            
            # Extract metallic
            if hasattr(material, 'metallicFactor') and material.metallicFactor is not None:
                metallic = float(material.metallicFactor)
            
            # Extract texture paths (PBRMaterial)
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                base_color_texture = resolve_texture_path(material.baseColorTexture)
            
            if hasattr(material, 'normalTexture') and material.normalTexture is not None:
                normal_texture = resolve_texture_path(material.normalTexture)
            
            if hasattr(material, 'metallicRoughnessTexture') and material.metallicRoughnessTexture is not None:
                # glTF uses combined metallic-roughness texture (G=roughness, B=metallic)
                # For now, use same texture for both - shader will sample correct channel
                combined_path = resolve_texture_path(material.metallicRoughnessTexture)
                if combined_path:
                    roughness_texture = combined_path
                    metallic_texture = combined_path
            
            if hasattr(material, 'emissiveTexture') and material.emissiveTexture is not None:
                emissive_texture = resolve_texture_path(material.emissiveTexture)
            
            # Extract alphaMode and alphaCutoff (glTF PBR)
            if hasattr(material, 'alphaMode') and material.alphaMode is not None:
                mode_str = material.alphaMode.upper() if isinstance(material.alphaMode, str) else material.alphaMode
                if mode_str == 'OPAQUE' or mode_str == 0:
                    alpha_mode = ALPHA_MODE_OPAQUE
                elif mode_str == 'MASK' or mode_str == 1:
                    alpha_mode = ALPHA_MODE_MASK
                elif mode_str == 'BLEND' or mode_str == 2:
                    alpha_mode = ALPHA_MODE_BLEND
            
            if hasattr(material, 'alphaCutoff') and material.alphaCutoff is not None:
                alpha_cutoff = float(material.alphaCutoff)

            if hasattr(material, 'doubleSided') and material.doubleSided is not None:
                double_sided = bool(material.doubleSided)
            
            # If trimesh didn't provide textures, try to get from gltf_textures by material name
            if gltf_textures and hasattr(material, 'name') and material.name:
                mat_name = material.name
                if mat_name in gltf_textures:
                    tex_info = gltf_textures[mat_name]
                    if not base_color_texture and tex_info.get('baseColorTexture'):
                        base_color_texture = tex_info['baseColorTexture']
                    if not normal_texture and tex_info.get('normalTexture'):
                        normal_texture = tex_info['normalTexture']
                    if not roughness_texture and tex_info.get('metallicRoughnessTexture'):
                        roughness_texture = tex_info['metallicRoughnessTexture']
                        metallic_texture = tex_info['metallicRoughnessTexture']
                    if not emissive_texture and tex_info.get('emissiveTexture'):
                        emissive_texture = tex_info['emissiveTexture']
                    if not specular_color_texture and tex_info.get('specularColorTexture'):
                        specular_color_texture = tex_info['specularColorTexture']
                    # Extract specular color factor
                    if tex_info.get('specularColorFactor') is not None:
                        sc = tex_info['specularColorFactor']
                        specular_color = spy.float3(float(sc[0]), float(sc[1]), float(sc[2]))
                    # Get alpha mode/cutoff from gltf_textures if not already set
                    if alpha_mode == ALPHA_MODE_OPAQUE and tex_info.get('alphaMode'):
                        mode_str = tex_info['alphaMode'].upper() if isinstance(tex_info['alphaMode'], str) else tex_info['alphaMode']
                        if mode_str == 'OPAQUE' or mode_str == 0:
                            alpha_mode = ALPHA_MODE_OPAQUE
                        elif mode_str == 'MASK' or mode_str == 1:
                            alpha_mode = ALPHA_MODE_MASK
                        elif mode_str == 'BLEND' or mode_str == 2:
                            alpha_mode = ALPHA_MODE_BLEND
                    if tex_info.get('alphaCutoff') is not None:
                        alpha_cutoff = float(tex_info['alphaCutoff'])
                    if tex_info.get('doubleSided') is not None:
                        double_sided = bool(tex_info['doubleSided'])
        
        # Try vertex colors as fallback for base color
        if base_color_texture is None:
            if hasattr(trimesh_mesh.visual, 'vertex_colors'):
                vertex_colors = trimesh_mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    avg_color = np.mean(vertex_colors[:, :3], axis=0).astype(np.float32) / 255.0
                    base_color = spy.float3(avg_color)
        
        # Use default if provided and no material info found
        if default is not None and not hasattr(trimesh_mesh.visual, 'material'):
            return default
        
        # Debug: print extracted texture paths
        if base_color_texture or normal_texture or roughness_texture or metallic_texture or emissive_texture or specular_color_texture:
            print(f"[SceneNode] Extracted textures:")
            print(f"  base_color: {base_color_texture}")
            print(f"  normal: {normal_texture}")
            print(f"  roughness: {roughness_texture}")
            print(f"  metallic: {metallic_texture}")
            print(f"  emissive: {emissive_texture}")
            print(f"  specular_color: {specular_color_texture}")
        
        return Material(
            base_color=base_color,
            emissive=emissive,
            specular_color=specular_color,
            roughness=roughness,
            metallic=metallic,
            base_color_texture=base_color_texture,
            normal_texture=normal_texture,
            roughness_texture=roughness_texture,
            metallic_texture=metallic_texture,
            emissive_texture=emissive_texture,
            specular_color_texture=specular_color_texture,
            alpha_mode=alpha_mode,
            alpha_cutoff=alpha_cutoff,
            double_sided=double_sided,
        )

    @staticmethod
    def _normalize_to_unit_sphere(trimesh_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Normalize mesh to fit in a unit sphere centered at origin."""
        trimesh_mesh.vertices = trimesh_mesh.vertices - trimesh_mesh.vertices.mean(axis=0)
        bounding_sphere_radius = np.linalg.norm(trimesh_mesh.vertices, ord=2, axis=-1).max() * 2.0
        trimesh_mesh.vertices = trimesh_mesh.vertices / bounding_sphere_radius
        return trimesh_mesh

    @staticmethod
    def _apply_transforms(trimesh_mesh: trimesh.Trimesh, scale: float = 1.0, auto_center: bool = False) -> trimesh.Trimesh:
        """Apply scaling and centering transforms."""
        
        # Auto-center
        if auto_center:
            center = trimesh_mesh.bounds.mean(axis=0)
            trimesh_mesh.apply_translation(-center)
        
        # Apply scale
        if scale != 1.0:
            trimesh_mesh.apply_scale(scale)
        
        return trimesh_mesh

    @staticmethod
    def _setup_camera_for_bounds(scene_node: 'SceneNode', trimesh_meshes: list):
        """Setup camera to frame all loaded meshes."""
        
        # Compute combined bounds
        all_vertices = []
        for mesh in trimesh_meshes:
            all_vertices.append(mesh.vertices)
        
        if len(all_vertices) > 0:
            all_vertices = np.concatenate(all_vertices, axis=0)
            bounds_min = all_vertices.min(axis=0)
            bounds_max = all_vertices.max(axis=0)
            center = (bounds_min + bounds_max) / 2.0
            size = np.linalg.norm(bounds_max - bounds_min)
            
            # Position camera to view entire model
            scene_node.camera.target = spy.float3(center.astype(np.float32))
            
            # Camera distance: 1.5x the bounding sphere radius
            distance = size * 1.5
            scene_node.camera.position = spy.float3(
                float(center[0]), 
                float(center[1] + size * 0.3),  # Slightly above
                float(center[2] + distance)
            )

    @classmethod
    def _load_gltf_textures(cls, path: str) -> dict:
        """
        Load glTF texture information using pygltflib.
        
        Returns:
            dict: Mapping from mesh name to texture paths dict
            {
                'mesh_name': {
                    'baseColorTexture': 'path/to/texture.png',
                    'normalTexture': ...,
                    ...
                }
            }
        """
        try:
            from pygltflib import GLTF2
        except ImportError:
            print("[SceneNode] Warning: pygltflib not installed, cannot load glTF textures")
            return {}
        
        asset_dir = os.path.dirname(os.path.abspath(path))
        
        try:
            gltf = GLTF2().load(path)
        except Exception as e:
            print(f"[SceneNode] Warning: Failed to load glTF with pygltflib: {e}")
            return {}
        
        # Build image path lookup
        image_paths = {}
        for idx, image in enumerate(gltf.images or []):
            if image.uri:
                # Handle relative paths and normalize path separators
                img_path = os.path.join(asset_dir, image.uri) if not os.path.isabs(image.uri) else image.uri
                img_path = os.path.normpath(img_path)  # Normalize path separators
                image_paths[idx] = img_path
        
        # Build texture -> image lookup
        texture_to_image = {}
        for idx, texture in enumerate(gltf.textures or []):
            if texture.source is not None:
                texture_to_image[idx] = image_paths.get(texture.source)
        
        # Build material textures lookup
        material_textures = {}
        for idx, mat in enumerate(gltf.materials or []):
            tex_info = {}
            
            # Standard PBR textures
            if mat.pbrMetallicRoughness:
                pbr = mat.pbrMetallicRoughness
                if pbr.baseColorTexture and pbr.baseColorTexture.index is not None:
                    tex_info['baseColorTexture'] = texture_to_image.get(pbr.baseColorTexture.index)
                if pbr.metallicRoughnessTexture and pbr.metallicRoughnessTexture.index is not None:
                    path_mr = texture_to_image.get(pbr.metallicRoughnessTexture.index)
                    tex_info['metallicRoughnessTexture'] = path_mr
            
            # Normal, emissive, occlusion textures (at material level)
            if mat.normalTexture and mat.normalTexture.index is not None:
                tex_info['normalTexture'] = texture_to_image.get(mat.normalTexture.index)
            if mat.emissiveTexture and mat.emissiveTexture.index is not None:
                tex_info['emissiveTexture'] = texture_to_image.get(mat.emissiveTexture.index)
            if mat.occlusionTexture and mat.occlusionTexture.index is not None:
                tex_info['occlusionTexture'] = texture_to_image.get(mat.occlusionTexture.index)
            
            # KHR_materials_pbrSpecularGlossiness extension
            if mat.extensions and 'KHR_materials_pbrSpecularGlossiness' in mat.extensions:
                sg_ext = mat.extensions['KHR_materials_pbrSpecularGlossiness']
                if 'diffuseTexture' in sg_ext and sg_ext['diffuseTexture']:
                    tex_idx = sg_ext['diffuseTexture'].get('index')
                    if tex_idx is not None:
                        tex_info['baseColorTexture'] = texture_to_image.get(tex_idx)
                if 'specularGlossinessTexture' in sg_ext and sg_ext['specularGlossinessTexture']:
                    tex_idx = sg_ext['specularGlossinessTexture'].get('index')
                    if tex_idx is not None:
                        # Use as metallicRoughness for now (not ideal but better than nothing)
                        tex_info['metallicRoughnessTexture'] = texture_to_image.get(tex_idx)
            
            # KHR_materials_specular extension
            if mat.extensions and 'KHR_materials_specular' in mat.extensions:
                spec_ext = mat.extensions['KHR_materials_specular']
                if 'specularColorTexture' in spec_ext and spec_ext['specularColorTexture']:
                    tex_idx = spec_ext['specularColorTexture'].get('index')
                    if tex_idx is not None:
                        tex_info['specularColorTexture'] = texture_to_image.get(tex_idx)
                if 'specularColorFactor' in spec_ext and spec_ext['specularColorFactor'] is not None:
                    tex_info['specularColorFactor'] = spec_ext['specularColorFactor']
            
            # Extract alpha mode and cutoff from glTF material
            if mat.alphaMode is not None:
                tex_info['alphaMode'] = mat.alphaMode
            if mat.alphaCutoff is not None:
                tex_info['alphaCutoff'] = mat.alphaCutoff
            if mat.doubleSided is not None:
                tex_info['doubleSided'] = mat.doubleSided
            
            material_textures[idx] = tex_info
            material_textures[mat.name] = tex_info  # Also store by name
        
        print(f"[SceneNode] Loaded {len(material_textures)//2} material textures from glTF")
        
        return material_textures

    @classmethod
    def load_asset(cls, 
                   path: str,
                   default_material: Material = None,
                   scale: float = 1.0,
                   auto_center: bool = False) -> 'SceneNode':
        """
        Load a 3D asset file and return a SceneNode containing the complete scene.
        
        Args:
            path: Path to asset file (OBJ, STL, PLY, glTF, etc.)
            default_material: Material to use if asset has no material info
            scale: Scale factor to apply to entire asset
            auto_center: If True, center asset at world origin
        
        Returns:
            SceneNode: Complete scene with all meshes, materials, and instances
        """
        
        print(f"[SceneNode] load_asset called with path: {path}")
        
        # 1. Validate file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Asset not found: {path}")
        
        # For glTF files, pre-load texture information using pygltflib
        gltf_textures = {}
        is_gltf = path.lower().endswith(('.gltf', '.glb'))
        if is_gltf:
            gltf_textures = cls._load_gltf_textures(path)
        
        # 2. Load with Trimesh (don't use force='mesh' which merges everything and loses materials)
        try:
            loaded = trimesh.load(path)
            print(f"[SceneNode] Loaded type: {type(loaded).__name__}")
        except Exception as e:
            raise ValueError(f"Failed to load asset: {e}")
        
        # 3. Handle Scene vs single mesh
        if isinstance(loaded, trimesh.Scene):
            # Use dump(concatenate=False) to get meshes with transforms baked in, but not merged
            # This preserves individual meshes with their materials while applying scene graph transforms
            trimesh_meshes = list(loaded.dump(concatenate=False))
        elif isinstance(loaded, trimesh.Trimesh):
            trimesh_meshes = [loaded]
        else:
            raise ValueError(f"Unsupported asset type: {type(loaded)}")
        
        # Filter out empty meshes
        trimesh_meshes = [m for m in trimesh_meshes if len(m.vertices) > 0]
        if len(trimesh_meshes) == 0:
            raise ValueError(f"Asset contains no valid geometry: {path}")
        
        # 4. Create new SceneNode
        scene_node = cls()
        scene_node.asset_path = path
        
        # Get asset directory for resolving relative texture paths
        asset_dir = os.path.dirname(os.path.abspath(path))
        
        # 5. Process each mesh
        for trimesh_mesh in trimesh_meshes:
            # Apply transforms
            trimesh_mesh = cls._apply_transforms(trimesh_mesh, scale, auto_center)
            
            # Convert to PathTracer mesh (flip UV Y for glTF format)
            mesh = cls._convert_trimesh_to_mesh(trimesh_mesh, flip_uv_y=is_gltf)
            mesh_id = scene_node.add_mesh(mesh)
            
            # Extract/create material with texture paths
            material = cls._extract_material(trimesh_mesh, asset_dir, default_material, gltf_textures)
            material_id = scene_node.add_material(material)
            
            # Create identity transform
            transform = Transform()
            transform.update_matrix()
            transform_id = scene_node.add_transform(transform)
            
            # Create instance
            scene_node.add_instance(mesh_id, material_id, transform_id)
        
        # 6. Setup camera
        cls._setup_camera_for_bounds(scene_node, trimesh_meshes)
        
        return scene_node

    @staticmethod
    def _convert_z_up_to_y_up(vec: list) -> list:
        """
        Convert a vector from Z-up to Y-up coordinate system.
        Z-up to Y-up: (x, y, z) -> (x, z, -y)
        
        Args:
            vec: 3D vector [x, y, z] in Z-up coordinates
        
        Returns:
            3D vector [x, y, z] in Y-up coordinates
        """
        if len(vec) != 3:
            return vec
        return [vec[0], vec[2], -vec[1]]
    
    @staticmethod
    def _convert_rotation_z_up_to_y_up(rot: list) -> list:
        """
        Convert rotation from Z-up to Y-up coordinate system.
        
        Args:
            rot: Rotation [rx, ry, rz] in degrees in Z-up coordinates
        
        Returns:
            Rotation [rx, ry, rz] in degrees in Y-up coordinates
        """
        if len(rot) != 3:
            return rot
        # When converting from Z-up to Y-up, we need to adjust rotations
        # Z-up (rx, ry, rz) -> Y-up (rz, rx, -ry)
        return [rot[2], rot[0], -rot[1]]

    def add_scene_node(self, node: 'SceneNode', transform: Transform) -> list[int]:
        """
        Add another SceneNode to this scene.
        
        Args:
            node: SceneNode to add
            transform: Transform to apply to the node
        
        Returns:
            list[int]: Instance IDs of all added objects
        """
        
        # Add the transform
        transform_id = self.add_transform(transform)
        
        # Transfer meshes and materials from node to this scene
        mesh_id_mapping = {}
        material_id_mapping = {}
        
        # Add all meshes
        for idx, mesh in enumerate(node.meshes):
            new_mesh_id = self.add_mesh(mesh)
            mesh_id_mapping[idx] = new_mesh_id
        
        # Add all materials
        for idx, mat in enumerate(node.materials):
            new_material_id = self.add_material(mat)
            material_id_mapping[idx] = new_material_id
        
        # Create instances with the provided transform
        instance_ids = []
        for mesh_id, material_id, _ in node.instances:
            new_mesh_id = mesh_id_mapping[mesh_id]
            new_material_id = material_id_mapping[material_id]
            instance_id = self.add_instance(new_mesh_id, new_material_id, transform_id)
            instance_ids.append(instance_id)
        
        return instance_ids

    @classmethod
    def demo(cls):
        """Create a Cornell box scene."""
        scene_node = cls()

        # Camera setup - at the front opening of the box looking in
        # Box front edge is at z=1, place camera just inside
        scene_node.camera.target = spy.float3(0, 1, 0)
        scene_node.camera.position = spy.float3(0, 1, 2.95)
        scene_node.camera.fov = 60.0  # slightly narrower FOV

        # Materials
        white_wall_mat = scene_node.add_material(
            Material(
                base_color=spy.float3(0.73, 0.73, 0.73),
                double_sided=True,
            )
        )
        red_wall_mat = scene_node.add_material(
            Material(
                base_color=spy.float3(0.65, 0.05, 0.05),
                double_sided=True,
            )
        )
        green_wall_mat = scene_node.add_material(
            Material(
                base_color=spy.float3(0.12, 0.45, 0.15),
                double_sided=True,
            )
        )
        white_box_mat = scene_node.add_material(
            Material(base_color=spy.float3(0.73, 0.73, 0.73))
        )
        light_mat = scene_node.add_material(Material(base_color=spy.float3(1.0, 1.0, 1.0), emissive=spy.float3(15.0, 15.0, 15.0)))

        # Shared quad mesh for floor/ceiling (XZ plane, normal +Y) - spans x:[-1,1], z:[-1,1]
        quad_mesh = scene_node.add_mesh(Mesh.create_quad([2, 2]))
        
        # Back wall quad (XY plane, normal +Z) - spans x:[-1,1], y:[0,2] at z=-1
        back_wall_mesh = scene_node.add_mesh(Mesh.create_quad_xy([2, 2], face_positive_z=True))
        
        # Left wall (YZ plane, normal +X) - spans y:[0,2], z:[-1,1]
        left_wall_mesh = scene_node.add_mesh(Mesh.create_quad_yz([2, 2], face_positive_x=True))
        
        # Right wall (YZ plane, normal -X) - spans y:[0,2], z:[-1,1]
        right_wall_mesh = scene_node.add_mesh(Mesh.create_quad_yz([2, 2], face_positive_x=False))

        # Floor (white) - at y=0, center at (0,0,0)
        floor_transform = Transform()
        floor_transform.translation = spy.float3(0, 0, 0)
        floor_transform.update_matrix()
        floor_tid = scene_node.add_transform(floor_transform)
        scene_node.add_instance(quad_mesh, white_wall_mat, floor_tid)

        # Ceiling (white) - at y=2, flipped to face down
        ceiling_transform = Transform()
        ceiling_transform.translation = spy.float3(0, 2, 0)
        ceiling_transform.rotation = spy.float3(math.pi, 0, 0)
        ceiling_transform.update_matrix()
        ceiling_tid = scene_node.add_transform(ceiling_transform)
        scene_node.add_instance(quad_mesh, white_wall_mat, ceiling_tid)

        # Back wall (white) - at z=-1, center at (0,1,-1)
        back_transform = Transform()
        back_transform.translation = spy.float3(0, 1, -1)
        back_transform.update_matrix()
        back_tid = scene_node.add_transform(back_transform)
        scene_node.add_instance(back_wall_mesh, white_wall_mat, back_tid)

        # Left wall (red) - at x=-1, center at (-1,1,0), facing +X
        left_transform = Transform()
        left_transform.translation = spy.float3(-1, 1, 0)
        left_transform.update_matrix()
        left_tid = scene_node.add_transform(left_transform)
        scene_node.add_instance(left_wall_mesh, red_wall_mat, left_tid)

        # Right wall (green) - at x=1, center at (1,1,0), facing -X
        right_transform = Transform()
        right_transform.translation = spy.float3(1, 1, 0)
        right_transform.update_matrix()
        right_tid = scene_node.add_transform(right_transform)
        scene_node.add_instance(right_wall_mesh, green_wall_mat, right_tid)

        # Light on ceiling (small quad)
        light_mesh = scene_node.add_mesh(Mesh.create_quad([0.4, 0.4]))
        light_transform = Transform()
        light_transform.translation = spy.float3(0, 1.99, 0)
        light_transform.rotation = spy.float3(math.pi, 0, 0)
        light_transform.update_matrix()
        light_tid = scene_node.add_transform(light_transform)
        scene_node.add_instance(light_mesh, light_mat, light_tid)

        # Tall box (left side) - rotated slightly
        tall_box_mesh = scene_node.add_mesh(Mesh.create_cube(spy.float3(0.5, 1.2, 0.5)))
        tall_box_transform = Transform()
        tall_box_transform.translation = spy.float3(-0.35, 0.6, -0.3)
        tall_box_transform.rotation = spy.float3(0, 18, 0)
        tall_box_transform.update_matrix()
        tall_box_tid = scene_node.add_transform(tall_box_transform)
        scene_node.add_instance(tall_box_mesh, white_box_mat, tall_box_tid)

        # Short box (right side) - rotated slightly the other way
        short_box_mesh = scene_node.add_mesh(Mesh.create_cube(spy.float3(0.5, 0.6, 0.5)))
        short_box_transform = Transform()
        short_box_transform.translation = spy.float3(0.35, 0.3, 0.3)
        short_box_transform.rotation = spy.float3(0, -18, 0)
        short_box_transform.update_matrix()
        short_box_tid = scene_node.add_transform(short_box_transform)
        scene_node.add_instance(short_box_mesh, white_box_mat, short_box_tid)

        return scene_node

