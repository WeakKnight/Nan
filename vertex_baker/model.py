from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from pygltflib import GLTF2
import trimesh


@dataclass
class Material:
    name: str
    base_color: npt.NDArray[np.float32]
    base_color_texture: npt.NDArray[np.float32] | None = None
    roughness: float = 1.0
    metallic: float = 0.0
    base_color_texture_path: Path | None = None
    metallic_roughness_texture: npt.NDArray[np.float32] | None = None
    metallic_roughness_texture_path: Path | None = None
    normal_texture: npt.NDArray[np.float32] | None = None
    normal_texture_path: Path | None = None
    normal_scale: float = 1.0
    occlusion_texture: npt.NDArray[np.float32] | None = None
    occlusion_texture_path: Path | None = None
    occlusion_strength: float = 1.0
    emissive: npt.NDArray[np.float32] | None = None
    emissive_texture: npt.NDArray[np.float32] | None = None
    emissive_texture_path: Path | None = None
    double_sided: bool = False


@dataclass
class Mesh:
    name: str
    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    uvs: npt.NDArray[np.float32]
    indices: npt.NDArray[np.uint32]
    material_index: int
    proxy_hash_positions: npt.NDArray[np.float32] | None = None
    proxy_hash_normals: npt.NDArray[np.float32] | None = None


@dataclass
class Model:
    meshes: list[Mesh]
    materials: list[Material]
    bounds_min: npt.NDArray[np.float32]
    bounds_max: npt.NDArray[np.float32]


def _load_image_rgba(path: Path) -> npt.NDArray[np.float32] | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGBA")
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def _load_gltf_materials(path: Path) -> list[Material]:
    gltf = GLTF2().load(str(path))
    asset_dir = path.parent
    image_paths: list[Path | None] = []
    for image in gltf.images or []:
        image_paths.append(asset_dir / image.uri if image.uri else None)

    texture_sources: list[int | None] = []
    for texture in gltf.textures or []:
        texture_sources.append(texture.source)

    def texture_path(texture_info) -> Path | None:
        if texture_info is None or texture_info.index is None:
            return None
        texture_index = int(texture_info.index)
        if not 0 <= texture_index < len(texture_sources):
            return None
        source_index = texture_sources[texture_index]
        if source_index is None or not 0 <= source_index < len(image_paths):
            return None
        return image_paths[source_index]

    materials: list[Material] = []
    for material in gltf.materials or []:
        base_color = np.ones(4, dtype=np.float32)
        base_color_path = None
        roughness = 1.0
        metallic = 1.0
        pbr = material.pbrMetallicRoughness
        if pbr is not None:
            if pbr.baseColorFactor is not None:
                base_color = np.asarray(pbr.baseColorFactor, dtype=np.float32)
            base_color_path = texture_path(pbr.baseColorTexture)
            if pbr.roughnessFactor is not None:
                roughness = float(pbr.roughnessFactor)
            if pbr.metallicFactor is not None:
                metallic = float(pbr.metallicFactor)
        metallic_roughness_path = texture_path(pbr.metallicRoughnessTexture if pbr is not None else None)
        normal_path = texture_path(material.normalTexture)
        occlusion_path = texture_path(material.occlusionTexture)
        emissive_path = texture_path(material.emissiveTexture)
        emissive = np.asarray(
            material.emissiveFactor if material.emissiveFactor is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        materials.append(
            Material(
                name=material.name or f"material_{len(materials)}",
                base_color=base_color,
                base_color_texture=_load_image_rgba(base_color_path) if base_color_path is not None else None,
                roughness=roughness,
                metallic=metallic,
                base_color_texture_path=base_color_path,
                metallic_roughness_texture=(
                    _load_image_rgba(metallic_roughness_path)
                    if metallic_roughness_path is not None
                    else None
                ),
                metallic_roughness_texture_path=metallic_roughness_path,
                normal_texture=_load_image_rgba(normal_path) if normal_path is not None else None,
                normal_texture_path=normal_path,
                normal_scale=(
                    float(material.normalTexture.scale)
                    if material.normalTexture is not None and material.normalTexture.scale is not None
                    else 1.0
                ),
                occlusion_texture=_load_image_rgba(occlusion_path) if occlusion_path is not None else None,
                occlusion_texture_path=occlusion_path,
                occlusion_strength=(
                    float(material.occlusionTexture.strength)
                    if material.occlusionTexture is not None and material.occlusionTexture.strength is not None
                    else 1.0
                ),
                emissive=emissive,
                emissive_texture=_load_image_rgba(emissive_path) if emissive_path is not None else None,
                emissive_texture_path=emissive_path,
                double_sided=bool(material.doubleSided),
            )
        )

    if not materials:
        materials.append(Material("default", np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)))
    return materials


def _material_lookup(materials: list[Material]) -> dict[str, int]:
    return {material.name: index for index, material in enumerate(materials)}


def load_gltf_model(path: str | Path) -> Model:
    path = Path(path)
    materials = _load_gltf_materials(path)
    material_by_name = _material_lookup(materials)

    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        trimesh_meshes = []
        for tri_mesh in loaded.dump(concatenate=False):
            node_name = tri_mesh.metadata.get("node") if hasattr(tri_mesh, "metadata") else None
            node_transform = np.eye(4, dtype=np.float64)
            if node_name is not None and node_name in loaded.graph.nodes_geometry:
                node_transform, _ = loaded.graph[node_name]
            trimesh_meshes.append((tri_mesh, np.asarray(node_transform, dtype=np.float64)))
    elif isinstance(loaded, trimesh.Trimesh):
        trimesh_meshes = [(loaded, np.eye(4, dtype=np.float64))]
    else:
        raise ValueError(f"Unsupported asset type: {type(loaded).__name__}")

    meshes: list[Mesh] = []
    for mesh_index, (tri_mesh, node_transform) in enumerate(trimesh_meshes):
        if len(tri_mesh.vertices) == 0 or len(tri_mesh.faces) == 0:
            continue

        positions = tri_mesh.vertices.astype(np.float32)
        normals = tri_mesh.vertex_normals.astype(np.float32)
        inverse_transform = np.linalg.inv(node_transform)
        homogeneous_positions = np.concatenate(
            [positions.astype(np.float64), np.ones((positions.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        proxy_hash_positions = (inverse_transform @ homogeneous_positions.T).T[:, :3].astype(np.float32)
        proxy_hash_normals = (inverse_transform[:3, :3] @ normals.astype(np.float64).T).T
        proxy_hash_normal_lengths = np.linalg.norm(proxy_hash_normals, axis=1)
        valid_proxy_hash_normals = proxy_hash_normal_lengths > 1e-20
        proxy_hash_normals[valid_proxy_hash_normals] /= proxy_hash_normal_lengths[valid_proxy_hash_normals, None]
        proxy_hash_normals[~valid_proxy_hash_normals] = 0.0
        if hasattr(tri_mesh.visual, "uv") and tri_mesh.visual.uv is not None:
            uvs = tri_mesh.visual.uv.astype(np.float32)
            uvs[:, 1] = 1.0 - uvs[:, 1]
        else:
            uvs = np.zeros((positions.shape[0], 2), dtype=np.float32)

        material_name = getattr(getattr(tri_mesh.visual, "material", None), "name", "")
        material_index = material_by_name.get(material_name, 0)
        meshes.append(
            Mesh(
                name=tri_mesh.metadata.get("name", f"mesh_{mesh_index}") if hasattr(tri_mesh, "metadata") else f"mesh_{mesh_index}",
                positions=positions,
                normals=normals,
                uvs=uvs,
                indices=tri_mesh.faces.astype(np.uint32),
                material_index=material_index,
                proxy_hash_positions=proxy_hash_positions,
                proxy_hash_normals=proxy_hash_normals.astype(np.float32),
            )
        )

    if not meshes:
        raise ValueError(f"Asset contains no renderable meshes: {path}")

    all_positions = np.concatenate([mesh.positions for mesh in meshes], axis=0)
    return Model(
        meshes=meshes,
        materials=materials,
        bounds_min=all_positions.min(axis=0).astype(np.float32),
        bounds_max=all_positions.max(axis=0).astype(np.float32),
    )
