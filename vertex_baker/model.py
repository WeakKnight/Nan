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


@dataclass
class Mesh:
    name: str
    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    uvs: npt.NDArray[np.float32]
    indices: npt.NDArray[np.uint32]
    material_index: int


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

    materials: list[Material] = []
    for material in gltf.materials or []:
        base_color = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
        texture_data = None
        pbr = material.pbrMetallicRoughness
        if pbr is not None:
            if pbr.baseColorFactor is not None:
                base_color = np.asarray(pbr.baseColorFactor, dtype=np.float32)
            if pbr.baseColorTexture is not None:
                texture_index = pbr.baseColorTexture.index
                if texture_index is not None and 0 <= texture_index < len(texture_sources):
                    source_index = texture_sources[texture_index]
                    if source_index is not None and 0 <= source_index < len(image_paths):
                        image_path = image_paths[source_index]
                        if image_path is not None:
                            texture_data = _load_image_rgba(image_path)
        materials.append(
            Material(
                name=material.name or f"material_{len(materials)}",
                base_color=base_color,
                base_color_texture=texture_data,
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
        trimesh_meshes = list(loaded.dump(concatenate=False))
    elif isinstance(loaded, trimesh.Trimesh):
        trimesh_meshes = [loaded]
    else:
        raise ValueError(f"Unsupported asset type: {type(loaded).__name__}")

    meshes: list[Mesh] = []
    for mesh_index, tri_mesh in enumerate(trimesh_meshes):
        if len(tri_mesh.vertices) == 0 or len(tri_mesh.faces) == 0:
            continue

        positions = tri_mesh.vertices.astype(np.float32)
        normals = tri_mesh.vertex_normals.astype(np.float32)
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
