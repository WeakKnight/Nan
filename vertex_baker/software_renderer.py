from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from model import Material, Model
from surface_sampler import SurfaceSamples
from visibility_cone_visualizer import build_visibility_cone_line_segments


@dataclass
class Camera:
    eye: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]
    up: npt.NDArray[np.float32]
    fov_y_degrees: float = 45.0


def default_camera(model: Model) -> Camera:
    center = (model.bounds_min + model.bounds_max) * 0.5
    extent = model.bounds_max - model.bounds_min
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = max(radius, 1.0)
    eye = center + np.array([0.25 * radius, 0.08 * radius, 2.2 * radius], dtype=np.float32)
    return Camera(
        eye=eye.astype(np.float32),
        target=center.astype(np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


def _normalize(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return v
    return (v / length).astype(np.float32)


def _camera_basis(camera: Camera):
    forward = _normalize(camera.target - camera.eye)
    right = _normalize(np.cross(forward, camera.up))
    up = _normalize(np.cross(right, forward))
    return right, up, forward


def _project(
    positions: npt.NDArray[np.float32],
    camera: Camera,
    width: int,
    height: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    right, up, forward = _camera_basis(camera)
    rel = positions - camera.eye[None, :]
    view_x = rel @ right
    view_y = rel @ up
    view_z = rel @ forward

    aspect = float(width) / float(height)
    tan_half_fov = np.tan(np.radians(camera.fov_y_degrees) * 0.5)
    ndc_x = view_x / np.maximum(view_z, 1e-8) / (tan_half_fov * aspect)
    ndc_y = view_y / np.maximum(view_z, 1e-8) / tan_half_fov
    screen = np.stack(
        [
            (ndc_x * 0.5 + 0.5) * float(width),
            (0.5 - ndc_y * 0.5) * float(height),
        ],
        axis=1,
    ).astype(np.float32)
    return screen, view_z.astype(np.float32)


def _edge(a, b, p):
    return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])


def _sample_material(material: Material, uv: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if material.base_color_texture is None:
        return material.base_color[:3]

    texture = material.base_color_texture
    h, w, _ = texture.shape
    u = float(uv[0] % 1.0)
    v = float(uv[1] % 1.0)
    x = min(w - 1, max(0, int(u * (w - 1))))
    y = min(h - 1, max(0, int(v * (h - 1))))
    return (texture[y, x, :3] * material.base_color[:3]).astype(np.float32)


def render_unlit_preview(
    model: Model,
    samples: SurfaceSamples | None,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    point_radius: int = 2,
    point_color=(1.0, 0.55, 0.12),
    vertex_colors: list[npt.NDArray[np.float32]] | None = None,
    visibility_cones: list[npt.NDArray[np.float32]] | None = None,
    visibility_cone_length: float = 0.0,
    visibility_cone_rim_segments: int = 12,
    visibility_cone_xray: bool = False,
) -> None:
    camera = default_camera(model)
    color = np.zeros((height, width, 3), dtype=np.float32)
    color[:, :, :] = np.array([0.055, 0.06, 0.065], dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    for mesh_index, mesh in enumerate(model.meshes):
        material = model.materials[mesh.material_index]
        mesh_vertex_colors = None
        if vertex_colors is not None and mesh_index < len(vertex_colors):
            mesh_vertex_colors = np.asarray(vertex_colors[mesh_index], dtype=np.float32)
        screen, view_z = _project(mesh.positions, camera, width, height)
        for triangle in mesh.indices:
            i0, i1, i2 = [int(i) for i in triangle]
            z = np.array([view_z[i0], view_z[i1], view_z[i2]], dtype=np.float32)
            if np.any(z <= 1e-4):
                continue

            p = np.array([screen[i0], screen[i1], screen[i2]], dtype=np.float32)
            min_xy = np.floor(np.min(p, axis=0)).astype(np.int32)
            max_xy = np.ceil(np.max(p, axis=0)).astype(np.int32)
            min_x = max(0, int(min_xy[0]))
            min_y = max(0, int(min_xy[1]))
            max_x = min(width - 1, int(max_xy[0]))
            max_y = min(height - 1, int(max_xy[1]))
            if min_x > max_x or min_y > max_y:
                continue

            area = _edge(p[0], p[1], p[2])
            if abs(float(area)) <= 1e-8:
                continue

            xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
            ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(xs, ys)
            points = np.stack([grid_x, grid_y], axis=-1)
            w0 = _edge(p[1], p[2], points) / area
            w1 = _edge(p[2], p[0], points) / area
            w2 = _edge(p[0], p[1], points) / area
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not np.any(inside):
                continue

            tri_depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
            depth_window = depth[min_y : max_y + 1, min_x : max_x + 1]
            update = inside & (tri_depth < depth_window)
            if not np.any(update):
                continue

            uv0, uv1, uv2 = mesh.uvs[i0], mesh.uvs[i1], mesh.uvs[i2]
            vc0 = vc1 = vc2 = None
            if mesh_vertex_colors is not None:
                vc0 = mesh_vertex_colors[i0]
                vc1 = mesh_vertex_colors[i1]
                vc2 = mesh_vertex_colors[i2]
            target = color[min_y : max_y + 1, min_x : max_x + 1]
            update_y, update_x = np.nonzero(update)
            for local_y, local_x in zip(update_y, update_x):
                if vc0 is not None and vc1 is not None and vc2 is not None:
                    value = w0[local_y, local_x] * vc0 + w1[local_y, local_x] * vc1 + w2[local_y, local_x] * vc2
                    if value.shape[0] == 1:
                        target[local_y, local_x, :] = value[0]
                    else:
                        target[local_y, local_x, :] = value[:3]
                else:
                    uv = w0[local_y, local_x] * uv0 + w1[local_y, local_x] * uv1 + w2[local_y, local_x] * uv2
                    target[local_y, local_x, :] = _sample_material(material, uv)
            depth_window[update] = tri_depth[update]

    if samples is not None and samples.positions.shape[0] > 0:
        _draw_samples(color, depth, samples.positions, camera, width, height, point_radius, np.asarray(point_color, dtype=np.float32), model)

    if visibility_cones is not None:
        cone_lines = build_visibility_cone_line_segments(
            model,
            visibility_cones,
            cone_length=float(visibility_cone_length),
            rim_segments=max(3, int(visibility_cone_rim_segments)),
        )
        _draw_line_segments(
            color,
            depth,
            cone_lines.starts,
            cone_lines.ends,
            cone_lines.colors,
            cone_lines.alphas,
            cone_lines.widths,
            camera,
            width,
            height,
            model,
            xray=bool(visibility_cone_xray),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(color, 0.0, 1.0)
    Image.fromarray((image * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(output_path)


def _draw_samples(
    color: npt.NDArray[np.float32],
    depth: npt.NDArray[np.float32],
    positions: npt.NDArray[np.float32],
    camera: Camera,
    width: int,
    height: int,
    point_radius: int,
    point_color: npt.NDArray[np.float32],
    model: Model,
) -> None:
    screen, view_z = _project(positions, camera, width, height)
    radius = max(1, int(point_radius))
    radius_sq = float(radius * radius)
    scene_radius = max(float(np.linalg.norm(model.bounds_max - model.bounds_min)), 1.0)
    depth_bias = scene_radius * 1e-4

    for i in range(positions.shape[0]):
        z = float(view_z[i])
        if z <= 1e-4:
            continue
        x = int(round(float(screen[i, 0])))
        y = int(round(float(screen[i, 1])))
        if x < -radius or y < -radius or x >= width + radius or y >= height + radius:
            continue
        if 0 <= x < width and 0 <= y < height and z > float(depth[y, x]) + depth_bias:
            continue

        for dy in range(-radius, radius + 1):
            py = y + dy
            if py < 0 or py >= height:
                continue
            for dx in range(-radius, radius + 1):
                px = x + dx
                if px < 0 or px >= width or dx * dx + dy * dy > radius_sq:
                    continue
                if z <= float(depth[py, px]) + depth_bias:
                    color[py, px, :] = point_color


def _draw_line_segments(
    color: npt.NDArray[np.float32],
    depth: npt.NDArray[np.float32],
    starts: npt.NDArray[np.float32],
    ends: npt.NDArray[np.float32],
    line_colors: npt.NDArray[np.float32],
    alphas: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int32],
    camera: Camera,
    width: int,
    height: int,
    model: Model,
    *,
    xray: bool,
) -> None:
    if starts.shape[0] == 0:
        return
    start_screen, start_z = _project(starts, camera, width, height)
    end_screen, end_z = _project(ends, camera, width, height)
    scene_diagonal = max(float(np.linalg.norm(model.bounds_max - model.bounds_min)), 1.0)
    depth_bias = scene_diagonal * 2.0e-4

    for index in range(starts.shape[0]):
        z0 = float(start_z[index])
        z1 = float(end_z[index])
        if z0 <= 1e-4 or z1 <= 1e-4:
            continue
        p0 = start_screen[index]
        p1 = end_screen[index]
        delta = p1 - p0
        step_count = max(1, int(np.ceil(float(np.max(np.abs(delta))))))
        if step_count > max(width, height) * 2:
            continue
        line_color = line_colors[index]
        alpha = float(np.clip(alphas[index], 0.0, 1.0))
        line_width = max(1, int(widths[index]))
        offset_begin = -(line_width // 2)
        offset_end = offset_begin + line_width
        inverse_z0 = 1.0 / z0
        inverse_z1 = 1.0 / z1
        for step in range(step_count + 1):
            t = float(step) / float(step_count)
            point = p0 + delta * t
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            inverse_z = inverse_z0 * (1.0 - t) + inverse_z1 * t
            line_z = 1.0 / max(inverse_z, 1e-12)
            for offset_y in range(offset_begin, offset_end):
                py = y + offset_y
                if py < 0 or py >= height:
                    continue
                for offset_x in range(offset_begin, offset_end):
                    px = x + offset_x
                    if px < 0 or px >= width:
                        continue
                    if not xray and line_z > float(depth[py, px]) + depth_bias:
                        continue
                    color[py, px, :] = color[py, px, :] * (1.0 - alpha) + line_color * alpha
