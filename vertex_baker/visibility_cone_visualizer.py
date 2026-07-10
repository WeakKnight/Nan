from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from model import Model


@dataclass(frozen=True)
class VisibilityConeLineSegments:
    starts: npt.NDArray[np.float32]
    ends: npt.NDArray[np.float32]
    colors: npt.NDArray[np.float32]
    alphas: npt.NDArray[np.float32]
    widths: npt.NDArray[np.int32]
    cone_count: int
    invalid_direction_count: int
    cone_length: float


def _normalize(value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32] | None:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1e-8:
        return None
    return (value / length).astype(np.float32)


def _cone_frame(axis: npt.NDArray[np.float32]):
    helper = (
        np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(axis[2])) > 0.9
        else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    tangent = _normalize(np.cross(helper, axis))
    if tangent is None:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    bitangent = _normalize(np.cross(axis, tangent))
    if bitangent is None:
        bitangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return tangent, bitangent


def build_visibility_cone_line_segments(
    model: Model,
    vertex_cones: list[npt.NDArray[np.float32]],
    *,
    cone_length: float = 0.0,
    rim_segments: int = 12,
) -> VisibilityConeLineSegments:
    if len(vertex_cones) != len(model.meshes):
        raise ValueError("vertex_cones must contain one array per mesh")
    rim_segments = max(3, int(rim_segments))
    scene_diagonal = max(float(np.linalg.norm(model.bounds_max - model.bounds_min)), 1e-6)
    resolved_length = float(cone_length)
    if not np.isfinite(resolved_length) or resolved_length <= 0.0:
        resolved_length = scene_diagonal * 0.008

    starts = []
    ends = []
    colors = []
    alphas = []
    widths = []
    cone_count = 0
    invalid_direction_count = 0
    axis_color = np.array([1.0, 0.78, 0.08], dtype=np.float32)
    low_scale_color = np.array([1.0, 0.16, 0.06], dtype=np.float32)
    high_scale_color = np.array([0.04, 0.88, 1.0], dtype=np.float32)
    invalid_color = np.array([1.0, 0.08, 0.58], dtype=np.float32)

    def append_segment(start, end, color, alpha, width):
        starts.append(np.asarray(start, dtype=np.float32))
        ends.append(np.asarray(end, dtype=np.float32))
        colors.append(np.asarray(color, dtype=np.float32))
        alphas.append(float(alpha))
        widths.append(int(width))

    for mesh, cones_value in zip(model.meshes, vertex_cones):
        cones = np.asarray(cones_value, dtype=np.float32)
        if cones.shape != (mesh.positions.shape[0], 5):
            raise ValueError(f"vertex cones for mesh {mesh.name!r} must have shape ({mesh.positions.shape[0]}, 5)")
        for position, cone in zip(mesh.positions, cones):
            cone_count += 1
            axis = _normalize(cone[:3])
            if axis is None:
                invalid_direction_count += 1
                marker_length = resolved_length * 0.14
                for basis in np.eye(3, dtype=np.float32):
                    append_segment(
                        position - basis * marker_length,
                        position + basis * marker_length,
                        invalid_color,
                        0.95,
                        2,
                    )
                continue

            aperture = float(np.clip(cone[3], 0.0, np.pi)) if np.isfinite(cone[3]) else 0.0
            scale = max(0.0, float(cone[4])) if np.isfinite(cone[4]) else 0.0
            display_scale = scale / (1.0 + scale)
            boundary_color = low_scale_color * (1.0 - display_scale) + high_scale_color * display_scale
            append_segment(position, position + axis * resolved_length, axis_color, 0.95, 2)

            tangent, bitangent = _cone_frame(axis)
            cosine = np.float32(np.cos(aperture))
            sine = np.float32(np.sin(aperture))
            rim_points = []
            generator_indices = {
                int(round(generator * rim_segments / 4.0)) % rim_segments
                for generator in range(4)
            }
            for segment in range(rim_segments):
                angle = (2.0 * np.pi * segment) / rim_segments
                radial = tangent * np.float32(np.cos(angle)) + bitangent * np.float32(np.sin(angle))
                boundary_direction = axis * cosine + radial * sine
                rim_point = position + boundary_direction * resolved_length
                rim_points.append(rim_point.astype(np.float32))
                if segment in generator_indices:
                    append_segment(position, rim_point, boundary_color, 0.58, 1)
            for segment in range(rim_segments):
                append_segment(
                    rim_points[segment],
                    rim_points[(segment + 1) % rim_segments],
                    boundary_color,
                    0.82,
                    1,
                )

    segment_count = len(starts)
    if segment_count == 0:
        return VisibilityConeLineSegments(
            starts=np.zeros((0, 3), dtype=np.float32),
            ends=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            alphas=np.zeros((0,), dtype=np.float32),
            widths=np.zeros((0,), dtype=np.int32),
            cone_count=cone_count,
            invalid_direction_count=invalid_direction_count,
            cone_length=resolved_length,
        )
    return VisibilityConeLineSegments(
        starts=np.ascontiguousarray(starts, dtype=np.float32),
        ends=np.ascontiguousarray(ends, dtype=np.float32),
        colors=np.ascontiguousarray(colors, dtype=np.float32),
        alphas=np.ascontiguousarray(alphas, dtype=np.float32),
        widths=np.ascontiguousarray(widths, dtype=np.int32),
        cone_count=cone_count,
        invalid_direction_count=invalid_direction_count,
        cone_length=resolved_length,
    )
