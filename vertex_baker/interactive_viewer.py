from __future__ import annotations

import json
import html
from pathlib import Path

import numpy as np
import numpy.typing as npt

from model import Model
from visibility_cone_visualizer import VisibilityConeLineSegments, build_visibility_cone_line_segments


def _flat_list(array: npt.NDArray) -> list:
    return np.asarray(array).reshape(-1).tolist()


def export_vertex_color_viewer(
    model: Model,
    vertex_colors: list[npt.NDArray[np.float32]],
    output_path: str | Path,
) -> None:
    _export_viewer(
        model,
        vertex_colors,
        output_path,
        title="Vertex Color Viewer",
        cone_lines=None,
        cone_instances=None,
    )


def export_visibility_cone_viewer(
    model: Model,
    vertex_colors: list[npt.NDArray[np.float32]],
    vertex_cones: list[npt.NDArray[np.float32]],
    output_path: str | Path,
    *,
    cone_length: float = 0.0,
    rim_segments: int = 12,
) -> None:
    cone_lines = build_visibility_cone_line_segments(
        model,
        vertex_cones,
        cone_length=cone_length,
        rim_segments=rim_segments,
    )
    cone_instances = _cone_instance_payload(
        model,
        vertex_cones,
        cone_length=cone_lines.cone_length,
        rim_segments=rim_segments,
    )
    _export_viewer(
        model,
        vertex_colors,
        output_path,
        title="Vertex Visibility Cones",
        cone_lines=cone_lines,
        cone_instances=cone_instances,
    )


def _cone_line_payload(cone_lines: VisibilityConeLineSegments | None) -> dict[str, list]:
    if cone_lines is None or cone_lines.starts.shape[0] == 0:
        return {"positions": [], "colors": []}
    positions = np.empty((cone_lines.starts.shape[0] * 2, 3), dtype=np.float32)
    colors = np.empty_like(positions)
    positions[0::2] = cone_lines.starts
    positions[1::2] = cone_lines.ends
    colors[0::2] = cone_lines.colors
    colors[1::2] = cone_lines.colors
    return {"positions": _flat_list(positions), "colors": _flat_list(colors)}


def _cone_instance_payload(
    model: Model,
    vertex_cones: list[npt.NDArray[np.float32]],
    *,
    cone_length: float,
    rim_segments: int,
) -> dict:
    if len(vertex_cones) != len(model.meshes):
        raise ValueError("vertex_cones must contain one array per mesh")

    positions = []
    axes = []
    parameters = []
    mesh_indices = []
    vertex_indices = []
    for mesh_index, (mesh, cones_value) in enumerate(zip(model.meshes, vertex_cones)):
        cones = np.asarray(cones_value, dtype=np.float32)
        expected_shape = (mesh.positions.shape[0], 5)
        if cones.shape != expected_shape:
            raise ValueError(f"vertex cones for mesh {mesh.name!r} must have shape {expected_shape}")
        for vertex_index, (position, cone) in enumerate(zip(mesh.positions, cones)):
            axis_length = float(np.linalg.norm(cone[:3]))
            if not np.isfinite(axis_length) or axis_length <= 1e-8:
                continue
            aperture = float(cone[3]) if np.isfinite(cone[3]) else 0.0
            scale = float(cone[4]) if np.isfinite(cone[4]) else 0.0
            positions.append(np.asarray(position, dtype=np.float32))
            axes.append(np.asarray(cone[:3] / axis_length, dtype=np.float32))
            parameters.append(np.array([np.clip(aperture, 0.0, np.pi), scale], dtype=np.float32))
            mesh_indices.append(mesh_index)
            vertex_indices.append(vertex_index)

    if positions:
        flat_positions = _flat_list(np.asarray(positions, dtype=np.float32))
        flat_axes = _flat_list(np.asarray(axes, dtype=np.float32))
        flat_parameters = _flat_list(np.asarray(parameters, dtype=np.float32))
    else:
        flat_positions = []
        flat_axes = []
        flat_parameters = []
    return {
        "positions": flat_positions,
        "axes": flat_axes,
        "parameters": flat_parameters,
        "meshIndices": mesh_indices,
        "vertexIndices": vertex_indices,
        "meshNames": [mesh.name for mesh in model.meshes],
        "length": float(cone_length),
        "segments": max(3, int(rim_segments)),
    }


def _export_viewer(
    model: Model,
    vertex_colors: list[npt.NDArray[np.float32]],
    output_path: str | Path,
    *,
    title: str,
    cone_lines: VisibilityConeLineSegments | None,
    cone_instances: dict | None,
) -> None:
    meshes = []
    for mesh_index, mesh in enumerate(model.meshes):
        colors = vertex_colors[mesh_index]
        if colors.shape[1] == 1:
            colors = np.repeat(colors, 3, axis=1)
        meshes.append(
            {
                "name": mesh.name,
                "positions": _flat_list(mesh.positions.astype(np.float32)),
                "colors": _flat_list(np.clip(colors[:, :3], 0.0, 1.0).astype(np.float32)),
                "indices": _flat_list(mesh.indices.astype(np.uint32)),
            }
        )

    center = ((model.bounds_min + model.bounds_max) * 0.5).astype(np.float32)
    extent = (model.bounds_max - model.bounds_min).astype(np.float32)
    radius = max(float(np.linalg.norm(extent) * 0.5), 1.0)
    html = _HTML_TEMPLATE.replace("__MESHES__", json.dumps(meshes, separators=(",", ":")))
    html = html.replace("__CENTER__", json.dumps(center.tolist(), separators=(",", ":")))
    html = html.replace("__RADIUS__", repr(radius))
    html = html.replace("__CONE_LINES__", json.dumps(_cone_line_payload(cone_lines), separators=(",", ":")))
    html = html.replace(
        "__CONE_INSTANCES__",
        json.dumps(cone_instances or _empty_cone_instance_payload(), separators=(",", ":")),
    )
    html = html.replace("__VIEWER_TITLE__", html_module_escape(title))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def html_module_escape(value: str) -> str:
    return html.escape(value, quote=True)


def _empty_cone_instance_payload() -> dict:
    return {
        "positions": [],
        "axes": [],
        "parameters": [],
        "meshIndices": [],
        "vertexIndices": [],
        "meshNames": [],
        "length": 0.0,
        "segments": 12,
    }


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__VIEWER_TITLE__</title>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #0d0f10; color: #e8e8e8; font-family: system-ui, sans-serif; }
#view { display: block; width: 100vw; height: 100vh; cursor: crosshair; }
.hud { position: fixed; left: 12px; top: 12px; width: 300px; max-height: calc(100vh - 24px); overflow-x: hidden; overflow-y: auto; box-sizing: border-box; background: rgba(14, 16, 18, 0.9); border: 1px solid rgba(255,255,255,0.18); padding: 10px 12px; border-radius: 6px; font-size: 13px; line-height: 1.45; user-select: none; }
.hud label { display: block; margin-top: 6px; }
.hud input { vertical-align: middle; }
.cone-selection { margin-top: 9px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.14); color: #d8dde0; font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.cone-selection:empty { display: none; }
.cone-diagram { display: none; width: 100%; height: auto; margin: 8px 0 2px; border: 1px solid rgba(255,255,255,0.14); background: #111517; box-sizing: border-box; }
.cone-diagram.visible { display: block; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div class="hud">
  <div><b>__VIEWER_TITLE__</b></div>
  <label><input id="wire" type="checkbox"> Wire overlay</label>
  <label><input id="backface" type="checkbox" checked> Show backfaces</label>
  <div id="cone-controls">
    <label><input id="cone-all" type="checkbox"> Show all vertices</label>
    <label><input id="cone-surface" type="checkbox" checked> Cone walls</label>
    <label><input id="cone-cap" type="checkbox"> Ray-distance cap</label>
    <label><input id="cone-wire" type="checkbox" checked> Cone outline</label>
    <label><input id="cone-xray" type="checkbox" checked> Cone X-ray</label>
    <div id="cone-selection" class="cone-selection"></div>
    <canvas id="cone-diagram" class="cone-diagram" width="500" height="380"></canvas>
  </div>
  <button id="reset">Reset</button>
</div>
<script>
"use strict";
const meshData = __MESHES__;
const coneLineData = __CONE_LINES__;
const coneInstanceData = __CONE_INSTANCES__;
const modelCenter = __CENTER__;
const modelRadius = __RADIUS__;
const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl2", { antialias: true });
if (!gl) {
  document.body.innerHTML = "<p style='padding:20px'>WebGL2 is not available in this browser.</p>";
  throw new Error("WebGL2 unavailable");
}

function compile(type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const prog = program(`#version 300 es
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_col;
uniform mat4 u_mvp;
out vec3 v_col;
void main() {
  v_col = a_col;
  gl_Position = u_mvp * vec4(a_pos, 1.0);
}`, `#version 300 es
precision highp float;
in vec3 v_col;
uniform float u_alpha;
out vec4 outColor;
void main() {
  outColor = vec4(clamp(v_col, 0.0, 1.0), u_alpha);
}`);
const wireProg = program(`#version 300 es
layout(location=0) in vec3 a_pos;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
}`, `#version 300 es
precision highp float;
out vec4 outColor;
void main() {
  outColor = vec4(0.02, 0.02, 0.02, 0.72);
}`);
const coneProg = program(`#version 300 es
layout(location=0) in vec3 a_shape;
layout(location=1) in vec3 a_origin;
layout(location=2) in vec3 a_axis;
layout(location=3) in vec2 a_params;
uniform mat4 u_mvp;
uniform float u_length;
uniform int u_selected;
uniform int u_show_all;
out vec3 v_world;
out vec3 v_normal;
out float v_scale;
flat out float v_surface;
flat out float v_selected;
const float PI = 3.141592653589793;
void main() {
  vec3 axis = normalize(a_axis);
  vec3 helper = abs(axis.z) > 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 tangent = normalize(cross(helper, axis));
  vec3 bitangent = normalize(cross(axis, tangent));
  vec3 radial = tangent * cos(a_shape.z) + bitangent * sin(a_shape.z);
  float aperture = clamp(a_params.x, 0.0, PI);
  bool cap = a_shape.x < 0.5;
  float theta = cap ? aperture * a_shape.y : aperture;
  float rayLength = cap ? u_length : u_length * a_shape.y;
  vec3 direction = axis * cos(theta) + radial * sin(theta);
  v_world = a_origin + direction * rayLength;
  v_normal = cap
    ? direction
    : normalize(radial * cos(aperture) - axis * sin(aperture));
  v_scale = a_params.y;
  v_surface = cap ? 0.0 : 1.0;
  v_selected = gl_InstanceID == u_selected ? 1.0 : 0.0;
  gl_Position = (u_show_all != 0 || gl_InstanceID == u_selected)
    ? u_mvp * vec4(v_world, 1.0)
    : vec4(2.0, 2.0, 2.0, 1.0);
}`, `#version 300 es
precision highp float;
in vec3 v_world;
in vec3 v_normal;
in float v_scale;
flat in float v_surface;
flat in float v_selected;
uniform vec3 u_eye;
out vec4 outColor;
void main() {
  float positiveScale = max(v_scale, 0.0);
  float mappedScale = positiveScale / (1.0 + positiveScale);
  vec3 color = mix(vec3(1.0, 0.16, 0.06), vec3(0.04, 0.88, 1.0), mappedScale);
  vec3 viewDirection = normalize(u_eye - v_world);
  float facing = abs(dot(normalize(v_normal), viewDirection));
  float edge = pow(1.0 - facing, 2.5);
  color *= 0.55 + 0.35 * facing;
  color += vec3(0.16) * edge;
  if (v_surface > 0.5) color *= 0.86;
  float alpha = v_surface > 0.5 ? 0.22 : 0.08;
  if (v_selected > 0.5) {
    color = mix(color, vec3(1.0, 0.82, 0.12), 0.78);
    alpha = 0.46;
  }
  outColor = vec4(clamp(color, 0.0, 1.0), alpha);
}`);

function makeMesh(data) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const pos = new Float32Array(data.positions);
  const col = new Float32Array(data.colors);
  const idx = new Uint32Array(data.indices);
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const colBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

  const lineIdx = [];
  for (let i = 0; i < idx.length; i += 3) {
    lineIdx.push(idx[i], idx[i+1], idx[i+1], idx[i+2], idx[i+2], idx[i]);
  }
  const lineBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(lineIdx), gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, idxBuf, indexCount: idx.length, lineBuf, lineCount: lineIdx.length };
}
const meshes = meshData.map(makeMesh);

function makeConeLines(data) {
  const positions = new Float32Array(data.positions);
  const colors = new Float32Array(data.colors);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const colorBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuf);
  gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, count: positions.length / 3 };
}
const coneLines = makeConeLines(coneLineData);

function makeDynamicConeLines() {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, 0, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const colorBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuf);
  gl.bufferData(gl.ARRAY_BUFFER, 0, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, posBuf, colorBuf, count: 0 };
}
const selectedConeLines = makeDynamicConeLines();

function makeConeSurface(data) {
  const segments = Math.max(3, data.segments | 0);
  const capRings = Math.max(3, Math.ceil(segments / 4));
  const wallShape = [];
  const capShape = [];
  function append(target, kind, fraction, phi) { target.push(kind, fraction, phi); }
  for (let segment = 0; segment < segments; ++segment) {
    const phi0 = 2 * Math.PI * segment / segments;
    const phi1 = 2 * Math.PI * (segment + 1) / segments;
    append(wallShape, 1, 0, phi0);
    append(wallShape, 1, 1, phi0);
    append(wallShape, 1, 1, phi1);
    append(capShape, 0, 0, phi0);
    append(capShape, 0, 1 / capRings, phi0);
    append(capShape, 0, 1 / capRings, phi1);
    for (let ring = 1; ring < capRings; ++ring) {
      const theta0 = ring / capRings;
      const theta1 = (ring + 1) / capRings;
      append(capShape, 0, theta0, phi0);
      append(capShape, 0, theta1, phi0);
      append(capShape, 0, theta1, phi1);
      append(capShape, 0, theta0, phi0);
      append(capShape, 0, theta1, phi1);
      append(capShape, 0, theta0, phi1);
    }
  }
  const shape = wallShape.concat(capShape);

  const origins = new Float32Array(data.positions);
  const axes = new Float32Array(data.axes);
  const parameters = new Float32Array(data.parameters);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const shapeBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, shapeBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(shape), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  function instanceAttribute(location, size, values) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(location, 1);
    return buffer;
  }
  const originBuf = instanceAttribute(1, 3, origins);
  const axisBuf = instanceAttribute(2, 3, axes);
  const parameterBuf = instanceAttribute(3, 2, parameters);
  gl.bindVertexArray(null);
  return {
    vao,
    wallVertexCount: wallShape.length / 3,
    capVertexCount: capShape.length / 3,
    instanceCount: origins.length / 3,
    origins,
    axes,
    parameters,
    meshIndices: data.meshIndices,
    vertexIndices: data.vertexIndices,
    meshNames: data.meshNames,
    length: Number(data.length),
    segments,
    buffers: [shapeBuf, originBuf, axisBuf, parameterBuf],
  };
}
const coneSurface = makeConeSurface(coneInstanceData);
if (coneSurface.instanceCount === 0 && coneLines.count === 0) {
  document.getElementById("cone-controls").style.display = "none";
}

function mat4Identity() { return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
function mat4Mul(a,b) {
  const o = new Array(16).fill(0);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) for (let k=0;k<4;k++) o[c*4+r] += a[k*4+r]*b[c*4+k];
  return o;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
}
function normalize(v) {
  const l = Math.hypot(v[0],v[1],v[2]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l];
}
function cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function lookAt(eye, center, up) {
  const z = normalize([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1];
}

function add3(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function scale3(value, scalar) { return [value[0]*scalar, value[1]*scalar, value[2]*scalar]; }
function updateSelectedConeLines(index) {
  const positions = [];
  const colors = [];
  function appendLine(start, end, color) {
    positions.push(...start, ...end);
    colors.push(...color, ...color);
  }
  if (index >= 0) {
    const offset = index * 3;
    const origin = [
      coneSurface.origins[offset],
      coneSurface.origins[offset + 1],
      coneSurface.origins[offset + 2],
    ];
    const axis = normalize([
      coneSurface.axes[offset],
      coneSurface.axes[offset + 1],
      coneSurface.axes[offset + 2],
    ]);
    const helper = Math.abs(axis[2]) > 0.9 ? [0, 1, 0] : [0, 0, 1];
    const tangent = normalize(cross(helper, axis));
    const bitangent = normalize(cross(axis, tangent));
    const aperture = Math.max(0, Math.min(Math.PI, coneSurface.parameters[index * 2]));
    const length = coneSurface.length;
    const axisColor = [1.0, 0.78, 0.08];
    const boundaryColor = [0.05, 0.9, 1.0];
    const apexColor = [1.0, 1.0, 1.0];
    const axisTip = add3(origin, scale3(axis, length));
    appendLine(origin, axisTip, axisColor);

    const arrowBase = add3(axisTip, scale3(axis, -length * 0.16));
    for (let arrow = 0; arrow < 4; ++arrow) {
      const angle = 2 * Math.PI * arrow / 4;
      const radial = add3(scale3(tangent, Math.cos(angle)), scale3(bitangent, Math.sin(angle)));
      appendLine(add3(arrowBase, scale3(radial, length * 0.065)), axisTip, axisColor);
    }
    appendLine(add3(origin, scale3(tangent, -length * 0.035)), add3(origin, scale3(tangent, length * 0.035)), apexColor);
    appendLine(add3(origin, scale3(bitangent, -length * 0.035)), add3(origin, scale3(bitangent, length * 0.035)), apexColor);

    const cosine = Math.cos(aperture);
    const sine = Math.sin(aperture);
    const rim = [];
    const generatorIndices = new Set();
    for (let generator = 0; generator < 4; ++generator) {
      generatorIndices.add(Math.round(generator * coneSurface.segments / 4) % coneSurface.segments);
    }
    for (let segment = 0; segment < coneSurface.segments; ++segment) {
      const angle = 2 * Math.PI * segment / coneSurface.segments;
      const radial = add3(scale3(tangent, Math.cos(angle)), scale3(bitangent, Math.sin(angle)));
      const boundary = add3(scale3(axis, cosine), scale3(radial, sine));
      const point = add3(origin, scale3(boundary, length));
      rim.push(point);
      if (generatorIndices.has(segment)) appendLine(origin, point, boundaryColor);
    }
    for (let segment = 0; segment < coneSurface.segments; ++segment) {
      appendLine(rim[segment], rim[(segment + 1) % coneSurface.segments], boundaryColor);
    }
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, selectedConeLines.posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, selectedConeLines.colorBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);
  selectedConeLines.count = positions.length / 3;
}

function drawConeDiagram(index) {
  const diagram = document.getElementById("cone-diagram");
  if (index < 0) {
    diagram.classList.remove("visible");
    return;
  }
  diagram.classList.add("visible");
  const context = diagram.getContext("2d");
  const width = diagram.width;
  const height = diagram.height;
  const centerX = width * 0.5;
  const centerY = height * 0.54;
  const radius = Math.min(width, height) * 0.38;
  const aperture = coneSurface.parameters[index * 2];
  const degrees = aperture * 180 / Math.PI;
  const axisAngle = -Math.PI * 0.5;
  const startAngle = axisAngle - aperture;
  const endAngle = axisAngle + aperture;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111517";
  context.fillRect(0, 0, width, height);

  context.beginPath();
  context.moveTo(centerX, centerY);
  context.arc(centerX, centerY, radius, startAngle, endAngle, false);
  context.closePath();
  context.fillStyle = "rgba(29, 211, 240, 0.2)";
  context.fill();

  context.save();
  context.setLineDash([7, 7]);
  context.strokeStyle = "rgba(220, 231, 235, 0.42)";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(centerX, centerY, radius, 0, Math.PI * 2);
  context.stroke();
  context.restore();

  function pointAt(angle, distance) {
    return [centerX + Math.cos(angle) * distance, centerY + Math.sin(angle) * distance];
  }
  const start = pointAt(startAngle, radius);
  const end = pointAt(endAngle, radius);
  context.strokeStyle = "#24dff5";
  context.lineWidth = 5;
  context.beginPath();
  context.moveTo(centerX, centerY);
  context.lineTo(start[0], start[1]);
  context.moveTo(centerX, centerY);
  context.lineTo(end[0], end[1]);
  context.stroke();

  const axisTip = pointAt(axisAngle, radius);
  context.strokeStyle = "#ffca22";
  context.fillStyle = "#ffca22";
  context.lineWidth = 5;
  context.beginPath();
  context.moveTo(centerX, centerY);
  context.lineTo(axisTip[0], axisTip[1]);
  context.stroke();
  context.beginPath();
  context.moveTo(axisTip[0], axisTip[1]);
  context.lineTo(axisTip[0] - 10, axisTip[1] + 18);
  context.lineTo(axisTip[0] + 10, axisTip[1] + 18);
  context.closePath();
  context.fill();

  const arcRadius = radius * 0.34;
  context.strokeStyle = "#ffffff";
  context.lineWidth = 3;
  context.beginPath();
  context.arc(centerX, centerY, arcRadius, startAngle, axisAngle, false);
  context.stroke();
  context.beginPath();
  context.arc(centerX, centerY, arcRadius, axisAngle, endAngle, false);
  context.stroke();

  context.fillStyle = "#ffffff";
  context.font = "600 28px system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`theta = ${degrees.toFixed(2)}\u00b0`, 18, 34);
  context.font = "22px ui-monospace, Consolas, monospace";
  context.fillStyle = "#ffca22";
  context.fillText("axis", centerX + 13, centerY - radius * 0.6);
  context.fillStyle = "#24dff5";
  context.fillText(`L = ${coneSurface.length.toFixed(3)}`, 18, height - 18);
  context.fillStyle = "#ffffff";
  context.beginPath();
  context.arc(centerX, centerY, 7, 0, Math.PI * 2);
  context.fill();
}

let selectedCone = -1;
let lastMvp = null;
function projectPoint(position, matrix) {
  const x = position[0], y = position[1], z = position[2];
  const clipX = matrix[0]*x + matrix[4]*y + matrix[8]*z + matrix[12];
  const clipY = matrix[1]*x + matrix[5]*y + matrix[9]*z + matrix[13];
  const clipZ = matrix[2]*x + matrix[6]*y + matrix[10]*z + matrix[14];
  const clipW = matrix[3]*x + matrix[7]*y + matrix[11]*z + matrix[15];
  if (clipW <= 0) return null;
  return [clipX / clipW, clipY / clipW, clipZ / clipW];
}
function showConeDetails(index) {
  const details = document.getElementById("cone-selection");
  if (index < 0) {
    details.textContent = "";
    return;
  }
  const parameterOffset = index * 2;
  const axisOffset = index * 3;
  const aperture = coneSurface.parameters[parameterOffset];
  const scale = coneSurface.parameters[parameterOffset + 1];
  const radius = coneSurface.length * Math.sin(aperture);
  const axialReach = coneSurface.length * Math.cos(aperture);
  const meshIndex = coneSurface.meshIndices[index];
  const meshName = coneSurface.meshNames[meshIndex] || `mesh ${meshIndex}`;
  details.textContent =
    `${meshName} | vertex ${coneSurface.vertexIndices[index]}\n` +
    `Aperture   ${(aperture * 180 / Math.PI).toFixed(4)}\u00b0\n` +
    `Scale      ${scale.toFixed(6)}\n` +
    `Axis       (${coneSurface.axes[axisOffset].toFixed(5)}, ` +
    `${coneSurface.axes[axisOffset + 1].toFixed(5)}, ${coneSurface.axes[axisOffset + 2].toFixed(5)})\n` +
    `Ray length ${coneSurface.length.toFixed(6)}\n` +
      `Rim radius ${radius.toFixed(6)}\n` +
      `Axis reach ${axialReach.toFixed(6)}`;
}
function setSelectedCone(index) {
  selectedCone = index;
  showConeDetails(index);
  updateSelectedConeLines(index);
  drawConeDiagram(index);
}
function selectCone(clientX, clientY, maximumDistance = 26) {
  if (!lastMvp || coneSurface.instanceCount === 0) return;
  const rect = canvas.getBoundingClientRect();
  let nearest = -1;
  let nearestDistanceSquared = maximumDistance * maximumDistance;
  let nearestDepth = Infinity;
  for (let index = 0; index < coneSurface.instanceCount; ++index) {
    const offset = index * 3;
    const projected = projectPoint(
      [coneSurface.origins[offset], coneSurface.origins[offset + 1], coneSurface.origins[offset + 2]],
      lastMvp,
    );
    if (!projected || projected[2] < -1 || projected[2] > 1) continue;
    const screenX = rect.left + (projected[0] * 0.5 + 0.5) * rect.width;
    const screenY = rect.top + (0.5 - projected[1] * 0.5) * rect.height;
    const dx = screenX - clientX;
    const dy = screenY - clientY;
    const distanceSquared = dx * dx + dy * dy;
    if (distanceSquared < nearestDistanceSquared ||
        (distanceSquared === nearestDistanceSquared && projected[2] < nearestDepth)) {
      nearest = index;
      nearestDistanceSquared = distanceSquared;
      nearestDepth = projected[2];
    }
  }
  setSelectedCone(nearest);
}
function selectReadableInitialCone() {
  if (!lastMvp || coneSurface.instanceCount === 0) return;
  const rect = canvas.getBoundingClientRect();
  let bestIndex = -1;
  let bestScore = Infinity;
  for (let index = 0; index < coneSurface.instanceCount; ++index) {
    const offset = index * 3;
    const origin = [
      coneSurface.origins[offset],
      coneSurface.origins[offset + 1],
      coneSurface.origins[offset + 2],
    ];
    const axisTip = [
      origin[0] + coneSurface.axes[offset] * coneSurface.length,
      origin[1] + coneSurface.axes[offset + 1] * coneSurface.length,
      origin[2] + coneSurface.axes[offset + 2] * coneSurface.length,
    ];
    const projectedOrigin = projectPoint(origin, lastMvp);
    const projectedTip = projectPoint(axisTip, lastMvp);
    if (!projectedOrigin || !projectedTip || projectedOrigin[2] < -1 || projectedOrigin[2] > 1) continue;
    if (Math.abs(projectedOrigin[0]) > 1 || Math.abs(projectedOrigin[1]) > 1) continue;
    const originX = (projectedOrigin[0] * 0.5 + 0.5) * rect.width;
    const originY = (0.5 - projectedOrigin[1] * 0.5) * rect.height;
    const tipX = (projectedTip[0] * 0.5 + 0.5) * rect.width;
    const tipY = (0.5 - projectedTip[1] * 0.5) * rect.height;
    const centerDistance = Math.hypot(originX - rect.width * 0.5, originY - rect.height * 0.5);
    const axisPixels = Math.hypot(tipX - originX, tipY - originY);
    const apertureDegrees = coneSurface.parameters[index * 2] * 180 / Math.PI;
    const score = centerDistance * 0.2 + Math.abs(apertureDegrees - 65) * 5 - axisPixels * 6;
    if (score < bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }
  setSelectedCone(bestIndex);
}

let yaw, pitch, distance, pan;
let initialSelectionDone = false;
function resetCamera() {
  yaw = 0.08;
  pitch = 0.02;
  distance = modelRadius * 4.6;
  pan = [0, 0, 0];
}
resetCamera();
document.getElementById("reset").onclick = resetCamera;

let dragging = false, button = 0, lastX = 0, lastY = 0;
let pointerStartX = 0, pointerStartY = 0, pointerMoved = false;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => {
  dragging = true;
  button = e.button;
  lastX = pointerStartX = e.clientX;
  lastY = pointerStartY = e.clientY;
  pointerMoved = false;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointerup", e => {
  const select = button === 0 && !pointerMoved;
  dragging = false;
  canvas.releasePointerCapture(e.pointerId);
  if (select) selectCone(e.clientX, e.clientY);
});
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  if (Math.hypot(e.clientX - pointerStartX, e.clientY - pointerStartY) > 3) pointerMoved = true;
  lastX = e.clientX; lastY = e.clientY;
  if (button === 0) {
    yaw += dx * 0.006;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.006));
  } else {
    const scale = distance * 0.0015;
    pan[0] -= dx * scale;
    pan[1] += dy * scale;
  }
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  distance *= Math.exp(e.deltaY * 0.001);
  distance = Math.max(modelRadius * 0.25, Math.min(modelRadius * 20, distance));
}, { passive: false });

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
}
function render() {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.06, 0.065, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  if (document.getElementById("backface").checked) gl.disable(gl.CULL_FACE);
  else { gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK); }

  const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
  const center = [modelCenter[0] + pan[0], modelCenter[1] + pan[1], modelCenter[2] + pan[2]];
  const eye = [center[0] + distance * sy * cp, center[1] + distance * sp, center[2] + distance * cy * cp];
  const view = lookAt(eye, center, [0, 1, 0]);
  const proj = perspective(45 * Math.PI / 180, canvas.width / canvas.height, modelRadius * 0.001, modelRadius * 100);
  const mvp = new Float32Array(mat4Mul(proj, view));
  lastMvp = mvp;
  if (!initialSelectionDone && coneSurface.instanceCount > 0) {
    selectReadableInitialCone();
    initialSelectionDone = true;
  }
  const showAllCones = document.getElementById("cone-all").checked;

  gl.useProgram(prog);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_mvp"), false, mvp);
  gl.uniform1f(gl.getUniformLocation(prog, "u_alpha"), 1.0);
  for (const mesh of meshes) {
    gl.bindVertexArray(mesh.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idxBuf);
    gl.drawElements(gl.TRIANGLES, mesh.indexCount, gl.UNSIGNED_INT, 0);
  }

  if (coneSurface.instanceCount > 0 && document.getElementById("cone-surface").checked) {
    if (document.getElementById("cone-xray").checked) gl.disable(gl.DEPTH_TEST);
    else { gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); }
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.useProgram(coneProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(coneProg, "u_mvp"), false, mvp);
    gl.uniform1f(gl.getUniformLocation(coneProg, "u_length"), coneSurface.length);
    gl.uniform1i(gl.getUniformLocation(coneProg, "u_selected"), selectedCone);
    gl.uniform1i(gl.getUniformLocation(coneProg, "u_show_all"), showAllCones ? 1 : 0);
    gl.uniform3fv(gl.getUniformLocation(coneProg, "u_eye"), eye);
    gl.bindVertexArray(coneSurface.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, coneSurface.wallVertexCount, coneSurface.instanceCount);
    if (document.getElementById("cone-cap").checked) {
      gl.drawArraysInstanced(
        gl.TRIANGLES,
        coneSurface.wallVertexCount,
        coneSurface.capVertexCount,
        coneSurface.instanceCount,
      );
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
  }

  const activeConeLines = showAllCones ? coneLines : selectedConeLines;
  if (activeConeLines.count > 0 && document.getElementById("cone-wire").checked) {
    if (document.getElementById("cone-xray").checked) gl.disable(gl.DEPTH_TEST);
    else { gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); }
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_mvp"), false, mvp);
    gl.uniform1f(gl.getUniformLocation(prog, "u_alpha"), 0.82);
    gl.bindVertexArray(activeConeLines.vao);
    gl.drawArrays(gl.LINES, 0, activeConeLines.count);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
  }

  if (document.getElementById("wire").checked) {
    gl.useProgram(wireProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(wireProg, "u_mvp"), false, mvp);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    for (const mesh of meshes) {
      gl.bindVertexArray(mesh.vao);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.lineBuf);
      gl.drawElements(gl.LINES, mesh.lineCount, gl.UNSIGNED_INT, 0);
    }
    gl.disable(gl.POLYGON_OFFSET_FILL);
  }

  gl.bindVertexArray(null);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
</script>
</body>
</html>
"""
