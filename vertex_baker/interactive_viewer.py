from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from model import Model


def _flat_list(array: npt.NDArray) -> list:
    return np.asarray(array).reshape(-1).tolist()


def export_vertex_color_viewer(
    model: Model,
    vertex_colors: list[npt.NDArray[np.float32]],
    output_path: str | Path,
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vertex Color Viewer</title>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #0d0f10; color: #e8e8e8; font-family: system-ui, sans-serif; }
canvas { display: block; width: 100vw; height: 100vh; }
.hud { position: fixed; left: 12px; top: 12px; background: rgba(14, 16, 18, 0.82); border: 1px solid rgba(255,255,255,0.18); padding: 10px 12px; border-radius: 6px; font-size: 13px; line-height: 1.45; user-select: none; }
.hud label { display: block; margin-top: 6px; }
.hud input { vertical-align: middle; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div class="hud">
  <div><b>Vertex Color Viewer</b></div>
  <div>Left drag: orbit</div>
  <div>Right/Middle drag: pan</div>
  <div>Wheel: zoom</div>
  <label><input id="wire" type="checkbox"> Wire overlay</label>
  <label><input id="backface" type="checkbox" checked> Show backfaces</label>
  <button id="reset">Reset</button>
</div>
<script>
"use strict";
const meshData = __MESHES__;
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
out vec4 outColor;
void main() {
  outColor = vec4(clamp(v_col, 0.0, 1.0), 1.0);
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

let yaw, pitch, distance, pan;
function resetCamera() {
  yaw = 0.08;
  pitch = 0.02;
  distance = modelRadius * 4.6;
  pan = [0, 0, 0];
}
resetCamera();
document.getElementById("reset").onclick = resetCamera;

let dragging = false, button = 0, lastX = 0, lastY = 0;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => {
  dragging = true; button = e.button; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointerup", e => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
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

  gl.useProgram(prog);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_mvp"), false, mvp);
  for (const mesh of meshes) {
    gl.bindVertexArray(mesh.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idxBuf);
    gl.drawElements(gl.TRIANGLES, mesh.indexCount, gl.UNSIGNED_INT, 0);
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
