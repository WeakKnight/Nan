# Path Tracer – Agent Guide

## Purpose & Scope
- Real-time GPU path tracer, authored in Python + Slang via SlangPy.
- Targets DXR/Vulkan-RT capable GPUs; renders interactively with temporal accumulation and tone mapping.
- This guide gives AI agents enough context to reason about the codebase without re-reading everything.

---

## Runtime Flow
1. `entry_point.py` creates a `PathTracingRenderer`.
2. `App` (window, device, swapchain, UI) loads a JSON/asset scene through `SceneNode` → `Scene`.
3. Camera/controller feeds jittered view data; event dispatcher announces camera/key changes.
4. Renderer orchestrates compute passes (path tracing, accumulation, tonemap) on `spy.CommandEncoder`.
5. Output textures blit to the surface; UI draws; optional RenderDoc/TEV capture hooks run.

---

## Headless Mode
- `entry_point.py --headless` runs without a window or swapchain.
- Control accumulation with `--frames N`, raster size with `--width W --height H`, and output path via `--output path/to/image.png`.
- sRGB tonemapping is enabled by default; add `--no-srgb` to preserve linear data in the saved bitmap.

---

## Renderer Pipeline
| Renderer | Pass Chain | Highlights |
|----------|------------|------------|
| `PathTracingRenderer` | `PathTracer` → `Accumulator` → `ToneMapper` | Standard path tracing with cosine BSDF, up to 5 bounces. |

---

## Host Modules

### App & Control
- `app.py` – Window/device bootstrap, input routing, per-frame loop, RenderDoc/TEV integration, screenshot shortcuts (`F1` / `F2` / `F11` / `Esc`). Owns a `RenderData` cache and passes it to renderers each frame.
- `renderer.py` – Protocol every renderer follows (`initialize`, `render`, `setup_ui`). `render` receives both `scene` and `render_data`.
- `camera.py` – Camera state + jitter + WASD/mouse controller.
- `event_dispatcher` (external dependency) – Pub/sub used for camera/key notifications.

### Scene & Assets
- `scene_node.py` – High-level scene graph, asset + JSON loading, axis conversion (`"z_up_to_y_up"`), material extraction fallback chain.
- `scene.py` – GPU-ready scene (descriptors, buffers, BLAS/TLAS builds, env map).
- `Scene.md` – Supplemental deep dive into scene packing (reference when editing descriptors).
- `mesh.py`, `material.py`, `transform.py` – Data containers for scene construction.
- `utils.py` – HDR EXR helpers.

### Rendering Pass Wrappers
- `path_tracer.py` – Thin wrapper around path tracing compute shader; binds the `Scene`.
- `accumulator.py` – Float accumulation with reset flag; history texture fetched via `RenderData`.
- `tone_mapper.py` – ACES filmic tonemap pass; toggled via UI checkbox in renderers.
- `ping_pong_texture.py` – Double-buffer helper backed by `RenderData` texture cache.
- `low_discrepancy_disk_pattern.py` – Hex-disk samples for spatial sampling patterns.

---

## Shader Modules
| File | Purpose | Hosts |
|------|---------|-------|
| `common.slang` | RNG, ray structs, sampling helpers, math utilities. | Used everywhere. |
| `scene.slang` | GPU-side scene accessors (vertex fetch, env sampling, ray queries). | `path_tracer`. |
| `camera.slang` | Camera matrices, jitter, ray generation. | All rendering passes. |
| `path_tracer.slang` | GI path tracer, cosine BSDF, up to 5 bounces. | `PathTracer`. |
| `accumulator.slang` | Temporal accumulation kernel. | Renderer. |
| `tone_mapper.slang` | ACES-like filmic operator. | Final pass. |
| `atmosphere.slang` | LUT generation utilities (sky/atmosphere research experiments). | Used by LUT scripts. |

Include new shaders via `device.load_program(..., ["entry_point"])`; ensure host struct packing matches shader expectations.

---

## Scene & Asset Workflow
- Preferred entry path: `SceneNode.load_json("examples/...")`; axis conversion defaults to `"none"` but `"z_up_to_y_up"` handles common DCC exports.
- Asset loader supports OBJ/STL/PLY/glTF/OFF/3MF/Collada. Materials default to extracted diffuse → vertex color → provided override → neutral gray.
- Camera metadata from JSON seeds `CameraController`. If not provided, `SceneNode` creates a default camera aimed at the scene bounds.
- Instancing: `SceneNode.add_scene_node` reuses meshes/materials; `Scene` flags odd-negative transforms to maintain correct winding.

---

## Resources & LUTs
- HDR assets: `plains_sunset_2k.exr` (default env), `envmap.exr` (legacy fallback).
- Atmospheric LUTs: `sky_view_lut.png`, `transmittance_lut.png`, `multiscatt_lut.png`; produced by `atmosphere.py` experiments.
- Example scenes & meshes live under `examples/`; structured by scenario with reusable object libraries.

---

## UI, Input & Events
- Camera: WASD + mouse look; `CameraController.update` publishes `"camera_move"` when position/orientation changes.
- Hotkeys: `Esc` quit, `F1` TEV viewer, `F2` screenshot, `F11` RenderDoc capture toggle.
- Renderers define UI controls in `setup_ui`; current toggles: "Use Accum".
- Event dispatcher routes keyboard presses to renderers; add new listeners via `scene.event_distpacher.subscribe`.

---

## Tooling & Debugging
- RenderDoc integration (auto-detected via `spy.renderdoc.is_available()`); call `spy.renderdoc.start_frame_capture` when `self.should_capture` is set.
- TEV viewer: `spy.tev.show_async` for quick HDR inspection.
- Shader debugging: drop colored diagnostics into RW textures; flush prints with `device.flush_print()` (first frame already does this).
- `test_print.py` + `test_print.slang` demonstrate GPU-side `print` debugging; run the script to verify Slang `print` output and use it as a template for logging thread-local data.
- GPU crashes: double-check `device` compiler options (`include_paths`, defines) in `app.py`.

---

## Testing & Verification
- Tests assume a GPU-capable Python runtime; they instantiate Slang devices—avoid running on headless CI without RT hardware.

---

## Maintenance Checklist
- When adding/modifying passes or shaders, update both host wrapper and shader tables above.
- Smoke-test changes headlessly (`python entry_point.py --headless --frames 16`) to catch runtime issues without the UI loop.
- Leverage GPU-side `print` debugging (see `test_print.py` / `test_print.slang`) to trace per-thread state and flush with `device.flush_print()`.
