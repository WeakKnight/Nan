# Agent Guide

## Purpose & Scope
- Real-time GPU path tracer, authored in Python + Slang via SlangPy.
- Targets DXR/Vulkan-RT capable GPUs; renders interactively with temporal accumulation and tone mapping.
- This guide gives AI agents enough context to reason about the codebase without re-reading everything.

---

## Runtime Flow
1. `entry_point.py` creates a `PathTracingRenderer` or `TextureSpacePathTracingRenderer` from `--renderer`.
2. `App` (window, device, swapchain, UI) loads an asset scene through `SceneNode` → `Scene`.
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
| `TextureSpacePathTracingRenderer` | `TextureSpacePathTracer` → `MeshColorsResolve` → `Accumulator` → `ToneMapper` | View-independent irradiance accumulation on per-triangle barycentric grids; back slots are allocated only for double-sided materials. |

---

## Host Modules

### App & Control
- `app.py` – Window/device bootstrap, input routing, per-frame loop, RenderDoc/TEV integration, screenshot shortcuts (`F1` / `F2` / `F11` / `Esc`). Owns a `RenderData` cache and passes it to renderers each frame.
- `renderer.py` – Protocol every renderer follows (`initialize`, `render`, `setup_ui`). `render` receives both `scene` and `render_data`.
- `camera.py` – Camera state + jitter + WASD/mouse controller.
- `event_dispatcher` (external dependency) – Pub/sub used for camera/key notifications.

### Scene & Assets
- `scene_node.py` – High-level scene graph, asset loading, axis conversion (`"z_up_to_y_up"`), material extraction fallback chain.
- `scene.py` – GPU-ready scene (descriptors, buffers, BLAS/TLAS builds, env map).
- `Scene.md` – Supplemental deep dive into scene packing (reference when editing descriptors).
- `mesh.py`, `material.py`, `transform.py` – Data containers for scene construction.
- `mesh_colors.py` – CPU Mesh Colors layout, per-face/instance offsets, compact material-driven side slots, resolution clamps, and payload-slot budget enforcement.
- `utils.py` – HDR EXR helpers.
- `vertex_baker/slang_viewer.py` – Native SlangPy baker viewer with HWRT or raster GBuffer, orbit camera, GPU picking, and visibility-cone diagnostics.

### Rendering Pass Wrappers
- `path_tracer.py` – Thin wrapper around path tracing compute shader; binds the `Scene`.
- `texture_space_path_tracer.py` – Dispatches progressive irradiance tracing over Mesh Colors texels.
- `mesh_colors_resolve.py` – Resolves the view-independent Mesh Colors cache through current-camera primary hits.
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
| `mesh_colors.slang` | Barycentric lattice addressing, metadata lookup, and irradiance interpolation. | Texture-space passes. |
| `texture_space_path_tracer.slang` | Progressive per-texel irradiance path tracing with Nan sky/sun/static-shadow lighting. | `TextureSpacePathTracer`. |
| `mesh_colors_resolve.slang` | Primary-ray lookup of cached irradiance for the current camera. | `MeshColorsResolve`. |
| `accumulator.slang` | Temporal accumulation kernel. | Renderer. |
| `tone_mapper.slang` | ACES-like filmic operator. | Final pass. |
| `atmosphere.slang` | LUT generation utilities (sky/atmosphere research experiments). | Used by LUT scripts. |
| `vertex_baker/slang_viewer.slang` | Inline-ray-query or raster material/cone GBuffer, PMR diffuse/specular occlusion, PBR debug composite, picking, and selected-cone overlay. | Native vertex baker viewer. |
| `vertex_baker/ibl_precompute.slang` | Runtime HDR cubemap conversion, SH9 projection, GGX prefilter, DFG integration, and PMR cone-BRDF specular-occlusion LUT generation. | Native vertex baker viewer. |

Include new shaders via `device.load_program(..., ["entry_point"])`; ensure host struct packing matches shader expectations.

---

## Scene & Asset Workflow
- Axis conversion defaults to `"none"` but `"z_up_to_y_up"` handles common DCC exports.
- Asset loader supports OBJ/STL/PLY/glTF/OFF/3MF/Collada. Materials default to extracted diffuse → vertex color → provided override → neutral gray; glTF `doubleSided` is preserved while other sources default to single-sided.
- Camera metadata from JSON seeds `CameraController`. If not provided, `SceneNode` creates a default camera aimed at the scene bounds.
- Instancing: `SceneNode.add_scene_node` reuses meshes/materials; `Scene` flags odd-negative transforms to maintain correct winding.

---

## UI, Input & Events
- Camera: WASD + mouse look; `CameraController.update` publishes `"camera_move"` when position/orientation changes.
- Hotkeys: `Esc` quit, `F1` TEV viewer, `F2` screenshot, `F11` RenderDoc capture toggle.
- Renderers define UI controls in `setup_ui`; texture-space mode adds screen accumulation, pause/reset, exposure, and cache status.
- Event dispatcher routes keyboard presses to renderers; add new listeners via `scene.event_distpacher.subscribe`.

---

## Tooling & Debugging
- RenderDoc integration (auto-detected via `spy.renderdoc.is_available()`); call `spy.renderdoc.start_frame_capture` when `self.should_capture` is set.
- TEV viewer: `spy.tev.show_async` for quick HDR inspection.
- Shader debugging: drop colored diagnostics into RW textures; flush prints with `device.flush_print()` (first frame already does this).
- `test_print.py` + `test_print.slang` demonstrate GPU-side `print` debugging; run the script to verify Slang `print` output and use it as a template for logging thread-local data.
- GPU crashes: double-check `device` compiler options (`include_paths`, defines) in `app.py`.

---

## Documentation
Reference materials in `docs/`:
| File | Content |
|------|---------|
| `slang_guide.md` | Slang language overview and syntax. |
| `slangpy_guide.md` | SlangPy Python bindings usage. |
| `slangpy_bindless.md` | Bindless resource patterns in SlangPy. |
| `slangpy_texture_loader.md` | Texture loading utilities. |

---

## Testing & Verification
- Tests assume a GPU-capable Python runtime; they instantiate Slang devices—avoid running on headless CI without RT hardware.
- `test_mesh_colors.py` validates the CPU-only barycentric layout and budget guard.
- Texture-space GPU smoke: `python entry_point.py --renderer texture-space --headless --frames 16 --width 512 --height 512`.

---

## Maintenance Checklist
- When adding/modifying passes or shaders, update both host wrapper and shader tables above.
- Smoke-test changes headlessly (`python entry_point.py --headless --frames 16`) to catch runtime issues without the UI loop.
- Leverage GPU-side `print` debugging (see `test_print.py` / `test_print.slang`) to trace per-thread state and flush with `device.flush_print()`.
