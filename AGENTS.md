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
| `TextureSpacePathTracingRenderer` | `TextureSpacePathTracer` → `MeshColorsResolve` → `Accumulator` → `ToneMapper`; Save: `MeshColorsSurfaceFilter` → RGB9E5 | View-independent irradiance accumulation on per-triangle barycentric grids; back slots are allocated only for double-sided materials. |
| `SurfaceProbePathTracingRenderer` | Adaptive WSE → budgeted deficit repair → protected zero-count closure; `SurfaceProbePathTracer` → `SurfaceProbeResolve` → optional `Accumulator` → `ToneMapper` | Sparse surface irradiance cache with point-octree reconstruction. Protected closure probes bypass WSE and share the normal reconstruction estimator; legacy vertex fallback is opt-in. |

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
- `mesh_colors.py` – CPU Mesh Colors layout, per-face/instance offsets, Ptex-style triangle adjacency expansion, compact material-driven side slots, resolution clamps, and payload-slot budget enforcement.
- `mesh_colors_adjacency.py` – Shared-index triangle topology builder, edge-coordinate convention, diagnostics, and compact CPU/GPU adjacency packing. Split-index seams are boundaries; degenerate and non-manifold edges remain unlinked.
- `mesh_colors_rgb9e5.py` – GPU wrapper that packs filtered float irradiance into one RGB9E5 `uint` per payload.
- `surface_probe_vertex_lighting.py` – Builds an instanced, hard-edge-aware vertex topology graph and projects Surface Probe irradiance into a confidence-weighted screened-diffusion solve before RGBM packing.
- `surface_probes.py` – Compatible-support pre-analysis, `area*m` allocation, adaptive WSE, budgeted repair, protected zero-count closure, optional vertex anchors, double-sided expansion, and compact per-instance point-octree packing.
- `surface_probe_sampler.py` – Lazy CMake build, `ctypes` bindings, validation, and Python references for native weighted sample elimination, deficit repair, and compatible-kernel support estimation.
- `surface_probe_sampler.cpp`, `surface_probe_sampler.h` – C ABI around the vendored cyCodeBase WSE plus variable-radius adaptive WSE, deterministic global greedy repair, and parallel `f(x)`/`m(x)` estimation. Adaptive initialization is parallel; elimination uses an indexed mutable max-heap and neighbor-owned radii for correct asymmetric updates.
- `utils.py` – HDR EXR helpers.
- `vertex_baker/slang_viewer.py` – Native SlangPy baker viewer with HWRT or raster GBuffer, orbit camera, GPU picking, and visibility-cone diagnostics.

### Rendering Pass Wrappers
- `path_tracer.py` – Thin wrapper around path tracing compute shader; binds the `Scene`.
- `texture_space_path_tracer.py` – Dispatches progressive irradiance tracing over Mesh Colors texels.
- `mesh_colors_surface_filter.py` – Runs freeze-time one-ring Gaussian diffusion through one payload-sized ping-pong scratch buffer. Flat world normals are precomputed with transformed winding, including mirrored transforms.
- `mesh_colors_resolve.py` – Resolves the view-independent Mesh Colors cache through current-camera primary hits.
- `surface_probe_path_tracer.py` – Progressively traces diffuse irradiance into the sparse surface-probe cache.
- `surface_probe_resolve.py` – Reconstructs primary hits from the strongest compatible point-octree samples and exposes gathered-count, `f(x)`, `m(x)`, probe self-hit, and legacy fallback diagnostics.
- `surface_probe_path_tracing_renderer.py` – Orchestrates probe tracing, resolve, optional screen accumulation, and tone mapping.
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
| `mesh_colors.slang` | Barycentric lattice addressing, metadata lookup, triangle adjacency decode/edge remapping, and irradiance interpolation. | Texture-space passes. |
| `texture_space_path_tracer.slang` | Progressive per-texel irradiance path tracing with Nan sky/sun/static-shadow lighting. | `TextureSpacePathTracer`. |
| `mesh_colors_surface_filter.slang` | One-ring triangular-lattice Gaussian diffusion with one-edge adjacency remapping and flat-normal gating. | `MeshColorsSurfaceFilter`. |
| `mesh_colors_resolve.slang` | Primary-ray lookup of cached irradiance for the current camera. | `MeshColorsResolve`. |
| `mesh_colors_rgb9e5_pack.slang` | Packs writable float3 irradiance into a 32-bit shared-exponent RGB9E5 payload. | `MeshColorsRGB9E5Packer`. |
| `surface_probe_vertex_lighting.slang` | Gathers the indirect-only Surface Probe cache at render vertices, performs topology-aware screened diffusion, and packs one RGBM `uint` per instance vertex. Direct sun is evaluated in the preview resolve. | `SurfaceProbeVertexLighting`. |
| `mesh_colors_frozen_resolve.slang` | Primary-ray lookup from a read-only RGB9E5 irradiance buffer. | `MeshColorsResolve`. |
| `surface_probes.slang` | Point-octree gather, compact surface kernel, and irradiance reconstruction. | Surface-probe passes. |
| `surface_probe_path_tracer.slang` | Progressive irradiance tracing for base, repair, protected, and optional vertex-anchor records. | `SurfaceProbePathTracer`. |
| `surface_probe_resolve.slang` | Primary-hit visible gather, 2R emergency tier, diagnostics, and optional legacy vertex fallback. | `SurfaceProbeResolve`. |
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
- Renderers define UI controls in `setup_ui`; texture-space mode adds screen accumulation, pause/reset, exposure, cache status, and `Filter Before RGB9E5` controls for pass count, texel-hop spatial sigma, and normal-angle sigma.
- Surface-probe mode adds optional screen accumulation, pause/reset, exposure, status, and `Show Gather Count`; its false-color legend is magenta=0, red/orange=1-3, yellow=4-7, green=8-15, cyan=16-23, and blue/white=24-32.
- Event dispatcher routes keyboard presses to renderers; add new listeners via `scene.event_distpacher.subscribe`.

### Mesh Colors Surface-Filter Contract
- Each pass gathers only the center and six axial triangular-lattice neighbors. An out-of-face tap performs at most one adjacency crossing; corner taps that would need a second crossing are skipped. Therefore `N` ping-pong passes have finite support of at most `N` face rings, including on cyclic fans.
- Same-face taps use normal weight 1. Cross-face taps multiply the spatial Gaussian by `exp(-0.5 * (acos(clamp(dot(n0, n1), 0, 1)) / normalSigmaRadians)^2)`. Boundary, split-seam, degenerate, and non-manifold crossings are rejected, accepted weights are renormalized, and front/back caches are filtered independently.
- Save runs GPU filtering and RGB9E5 packing in the same command stream, switches immediately to frozen resolve, and releases the 16-byte writable and scratch caches. Reset discards the packed cache and restores iteration.

---

## Tooling & Debugging
- RenderDoc integration (auto-detected via `spy.renderdoc.is_available()`); call `spy.renderdoc.start_frame_capture` when `self.should_capture` is set.
- TEV viewer: `spy.tev.show_async` for quick HDR inspection.
- Shader debugging: drop colored diagnostics into RW textures; flush prints with `device.flush_print()` (first frame already does this).
- `test_print.py` + `test_print.slang` demonstrate GPU-side `print` debugging; run the script to verify Slang `print` output and use it as a template for logging thread-local data.
- GPU crashes: double-check `device` compiler options (`include_paths`, defines) in `app.py`.
- Native surface-probe sampling defaults to `auto`; use `--surface-probe-sampler-backend cpp` to require the C++ backend or `python` for the reference implementation. The backend controls both WSE and deficit repair; CMake output lives in `surface_probe_sampler_build/`.
- Surface-probe deficit repair runs after the exact-budget WSE base, targets the primary-radius gather count, may append up to `repair_budget_ratio * base_sites`, and never recomputes the base kernel radius.
- Adaptive surface-probe WSE is enabled by default. Triangle support produces `m=clamp(1/f,1,M)`; base allocation and proposal sampling use `area*m`, while WSE receives the normalized relative density `m/mean(m)` and local radius `r_base/sqrt(relative_density)`. `--surface-probe-no-adaptive-wse` restores the area-only path.
- Use `--surface-probe-profile-build` to emit flushed startup samples for asset import, Scene GPU/AS creation, adaptive prepass, candidate/WSE, budgeted repair, protected closure, support recomputation, octree packing, GPU upload, and shader initialization.

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
- `test_mesh_colors_adjacency.py` validates boundaries, strips, fans, winding, seams, closed and non-manifold topology, multi-instance ranges, and Python/Slang packing agreement.
- `test_mesh_colors_surface_filter.py` validates one-edge remapping, bounded fan propagation, normal gating, seam/non-manifold isolation, front/back separation, constant preservation, and variance reduction.
- `test_mesh_colors_rgb9e5.py` validates GPU packing, shader decoding, HDR edge cases, and shared-exponent quantization bounds.
- `test_surface_probes.py` validates native sampler determinism/version/spacing quality, adaptive population parity and normal isolation, C++/Python deficit-repair/support parity, budget caps, CPU/GPU strides, compact octree ranges, queries, and audit improvement.
- Texture-space GPU smoke: `python entry_point.py --renderer texture-space --headless --frames 16 --width 512 --height 512`.
- Surface-probe GPU smoke: `python entry_point.py --renderer surface-probe --surface-probe-count 37000 --headless --frames 4096 --width 512 --height 512`.

---

## Maintenance Checklist
- When adding/modifying passes or shaders, update both host wrapper and shader tables above.
- Smoke-test changes headlessly (`python entry_point.py --headless --frames 16`) to catch runtime issues without the UI loop.
- Leverage GPU-side `print` debugging (see `test_print.py` / `test_print.slang`) to trace per-thread state and flush with `device.flush_print()`.
