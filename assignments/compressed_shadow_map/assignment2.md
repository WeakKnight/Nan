# Assignment 2: Advanced Shadow Techniques

This assignment builds on Assignment 1. You will implement three techniques to improve shadow quality and performance.

**Prerequisite:** Assignment 1 completed (shadow map generation and `get_visibility_shadow_map()` working).

---

## Task 1: Poisson Disk PCF with Temporal Filtering

### Objective

Implement **Percentage Closer Filtering (PCF)** using Poisson disk samples, combined with **temporal filtering** to produce smooth soft shadows.

### Background

Basic shadow mapping produces hard, aliased edges. PCF softens them by sampling the shadow map at multiple offsets and averaging the binary results.

Using a **Poisson disk pattern** avoids banding artifacts of regular grids. By **rotating** the pattern randomly per pixel per frame and accumulating over time, the result converges to smooth soft shadows with few samples per frame.

The project provides `LowDiscrepancyDiskPattern` (`low_discrepancy_disk_pattern.py`), which stores R2 quasi-random disk samples in a GPU `StructuredBuffer<float2>`.

### Files to Modify

- `scene.slang` — Add PCF sampling logic to `get_visibility_shadow_map()`
- `scene.py` — Bind Poisson disk buffer and PCF parameters
- `path_tracing_renderer.py` — Create and pass `LowDiscrepancyDiskPattern`

### Key Ideas

1. Pass the disk sample buffer and a `pcf_filter_radius` to the shader
2. For each shading point, pick a random rotation angle (using `RNG` + `frame_index`) and rotate each disk offset before scaling it to shadow map texel space
3. Sample and average multiple shadow map lookups at the rotated offsets
4. The existing `Accumulator` handles temporal averaging — because the rotation changes each frame, accumulated results converge naturally

### Verification

- `filter_radius = 0` should reproduce Assignment 1's hard shadows
- With accumulation **off**, soft edges should appear noisy
- With accumulation **on**, shadows should converge to smooth results after ~32 frames

---

## Task 2: Adaptive Shadow Mask

### Objective

Implement an **adaptive shadow mask** pass inspired by [Deferred Adaptive Compute Shading](https://github.com/WeakKnight/DeferredAdaptiveComputeShading), reducing shadow evaluations by interpolating in geometrically smooth regions.

### Background

Full-screen PCF is expensive. Neighboring pixels on the same surface often share identical shadow values. The adaptive approach:

1. Evaluates shadow for a sparse set of **anchor pixels** (e.g., one per 4×4 block)
2. For remaining pixels, tests geometric similarity with already-evaluated neighbors (using depth and normal)
3. **Interpolates** if similar; **evaluates** if dissimilar
4. Organized as a multi-pass progressive fill over screen-space blocks

The output is a single-channel **shadow mask** texture (0 = shadow, 1 = lit).

### Files to Create/Modify

- **Create `adaptive_shadow_mask.slang`** — Multi-pass compute shader
- **Create `adaptive_shadow_mask.py`** — Python wrapper
- **Modify `path_tracer.slang`** — Output G-Buffer (depth + normal) for primary hits
- **Modify `path_tracing_renderer.py`** — Integrate the new pass

### Key Ideas

1. **G-Buffer**: Output linear depth and world-space normal from the path tracer's primary ray hit
2. **Similarity test**: Two pixels are "similar" if their depth difference and normal angle are both within thresholds
3. **Multi-pass fill**: Design a fill pattern that starts sparse (anchors) and progressively fills in detail — only evaluating shadow where neighbors disagree geometrically
4. **World position reconstruction**: Use inverse view-projection + depth to get world position for `get_visibility_shadow_map()`

### Verification

- Compare the adaptive mask against a full-screen brute-force mask — they should be nearly identical
- Visualize evaluated (red) vs interpolated (green) pixels to confirm the adaptive pattern
- Tune thresholds: too aggressive → shadow leaking; too conservative → no speedup

---

## Task 3: Hybrid Ray Tracing + Shadow Map Shadows

### Objective

Implement a **hybrid shadow** technique: **ray tracing** for nearby shadows, **shadow maps** for distant shadows.

### Background

Ray traced shadows are precise but expensive. Shadow maps are fast but aliased. A hybrid approach uses each where it excels:

- **Near range** (distance to camera < threshold): Ray trace via `get_visibility()` for artifact-free shadows
- **Far range**: Use `get_visibility_shadow_map()` where aliasing is less noticeable
- **Blend zone**: Smooth transition between the two to avoid seams

### Files to Modify

- `scene.slang` — Add a hybrid visibility function
- `scene.py` — Pass distance thresholds

### Key Ideas

1. Add `get_visibility_hybrid()` to the `Scene` struct that selects between RT and shadow map based on camera distance
2. Use `smoothstep(near, far, dist)` in the blend zone to lerp between both results
3. Replace the visibility call in `sample_directional_light()` with the hybrid version
4. Expose `near_distance` and `far_distance` as tunable parameters

### Verification

- `near = far = 0`: Pure shadow map — should match Assignment 1
- `near = far = very large`: Pure ray tracing — should match original
- Normal hybrid: Near shadows crisp, far shadows use shadow map, blend zone seamless

---

## References

- [Percentage-Closer Filtering (GPU Gems)](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing)
- [Poisson Disk Sampling](https://www.jasondavies.com/poisson-disc/)
- [Deferred Adaptive Compute Shading](https://github.com/WeakKnight/DeferredAdaptiveComputeShading)
- [Hybrid Ray Traced Shadows](https://developer.nvidia.com/blog/hybrid-rendering-for-real-time-ray-tracing/)
