# Assignment 1: Tone Mapping Exposure & Shadow Mapping

This assignment contains two tasks to help you get familiar with the basic structure and rendering pipeline of the Nan renderer.

---

## Task 1: Tone Mapper Exposure Control

### Objective

Add an **exposure** parameter to the Tone Mapper, allowing users to adjust image brightness.

### Background

Exposure control is one of the most fundamental operations in a post-processing pipeline. Before tone mapping, we scale HDR color values by an exposure factor:

```
exposed_color = color * pow(2, exposure)
```

Where `exposure` is in **EV (Exposure Value)** units:
- `exposure = 0`: No change
- `exposure = 1`: Double brightness
- `exposure = -1`: Half brightness

### Files to Modify

1. **`tone_mapper.slang`** - Shader code
2. **`tone_mapper.py`** - Python wrapper

### Step-by-Step Guide

#### Step 1: Modify the Slang Shader

In `tone_mapper.slang`:

1. Add an `exposure` parameter to the `ToneMapper` struct
2. Apply exposure in the `execute()` function before calling `aces_film()`:

```slang
struct ToneMapper {
    Texture2D<float4> input;
    RWTexture2D<float4> output;
    float exposure;  // Add this

    void execute(uint2 pixel)
    {
        float3 i = input[pixel].xyz;
        // TODO: Apply exposure here
        float3 o = aces_film(i);
        output[pixel] = float4(o, 1.0);
    }
}
```

#### Step 2: Modify the Python Wrapper

In `tone_mapper.py`:

1. Add an `exposure` member variable (default value 0.0)
2. Modify the `execute()` method to pass `exposure` to the shader

```python
class ToneMapper:
    def __init__(self, device: spy.Device):
        # ...
        self.exposure = 0.0  # Add this

    def execute(self, command_encoder, input, output):
        self.kernel.dispatch(
            # ...
            vars={
                "g_tone_mapper": {
                    "input": input,
                    "output": output,
                    "exposure": self.exposure,  # Add this
                }
            },
            # ...
        )
```

### Verification

Modify `self.exposure` value and observe the rendering result:
- Set to `2.0`: The image should become noticeably brighter
- Set to `-2.0`: The image should become noticeably darker

### Bonus (Optional)

Add UI controls in `path_tracing_renderer.py` to adjust exposure in real-time using a slider.

---

## Task 2: Shadow Mapping for Directional Light

### Objective

Replace the current ray tracing visibility test with a **Shadow Map** for computing shadows from the directional light.

### Background

Shadow Mapping is a classic real-time shadow technique:

1. **First Pass**: Render the scene from the light's perspective, storing depth values into a shadow map
2. **Second Pass**: When rendering the scene, transform each pixel to light space and compare depth with the shadow map to determine if it's in shadow

For directional lights, we use **orthographic projection** to generate the shadow map.

### Important: Ray Tracing-Based Shadow Map Generation

**In this assignment, we do NOT use any rasterization.** The shadow map is generated using **ray tracing with orthographic projection**:

- Cast rays from a virtual orthographic camera aligned with the light direction
- For each pixel in the shadow map, trace a ray and record the hit distance as depth
- This approach is consistent with Nan's fully ray-traced architecture

This is different from traditional shadow mapping which uses rasterization. Here we use ray tracing for everything, including shadow map generation.

### Current Implementation

In `scene.slang`'s `sample_directional_light()` function, visibility is currently computed using ray tracing:

```slang
// Current implementation (line 392-394)
Ray shadow_ray = Ray(sd.compute_new_ray_origin(), L_dir);
float visibility = get_visibility(shadow_ray);
```

### Files to Create/Modify

1. **Create `shadow_map.slang`** - Shadow map generation compute shader
2. **Create `shadow_map.py`** - Shadow map generation pass
3. **Modify `scene.slang`** - Add shadow map sampling logic
4. **Modify `scene.py`** - Pass shadow map data
5. **Modify `path_tracing_renderer.py`** - Integrate shadow map pass

### Step-by-Step Guide

#### Step 1: Create Shadow Map Generation Pass

Create `shadow_map.slang`:

```slang
// Shadow map generation using ray tracing with orthographic projection
// 
// Key concept: Instead of rasterization, we trace rays from an orthographic
// camera to compute depth values.
//
// For each texel (x, y) in the shadow map:
// 1. Compute ray origin using orthographic projection (no perspective)
// 2. Ray direction = light direction (all rays are parallel)
// 3. Trace ray and store hit distance as depth

import scene;

struct ShadowMapGenerator {
    RWTexture2D<float> shadow_map;
    float4x4 light_view_matrix;
    float3 light_direction;
    float ortho_size;        // Half-size of orthographic frustum
    float near_plane;
    float far_plane;
    
    void execute(uint2 pixel)
    {
        uint2 dim;
        shadow_map.GetDimensions(dim.x, dim.y);
        
        // Convert pixel to normalized coordinates [-1, 1]
        float2 ndc = (float2(pixel) + 0.5) / float2(dim) * 2.0 - 1.0;
        
        // Orthographic ray origin (no perspective division)
        // TODO: Compute ray origin in world space
        
        // All rays point in the same direction (parallel projection)
        float3 ray_dir = light_direction;
        
        // TODO: Trace ray and store depth
    }
}
```

Create `shadow_map.py`:

```python
class ShadowMapPass:
    def __init__(self, device, shadow_map_size=2048):
        # Create shadow map texture (R32_FLOAT for depth)
        # Load compute shader
        pass
    
    def execute(self, command_encoder, scene, light_direction):
        # Compute light view matrix and orthographic bounds
        # Dispatch compute shader to trace rays and fill shadow map
        pass
```

#### Step 2: Modify Scene to Use Shadow Map

Add to `scene.slang`'s `Scene` struct:

```slang
struct Scene {
    // ... existing members ...
    
    Texture2D<float> shadow_map;          // Add this
    float4x4 light_view_proj_matrix;      // Add this
    float shadow_map_size;                // Add this
    
    // New function: compute visibility using shadow map
    float get_visibility_shadow_map(float3 world_pos)
    {
        // TODO: 
        // 1. Transform world_pos to light clip space
        // 2. Convert to shadow map UV coordinates
        // 3. Sample shadow map depth
        // 4. Compare depths to determine visibility
        // 5. Optional: Add bias to prevent shadow acne
    }
}
```

#### Step 3: Replace Visibility Test

In `sample_directional_light()`, replace:

```slang
Ray shadow_ray = Ray(sd.compute_new_ray_origin(), L_dir);
float visibility = get_visibility(shadow_ray);
```

With:

```slang
float visibility = get_visibility_shadow_map(sd.position);
```

### Verification

1. Run the renderer and check if shadows are displayed correctly
2. Compare the results between ray tracing visibility and shadow map
3. Look for shadow acne or peter panning artifacts

### Implementation Notes

1. **Light Space Matrix**:
   - View matrix: Look from light direction toward scene center
   - Projection matrix: Orthographic projection covering the entire scene

2. **Shadow Bias**:
   - Add depth bias to prevent shadow acne
   - Common approaches: constant bias `depth += bias` or slope-scale bias

3. **PCF (Percentage Closer Filtering)** (Optional Bonus):
   - Sample shadow map multiple times and average results
   - Produces soft shadow edges

### Why Ray Traced Shadow Maps?

You might wonder why we use ray tracing to generate shadow maps when we could just use ray tracing for direct visibility tests (which is what we're replacing). This is a **learning exercise** to understand:

1. How shadow maps work conceptually
2. The trade-offs between per-pixel ray queries vs. cached depth information
3. Shadow map artifacts (aliasing, acne, peter panning) that don't exist in pure ray tracing

In practice, ray traced visibility is often superior, but shadow maps remain important for understanding classical real-time rendering techniques.

### Bonus (Optional)

1. Implement PCF soft shadows
2. Implement Cascaded Shadow Maps (CSM) for better shadow quality at distance
3. Add UI controls for shadow map resolution and bias parameters

---

## Submission Requirements

1. Modified source code files
2. Screenshots demonstrating both tasks
3. Brief explanation of your implementation approach and any issues encountered

## Grading Rubric

| Item | Points |
|------|--------|
| Task 1: Basic exposure functionality | 30 |
| Task 1: UI controls (optional) | 10 |
| Task 2: Basic shadow map functionality | 40 |
| Task 2: Shadow bias handling | 10 |
| Task 2: PCF soft shadows (optional) | 10 |

---

## References

- [Learn OpenGL - Shadow Mapping](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping)
- [ACES Filmic Tone Mapping](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/)
- [Percentage-Closer Filtering](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing)
