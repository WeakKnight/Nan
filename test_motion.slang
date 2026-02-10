// Test shader for motion vector verification.
// Traces primary rays, outputs world position and motion vectors.

import common;
import scene;
import sgl.device.print;

RWTexture2D<float4> g_world_pos;    // xyz = world position, w = 1 if hit, 0 if miss
RWTexture2D<float4> g_motion;       // xy = motion in pixels, z = viewZ delta, w = 0
RWTexture2D<float4> g_color;        // xyz = path traced color, w = 1
uniform uint g_frame;

// Simple single-bounce path trace for color (deterministic, no RNG jitter on ray)
float3 trace_color(Ray ray)
{
    HitInfo hit = g_scene.intersect(ray);
    if (!hit.valid)
        return g_scene.sample_sky_with_sun_disk(ray.dir);

    ShadingData sd = g_scene.get_shading_data(hit, -ray.dir);
    
    // Direct lighting only (sun NEE) for deterministic color
    float3 L = sd.emissive;
    
    // NEE sun
    // Use a fixed sample for determinism
    float2 light_sample = float2(0.5, 0.5);
    float4 nee = g_scene.sample_directional_light(sd, light_sample);
    L += nee.xyz;
    
    // Add ambient term so we get non-zero color even in shadow
    L += sd.diffuse * 0.05;
    
    return L;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_world_pos.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    float2 rect_size = float2(dim);

    // Trace primary ray (no jitter for stable test)
    Ray ray = g_scene.camera.compute_ray_pinhole(pixel, dim, false);
    HitInfo hit = g_scene.intersect(ray);

    if (hit.valid)
    {
        Vertex vertex = g_scene.get_vertex(hit);
        float3 X = vertex.position;

        // Static geometry: Xprev = X
        float3 motion = g_scene.get_motion(X, X, rect_size);

        g_world_pos[pixel] = float4(X, 1.0);
        g_motion[pixel] = float4(motion, 0.0);
    }
    else
    {
        g_world_pos[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        g_motion[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    }

    // Trace color
    g_color[pixel] = float4(trace_color(ray), 1.0);
}
