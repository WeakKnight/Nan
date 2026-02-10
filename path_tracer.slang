import common;
import scene;
import sgl.device.print;

static const float PT_EPS = 1e-6;

struct Path {
    uint2 pixel;
    uint vertex_index;
    Ray ray;
    float3 thp;
    float3 L;
    RNG rng;

    __init(uint2 pixel, Ray ray, RNG rng)
    {
        this.pixel = pixel;
        this.vertex_index = 0;
        this.ray = ray;
        this.thp = float3(1);
        this.L = float3(0);
        this.rng = rng;
    }
};

RWTexture2D<float4> g_output;
RWTexture2D<float4> g_motion;   // .xy = motion in pixels, .z = viewZ delta, .w = viewZ (before PSR)
RWTexture2D<float>  g_viewz;    // linear view-space Z (sign * INF for miss)
uniform uint g_frame;

void trace_path(inout Path path)
{
    while (path.vertex_index < 5)
    {
        HitInfo hit = g_scene.intersect(path.ray);
        if (!hit.valid) {
            // For primary rays (vertex_index == 0), include sun disk
            // For bounce rays, only sample sky LUT (sun contribution handled by NEE)
            if (path.vertex_index == 0) {
                path.L += path.thp * g_scene.sample_sky_with_sun_disk(path.ray.dir);
            } else {
                path.L += path.thp * g_scene.sample_sky_lut(path.ray.dir);
            }
            break;
        }

        Vertex vertex = g_scene.get_vertex(hit);
        ShadingData shading_data = g_scene.get_shading_data(hit, -path.ray.dir);

        Frame frame = shading_data.shading_frame;
        float3 wi = frame.to_local(shading_data.view_direction);
        
        // Add emissive contribution
        path.L += path.thp * shading_data.emissive;
        
        // NEE: Sample directional light (sun) directly
        float4 nee_result = g_scene.sample_directional_light(shading_data, path.rng.next_2d());
        path.L += path.thp * nee_result.xyz;
        
        // Continue path with BSDF sampling
        float4 e = cosine_sample_hemisphere(path.rng.next_2d());
        float3 wo = e.xyz;
        float pdf = max(e.w, PT_EPS);
        float3 f = shading_data.diffuse * INV_PI;
        path.thp *= f * abs(wo.z) / pdf;
        path.vertex_index++;
        path.ray = Ray(shading_data.compute_new_ray_origin(), frame.to_global(wo));
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_output.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

#if 0 // DEBUG: Direct texture output without lighting
    Ray ray = g_scene.camera.compute_ray_pinhole(pixel, dim);
    HitInfo hit = g_scene.intersect(ray);
    
    if (hit.valid) {
        InstanceDesc inst = g_scene.instance_descs[hit.instance_id];
        MaterialDesc material = g_scene.load_material_desc(inst.material_id);
        Vertex vertex = g_scene.get_vertex(hit);
        
        if (material.has_base_color_texture()) {
            float4 tex_sample = material.base_color_tex_handle.SampleLevel(g_scene.linear_sampler, vertex.uv, 0);
            g_output[pixel] = float4(tex_sample.rgb, 1);
        } else {
            g_output[pixel] = float4(material.base_color, 1);
        }
    } else {
        g_output[pixel] = float4(g_scene.sample_sky_lut(ray.dir), 1);
    }
    return;
#endif

    // Primary ray for G-buffer (motion + viewZ)
    {
        Ray primary_ray = g_scene.camera.compute_ray_pinhole(pixel, dim, false); // no jitter for G-buffer
        HitInfo primary_hit = g_scene.intersect(primary_ray);

        if (primary_hit.valid)
        {
            Vertex v = g_scene.get_vertex(primary_hit);
            float3 X = v.position;
            // Static geometry: Xprev == X
            float3 motion = g_scene.get_motion(X, X, float2(dim));
            float viewZ = mul(g_scene.camera.view_matrix, float4(X, 1.0)).z;
            g_motion[pixel] = float4(motion, viewZ);
            g_viewz[pixel] = viewZ;
        }
        else
        {
            // Miss: compute a large viewZ along the ray direction
            float viewZ_sign = sign(mul(g_scene.camera.view_matrix, float4(primary_ray.dir, 0.0)).z);
            float viewZ = viewZ_sign * 1e30;
            g_motion[pixel] = float4(0, 0, 0, viewZ);
            g_viewz[pixel] = viewZ;
        }
    }

    // Path trace for color
    const uint spp = 1;
    float3 L = float3(0);
    for (uint i = 0; i < spp; i++) {
        RNG rng = RNG(pixel, g_frame * spp + i);
        float2 uv = float2(pixel + rng.next_2d()) / float2(dim);
        uv.y = 1 - uv.y;
        Ray ray = g_scene.camera.compute_ray_pinhole(pixel, dim);
        Path path = Path(pixel, ray, rng);
        trace_path(path);
        L += path.L;
    }
    L /= spp;
    g_output[pixel] = float4(L, 1);
}
