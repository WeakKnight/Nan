static const float PI = 3.14159265358979323846;
static const float TWO_PI = 6.28318530717958647692;
static const float INV_PI = 0.31830988618379067154;

float3 encode_normal(float3 N)
{
	return N * 0.5 + 0.5;
}

float3 decode_normal(float3 N)
{
	return N * 2 - 1;
}

/**
 * Uniform sampling of a triangle.
 * @param[in] u Uniform random numbers in [0,1)^2.
 * @return Barycentric coordinates (1-u-v,u,v) of the sampled point.
 */
float3 sample_triangle(float2 u)
{
    float su = sqrt(u.x);
    float2 b = float2(1.f - su, u.y * su);
    return float3(1.f - b.x - b.y, b.x, b.y);
}

float4 diffuse_brdf_sample(float2 uv)
{
    float phi = uv.y * TWO_PI;
    float cos_theta = sqrt(1.f - uv.x);
    float sin_theta = sqrt(1.f - cos_theta * cos_theta);
    float pdf = cos_theta * INV_PI;
    return float4(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta, pdf);
}

// PDF = 1 / (2 * PI)
float4 uniform_sample_hemisphere( float2 E )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = E.y;
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;

	float PDF = 1.0 / (2 * PI);

	return float4( H, PDF );
}

// PDF = NoL / PI
float4 cosine_sample_hemisphere( float2 E )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = sqrt(E.y);
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

	float3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;

	float PDF = CosTheta * (1.0 / PI);

	return float4(H, PDF);
}

float3 diffuse_lambert(float3 diffuse_color)
{
	return diffuse_color * (1 / PI);
}

float3 compute_ray_origin(float3 pos, float3 normal)
{
    const float origin = 1.f / 16.f;
    const float fScale = 3.f / 65536.f;
    const float iScale = 3 * 256.f;

    // const float origin = 1.f / 16.f;
    // const float fScale = 6.f / 65536.f;
    // const float iScale = 6.2 * 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);
    float3 iPos = asfloat(asint(pos) + select(pos < 0.f, -iOff, iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = normal * fScale;
    float3 result = select(abs(pos) < origin, pos + fOff, iPos);

    if (dot(result - pos, result - pos) < 1e-8f)
    {
        result = pos + normal * 1e-4f;
    }

    return result;
}

float3 compute_flat_normal(float3 pos0, float3 pos1, float3 pos2, bool isOddNegative)
{
    float3 flatNormal = normalize(cross(pos1 - pos0, pos2 - pos0));

    if (isOddNegative)
    {
        flatNormal = -flatNormal;
    }

    return flatNormal;
}

float luminance(float3 linear_color)
{
    return dot(linear_color, float3(0.2126390059, 0.7151686788, 0.0721923154));
}

struct Ray
{
    float3 origin;
    float t_min;
    float3 dir;
    float t_max;

    __init(float3 origin, float3 dir, float t_min = 0.f, float t_max = 1000.0)
    {
        this.origin = origin;
        this.dir = dir;
        this.t_min = t_min;
        this.t_max = t_max;
    }

    RayDesc to_ray_desc() 
    { 
        return { origin, t_min, dir, t_max }; 
    }
};

struct Frame 
{
    float3 n;
    float3 t;
    float3 b;

    __init(float3 n)
    {
        // from Spencer, Jones "Into the Blue", eq(3)
        this.n = n;
        t = normalize(cross(n, float3(-n.z, n.x, -n.y)));
        b = cross(n, t);
    }

    float3 to_local(float3 w) 
    { 
        return float3(dot(t, w), dot(b, w), dot(n, w)); 
    }

    float3 to_global(float3 w) 
    { 
        return w.x * t + w.y * b + w.z * n; 
    }
};

struct RNG {
    uint state;

    __init(uint seed) 
    { 
        state = seed; 
    }

    __init(uint2 pixel, uint sample_index)
    {
        state = (pixel.x * 100591) ^ (pixel.y * 102191) ^ (sample_index * 103643);
    }

    [mutating]
    uint next()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_1d() 
    { 
        return (next() >> 8) * 0x1p-24; 
    }

    [mutating]
    float2 next_2d()
    {
        float2 result;
        result.x = next_1d();
        result.y = next_1d();
        return result;
    }

    [mutating]
    float3 next_3d()
    {
        float3 result;
        result.x = next_1d();
        result.y = next_1d();
        result.z = next_1d();
        return result;
    }
};

enum MaterialFlags
{
    DoubleSided = 1 << 0,
};

// Texture presence flags for MaterialDesc
enum TextureFlags
{
    HasBaseColorTex = 1 << 0,
    HasNormalTex = 1 << 1,
    HasRoughnessTex = 1 << 2,
    HasMetallicTex = 1 << 3,
    HasEmissiveTex = 1 << 4,
    HasSpecularColorTex = 1 << 5,
};

// Alpha modes (matches glTF spec)
static const uint ALPHA_MODE_OPAQUE = 0;  // fully opaque
static const uint ALPHA_MODE_MASK = 1;    // discard if alpha < cutoff
static const uint ALPHA_MODE_BLEND = 2;   // stochastic alpha

struct MaterialDesc {
    float3 base_color;
    uint flags;
    float3 emissive;
    uint shading_model;
    float roughness;
    float metallic;
    uint texture_flags;  // Bitmask indicating which textures are present
    uint alpha_mode;     // AlphaMode: OPAQUE=0, MASK=1, BLEND=2
    float alpha_cutoff;  // Alpha cutoff threshold (default 0.5)
    float3 specular_color; // Specular color (default white)
    // Bindless texture handles
    Texture2D<float4>.Handle base_color_tex_handle;
    Texture2D<float4>.Handle normal_tex_handle;
    Texture2D<float4>.Handle roughness_tex_handle;
    Texture2D<float4>.Handle metallic_tex_handle;
    Texture2D<float4>.Handle emissive_tex_handle;
    Texture2D<float4>.Handle specular_color_tex_handle;

    bool is_double_sided()
    {
        return (bool)(flags & (uint)MaterialFlags::DoubleSided);
    }
    
    bool has_base_color_texture()
    {
        return (bool)(texture_flags & (uint)TextureFlags::HasBaseColorTex);
    }
    
    bool has_normal_texture()
    {
        return (bool)(texture_flags & (uint)TextureFlags::HasNormalTex);
    }
    
    bool has_roughness_texture()
    {
        return (bool)(texture_flags & (uint)TextureFlags::HasRoughnessTex);
    }
    
    bool has_metallic_texture()
    {
        return (bool)(texture_flags & (uint)TextureFlags::HasMetallicTex);
    }
    
    bool has_emissive_texture()
    {
        return (bool)(texture_flags & (uint)TextureFlags::HasEmissiveTex);
    }
    
    bool has_specular_color_texture()
    {
        return (bool)(texture_flags & (uint)TextureFlags::HasSpecularColorTex);
    }
};

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;

    // bool frontFacing;
    // float3 faceN;

    // /*
    //  *  Computes new ray origin based on the hit point to avoid self-intersection.
    //  *  The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
    //  *  Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
    //  *  \param viewside: True if the origin should be on the view side (reflection) or false otherwise (transmission).
    //  *  \return: Ray origin of the new ray.
    //  */
    // float3 compute_new_ray_origin(bool viewside = true)
    // {
    //     return compute_ray_origin(position, (frontFacing == viewside) ? faceN : -faceN);
    // }
};

struct MeshDesc 
{
    uint vertex_count;
    uint index_count;
    uint vertex_offset;
    uint index_offset;
};

enum InstanceFlags
{
    None = 0,
    OddNegativeTransform = 1 << 0,
    Transmissive = 1 << 1,
};

struct InstanceDesc 
{
    uint mesh_id;
    uint material_id;
    uint transform_id;
    uint instance_flags;

    bool is_odd_negative()
    {
        return (bool)(instance_flags & (uint)InstanceFlags::OddNegativeTransform);
    }
};

struct HitInfo 
{
    bool valid;
    float2 bary;
    uint instance_id;
    uint primitive_index;

    float3 get_barycentrics()
    {
        return float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    }
};

struct ShadingData
{
    float3 position;
    float3 view_direction;
    Frame geometry_frame;
    Frame shading_frame;
    float3 face_normal;     // Already flipped toward viewer for double-sided materials
    float3 diffuse;
    float3 specular;
    float3 roughness;
    float3 emissive;
    bool front_facing;      // Original geometric orientation (for IOR in refraction)
    bool double_sided;

    /*
     *  Computes new ray origin based on the hit point to avoid self-intersection.
     *  The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
     *  Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
     *  \param position_bias_sign: +1 for reflection (ray goes to same side as view), 
     *                             -1 for transmission/refraction (ray goes through surface).
     *  \return: Ray origin of the new ray.
     */
    float3 compute_new_ray_origin(float position_bias_sign = 1.0)
    {
        // face_normal already points toward viewer, so:
        // - position_bias_sign = +1: offset along face_normal (reflection, same side as viewer)
        // - position_bias_sign = -1: offset against face_normal (transmission, opposite side)
        return compute_ray_origin(position, position_bias_sign * face_normal);
    }
};

float2 direction_to_equirect_uv(float3 normalizedDirection)
{
    float elevation = asin(normalizedDirection.y);
    float azimuth = 0;
    if (abs(normalizedDirection.y) < 1.0)
        azimuth = atan2(normalizedDirection.z, normalizedDirection.x);

    float2 uv;
    uv.x = azimuth / (2 * PI) - 0.25;
    uv.y = 0.5 - elevation / PI;

    return uv;
}

// Maps a cubemap direction to (u, v, array layer). Using a six-layer
// Texture2DArray keeps compute writes portable to Metal, where writable
// single-face cubemap views are not supported.
float3 direction_to_cube_array_uv(float3 direction)
{
    float3 d = normalize(direction);
    float3 ad = abs(d);
    float2 p;
    uint face;
    if (ad.x >= ad.y && ad.x >= ad.z)
    {
        if (d.x >= 0.0)
        {
            face = 0u;
            p = float2(-d.z, -d.y) / ad.x;
        }
        else
        {
            face = 1u;
            p = float2(d.z, -d.y) / ad.x;
        }
    }
    else if (ad.y >= ad.z)
    {
        if (d.y >= 0.0)
        {
            face = 2u;
            p = float2(d.x, d.z) / ad.y;
        }
        else
        {
            face = 3u;
            p = float2(d.x, -d.z) / ad.y;
        }
    }
    else if (d.z >= 0.0)
    {
        face = 4u;
        p = float2(d.x, -d.y) / ad.z;
    }
    else
    {
        face = 5u;
        p = float2(-d.x, -d.y) / ad.z;
    }
    return float3(p * 0.5 + 0.5, float(face));
}

// Sample a direction within a cone around an axis
// Used for directional light (sun disk) sampling
// PDF = 1 / (2 * PI * (1 - cos_half_angle))
float4 sample_cone_direction(float3 axis, float cos_half_angle, float2 u)
{
    // Sample uniformly within the spherical cap
    float cos_theta = 1.0 - u.x * (1.0 - cos_half_angle);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float phi = TWO_PI * u.y;
    
    // Local direction (z-up)
    float3 local_dir = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    
    // Transform to world space using Frame
    Frame frame = Frame(axis);
    float3 world_dir = frame.to_global(local_dir);
    
    // PDF for uniform cone sampling
    float pdf = 1.0 / (TWO_PI * (1.0 - cos_half_angle));
    
    return float4(world_dir, pdf);
}

// Adjust shading normal to avoid dark artifacts at grazing angles
// Based on UE's AdjustShadingNormal (see PathTracingCommon.ush)
// NOTE: This assumes normals are already flipped to point toward the viewer (via TwoSidedSign)
float3 adjust_shading_normal(float3 shading_normal, float3 geo_normal, float3 ray_direction)
{
    // Clip shading normal in a view dependent way such that the reflection stays above the geometric normal
    // This introduces a bit of view-dependency to the shading normal but fixes dark artifacts around grazing angles
    
    float3 D = ray_direction;
    float3 R = reflect(D, shading_normal);
    
    // https://iquilezles.org/www/articles/dontflip/dontflip.htm
    float k = dot(R, geo_normal);
    if (k < 0.0)
    {
        return normalize(normalize(R - k * geo_normal) - D);
    }
    return shading_normal;
}

// Smart bent normal for ray tracing (legacy version that handles normal flipping internally)
// See appendix A.3 in https://arxiv.org/pdf/1705.01263.pdf
// WARNING: This function flips normals internally - don't use if normals are already flipped!
float3 get_bent_normal(float3 geometry_normal, float3 shading_normal, float3 view_direction)
{
    // Flip the normal in case we're looking at the geometry from its back side
    if (dot(geometry_normal, view_direction) > 0)
    {
        geometry_normal = -geometry_normal;
        shading_normal = -shading_normal;
    }

    // Specular reflection in shading normal
    float3 R = reflect(view_direction, shading_normal);
    float a = dot(geometry_normal, R);
    if (a < 0) // Perturb normal
    {
        float b = max(0.001, dot(shading_normal, geometry_normal));
        return normalize(-view_direction + normalize(R - shading_normal * a / b));
    }

    return shading_normal;
}

// ============================================================================
// Murmur hash functions for stochastic alpha
// ============================================================================

uint murmur_add(uint hash, uint element)
{
    element *= 0xcc9e2d51;
    element = (element << 15) | (element >> (32 - 15));
    element *= 0x1b873593;

    hash ^= element;
    hash = (hash << 13) | (hash >> (32 - 13));
    hash = hash * 5 + 0xe6546b64;
    return hash;
}

uint murmur_mix(uint hash)
{
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
}

uint murmur_key_mix(uint2 key)
{
    return murmur_mix(murmur_add(key.x, key.y));
}

uint position_hashing(float3 pos, float scale)
{
    int3 signed_pos = int3(pos * scale) + 0x3ffff;
    uint3 voxel = uint3(signed_pos) & 0x7ffff;

    uint3 key;
    key.x = voxel.x;
    key.y = voxel.y;
    key.z = voxel.z;

    uint2 hash_key = uint2(0, 0);
    hash_key.x = hash_key.x | ((key.x & 0x7fff) << 7);
    hash_key.x = hash_key.x | ((key.y & 0x7f) << 0);
    hash_key.y = hash_key.y | (key.y & 0xfff) << 20;
    hash_key.y = hash_key.y | (key.z & 0x7fff) << 1;

    return murmur_key_mix(hash_key);
}

// Simple RNG for stochastic alpha
struct RandomSamplerState
{
    uint seed;
    uint index;
};

float sample_uniform_rng(inout RandomSamplerState rng)
{
    rng.seed = murmur_key_mix(uint2(rng.seed, rng.index++));
    return float(rng.seed) / float(0xffffffffu);
}
