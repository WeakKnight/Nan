import common;
import camera;

ParameterBlock<Scene> g_scene;

// Directional light (sun) parameters
struct DirectionalLight {
    float3 direction;           // Light direction (pointing toward the sun)
    float cos_half_angle;       // cos(half_apex_angle), for cone sampling. Default: cos(0.2725°) ≈ 0.99999
    float3 color;               // Light color (default: white)
    float intensity;            // Light intensity in outer space (default: PI ≈ 3.14159)
    
    // Outer space illuminance (used for sky LUT, which already includes scattering)
    float3 get_outer_space_illuminance() {
        return color * intensity;
    }
};

struct SSTNode {
    uint flags;
    uint child_base;
    uint meta;
    uint reserved;
    float plane_a;
    float plane_b;
    float plane_c;
    float max_error;
};

struct StaticShadowDepthMapData {
    Texture2D<float> depth_texture;
    Texture2D<float> sst_decompressed_texture;
    SamplerState depth_sampler;
    StructuredBuffer<SSTNode> sst_nodes;
    StructuredBuffer<uint2> sst_packed_nodes;
    StructuredBuffer<uint> sst_compact_words;
    StructuredBuffer<uint> sst_tile_roots;
    StructuredBuffer<uint> sst_compact_roots;
    float4x4 world_to_shadow;
    uint enabled;
    uint2 resolution;
    float depth_bias;
    uint shadow_mode;
    uint sst_enabled;
    uint sst_tile_size;
    uint2 sst_tile_grid;
    uint sst_node_count;
    uint sst_max_traversal_steps;
    uint sst_compact_word_count;
    uint sst_branch_10bit_start_level;
    uint pad0;
};

static const uint SHADOW_MODE_REALTIME = 0;
static const uint SHADOW_MODE_DEPTH_TEXTURE = 1;
static const uint SHADOW_MODE_SST = 2;
static const uint SHADOW_MODE_PACKED_SST = 3;
static const uint SHADOW_MODE_COMPACT_SST = 4;
static const uint SHADOW_MODE_COMPACT_SST_PCF3 = 5;
static const uint SHADOW_MODE_DECOMPRESSED_SST = 6;
static const uint SST_NODE_FLAG_PLANE = 1 << 0;

// Sky atmosphere constants (from atmosphere.slang)
static const float groundRadiusMM = 6.360;
static const float atmosphereRadiusMM = 6.460;
// 200M above the ground
static const float3 viewPos = float3(0.0, groundRadiusMM + 0.0002, 0.0);
static const float2 skyLUTRes = float2(200.0, 200.0);

float safeacos(float x) {
    return acos(clamp(x, -1.0, 1.0));
}

struct Scene {
    RaytracingAccelerationStructure tlas;

    StructuredBuffer<MaterialDesc> material_descs;
    
    StructuredBuffer<MeshDesc> mesh_descs;
    StructuredBuffer<InstanceDesc> instance_descs;
    StructuredBuffer<Vertex> vertices;
    StructuredBuffer<uint> indices;

    StructuredBuffer<float4x4> transforms;
    StructuredBuffer<float4x4> inverse_transpose_transforms;

    Texture2D<float4> transmittance_lut;
    SamplerState transmittance_sampler;

    Texture2D<float4> sky_view_lut;
    SamplerState linear_sampler;
    float3 sun_direction;
    
    DirectionalLight directional_light;
    StaticShadowDepthMapData static_shadow;

    Camera camera;

    uint instance_count;
    uint frame_index;
    uint pad1;
    uint pad2;

    // Load MaterialDesc directly from StructuredBuffer
    MaterialDesc load_material_desc(uint index)
    {
        return material_descs[index];
    }


    MaterialDesc get_material(HitInfo hit_info)
    {
        InstanceDesc instance_desc = g_scene.instance_descs[hit_info.instance_id];
        return load_material_desc(instance_desc.material_id);
    }

    // Get opacity from material at given UV (samples base_color_texture alpha channel if available)
    float get_opacity(MaterialDesc material, float2 uv)
    {
        if (material.has_base_color_texture())
        {
            return material.base_color_tex_handle.SampleLevel(linear_sampler, uv, 0).a;
        }
        return 1.0;
    }

    // Evaluate alpha test based on material alpha mode
    // Returns true if the hit should be accepted, false if it should be rejected
    bool eval_alpha_test(MaterialDesc material, float opacity, float3 world_pos, uint frame_index)
    {
        if (material.alpha_mode == ALPHA_MODE_OPAQUE)
            return true;
        
        if (material.alpha_mode == ALPHA_MODE_MASK)
            return opacity >= material.alpha_cutoff;
        
        // BLEND: stochastic alpha
        RandomSamplerState rng;
        rng.seed = position_hashing(world_pos, 10000.0f);
        rng.index = frame_index;
        return opacity >= sample_uniform_rng(rng);
    }

    Vertex get_vertex(HitInfo hit_info)
    {
        return get_vertex(hit_info.instance_id, hit_info.primitive_index, hit_info.bary);
    }

    Vertex get_vertex(uint instance_id, uint primitive_index, float2 bary)
    {
        InstanceDesc instance_desc = g_scene.instance_descs[instance_id];
        MeshDesc mesh_desc = g_scene.mesh_descs[instance_desc.mesh_id];
        float4x4 transform = g_scene.transforms[instance_desc.transform_id];
        float4x4 inverse_transpose_transform = g_scene.inverse_transpose_transforms[instance_desc.transform_id];
        uint i0 = g_scene.indices[mesh_desc.index_offset + primitive_index * 3 + 0];
        uint i1 = g_scene.indices[mesh_desc.index_offset + primitive_index * 3 + 1];
        uint i2 = g_scene.indices[mesh_desc.index_offset + primitive_index * 3 + 2];
        Vertex v0 = g_scene.vertices[mesh_desc.vertex_offset + i0];
        Vertex v1 = g_scene.vertices[mesh_desc.vertex_offset + i1];
        Vertex v2 = g_scene.vertices[mesh_desc.vertex_offset + i2];
        float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
        return {
            mul(transform, float4(b.x * v0.position + b.y * v1.position + b.z * v2.position, 1.0)).xyz,
            normalize(
                mul(inverse_transpose_transform, float4(b.x * v0.normal + b.y * v1.normal + b.z * v2.normal, 0)).xyz
            ),
            b.x * v0.uv + b.y * v1.uv + b.z * v2.uv,
        };
    }

    ShadingData get_shading_data(HitInfo hit_info, const float3 view_direction)
    {
        uint instance_id = hit_info.instance_id;
        uint primitive_index = hit_info.primitive_index;
        float2 bary = hit_info.bary;

        InstanceDesc instance_desc = g_scene.instance_descs[instance_id];
        MeshDesc mesh_desc = g_scene.mesh_descs[instance_desc.mesh_id];
        float4x4 transform = g_scene.transforms[instance_desc.transform_id];
        float4x4 inverse_transpose_transform = g_scene.inverse_transpose_transforms[instance_desc.transform_id];
        uint i0 = g_scene.indices[mesh_desc.index_offset + primitive_index * 3 + 0];
        uint i1 = g_scene.indices[mesh_desc.index_offset + primitive_index * 3 + 1];
        uint i2 = g_scene.indices[mesh_desc.index_offset + primitive_index * 3 + 2];
        Vertex v0 = g_scene.vertices[mesh_desc.vertex_offset + i0];
        Vertex v1 = g_scene.vertices[mesh_desc.vertex_offset + i1];
        Vertex v2 = g_scene.vertices[mesh_desc.vertex_offset + i2];
        float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

        float3 position = mul(transform, float4(b.x * v0.position + b.y * v1.position + b.z * v2.position, 1.0)).xyz;
        float2 uv = b.x * v0.uv + b.y * v1.uv + b.z * v2.uv;
        float3 geometry_normal = normalize(
            mul(inverse_transpose_transform, float4(b.x * v0.normal + b.y * v1.normal + b.z * v2.normal, 0)).xyz
        );
        float3 shading_normal = geometry_normal;
        float3 object_space_flat_normal = normalize(cross(
            v1.position - v0.position,
            v2.position - v0.position));

        float3 face_normal = normalize(
            mul((float3x3)inverse_transpose_transform, object_space_flat_normal)
        );

        MaterialDesc material = g_scene.get_material(hit_info);

        // Sample textures using Bindless handles
        float3 diffuse_color = material.base_color;
        float3 emissive_color = material.emissive;
        float roughness_value = material.roughness;
        float metallic_value = material.metallic;
        float3 specular_color_value = material.specular_color;
        
        if (material.has_base_color_texture())
        {
            float4 tex_sample = material.base_color_tex_handle.SampleLevel(linear_sampler, uv, 0);
            diffuse_color = tex_sample.rgb;
        }
        
        if (material.has_emissive_texture())
        {
            emissive_color = material.emissive_tex_handle.SampleLevel(linear_sampler, uv, 0).rgb * material.emissive;
        }
        
        if (material.has_roughness_texture())
        {
            // glTF: roughness is in G channel of metallicRoughness texture
            roughness_value = material.roughness_tex_handle.SampleLevel(linear_sampler, uv, 0).g;
        }
        
        if (material.has_metallic_texture())
        {
            // glTF: metallic is in B channel of metallicRoughness texture
            metallic_value = material.metallic_tex_handle.SampleLevel(linear_sampler, uv, 0).b;
        }
        
        if (material.has_specular_color_texture())
        {
            specular_color_value = material.specular_color_tex_handle.SampleLevel(linear_sampler, uv, 0).rgb * material.specular_color;
        }
        
        // Normal mapping
        if (material.has_normal_texture())
        {
            float3 normal_sample = material.normal_tex_handle.SampleLevel(linear_sampler, uv, 0).rgb;
            float3 tangent_normal = normalize(normal_sample * 2.0 - 1.0);
            
            // Compute tangent frame from geometry (simplified - assumes UV mapping)
            // For production, should compute proper tangent from vertex data
            float3 dp1 = v1.position - v0.position;
            float3 dp2 = v2.position - v0.position;
            float2 duv1 = v1.uv - v0.uv;
            float2 duv2 = v2.uv - v0.uv;
            
            float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (abs(det) > 1e-6)
            {
                float3 tangent = normalize((dp1 * duv2.y - dp2 * duv1.y) / det);
                tangent = mul((float3x3)inverse_transpose_transform, tangent);
                tangent = normalize(tangent - geometry_normal * dot(tangent, geometry_normal));
                float3 bitangent = cross(geometry_normal, tangent);
                
                shading_normal = normalize(
                    tangent_normal.x * tangent +
                    tangent_normal.y * bitangent +
                    tangent_normal.z * geometry_normal
                );
            }
        }

        // Double-sided handling (similar to UE's TwoSidedSign approach)
        // IsFrontFace records the original geometric orientation
        // Normals are flipped to always point toward the viewer
        bool double_sided = material.is_double_sided();
        bool is_front_face = dot(face_normal, view_direction) > 0;
        
        // Compute TwoSidedSign: +1 for front face, -1 for back face (when double-sided)
        float two_sided_sign = (double_sided && !is_front_face) ? -1.0 : 1.0;
        
        // Apply TwoSidedSign to all normals (like UE's WorldGeoNormal = TwoSidedSign * GeoNormal)
        face_normal *= two_sided_sign;
        geometry_normal *= two_sided_sign;
        shading_normal *= two_sided_sign;
        
        // Adjust shading normal to avoid self-intersection issues (like UE's AdjustShadingNormal)
        // Note: use -view_direction as ray_direction (ray goes from camera toward surface)
        shading_normal = adjust_shading_normal(shading_normal, face_normal, -view_direction);

        ShadingData shading_data;
        shading_data.position = position;
        shading_data.view_direction = view_direction;
        shading_data.geometry_frame = Frame(geometry_normal);
        shading_data.shading_frame = Frame(shading_normal);
        shading_data.face_normal = face_normal;  // Already flipped toward viewer
        // is_front_face records original geometric orientation (like UE's IsFrontFace())
        // Used for IOR calculations in refraction, not for ray origin bias
        shading_data.front_facing = is_front_face;
        shading_data.double_sided = double_sided;
        shading_data.diffuse = diffuse_color;
        shading_data.emissive = emissive_color;
        shading_data.roughness = roughness_value;
        shading_data.specular = 0.04 * (1.0 - metallic_value) * specular_color_value;  // Dielectric F0 * specular color
        return shading_data;
    }

    // Ray-sphere intersection helper (returns t, or -1 if no intersection)
    float ray_sphere_intersect(float3 ray_origin, float3 ray_dir, float3 sphere_center, float sphere_radius)
    {
        float3 oc = ray_origin - sphere_center;
        float a = dot(ray_dir, ray_dir);
        float b = 2.0 * dot(oc, ray_dir);
        float c = dot(oc, oc) - sphere_radius * sphere_radius;
        float discriminant = b * b - 4.0 * a * c;
        
        if (discriminant < 0.0)
            return -1.0;
        
        float sqrt_disc = sqrt(discriminant);
        float t0 = (-b - sqrt_disc) / (2.0 * a);
        float t1 = (-b + sqrt_disc) / (2.0 * a);
        
        // Return nearest positive intersection
        if (t0 > 0.0) return t0;
        if (t1 > 0.0) return t1;
        return -1.0;
    }

    // Get sun disk luminance if view direction is within sun disk cone
    // Reference: UE SkyAtmosphereCommon.ush GetLightDiskLuminance()
    // UE's implementation is simple: just a soft edge factor, no limb darkening
    float3 get_sun_disk_luminance(float3 view_dir)
    {
        // Check if view ray intersects the planet (ground)
        float t = ray_sphere_intersect(viewPos, view_dir, float3(0.0), groundRadiusMM);
        if (t > 0.0)
        {
            return float3(0.0);
        }
        
        float view_dot_light = dot(view_dir, directional_light.direction);
        float cos_half_apex = directional_light.cos_half_angle;
        
        if (view_dot_light > cos_half_apex)
        {
            float3 transmittance = get_sun_transmittance_at_view();
            
            // Calculate sun disk luminance from illuminance
            float solid_angle = 2.0 * PI * (1.0 - cos_half_apex);
            float3 sun_disk_luminance = directional_light.get_outer_space_illuminance() / solid_angle;
            
            // UE's soft edge formula:
            // Soften out the sun disk to avoid bloom flickering at edge.
            // The soften is applied on the outer part of the disk (outer 50%).
            // At center (view_dot_light=1): soft_edge = 2.0 -> saturate -> 1.0
            // At edge (view_dot_light=cos_half_apex): soft_edge = 0.0
            // At 50% from edge: soft_edge = 1.0
            float soft_edge = saturate(2.0 * (view_dot_light - cos_half_apex) / (1.0 - cos_half_apex));
            
            return transmittance * sun_disk_luminance * soft_edge;
        }
        
        return float3(0.0);
    }

    // Sample sky LUT only (atmospheric scattering, no sun disk)
    // Used for bounce rays where sun contribution is already handled by NEE
    float3 sample_sky_lut(float3 rayDir)
    {
        float height = length(viewPos);
        float3 up = viewPos / height;
        
        float horizonAngle = safeacos(sqrt(height * height - groundRadiusMM * groundRadiusMM) / height) - 0.5 * PI;
        float altitudeAngle = clamp(asin(dot(rayDir, up)), -0.5 * PI + 1e-2, 0.5 * PI - 1e-2); // Between -PI/2 and PI/2
        
        // Compute azimuth relative to sun direction
        float azimuthAngle; // Between -PI and PI
        if (abs(altitudeAngle) > (0.5 * PI - 0.0001)) {
            // Looking nearly straight up or down
            azimuthAngle = 0.0;
        } else {
            float3 right = cross(sun_direction, up);
            float3 forward = cross(up, right);
            
            float3 projectedDir = normalize(rayDir - up * dot(rayDir, up));
            float sinTheta = dot(projectedDir, right);
            float cosTheta = dot(projectedDir, forward);
            azimuthAngle = atan2(sinTheta, cosTheta); // Range [-PI, PI]
        }
        
        // Inverse of the non-linear mapping used in sky_view_lut generation
        // Generation: altitudeAngle = adjV * 0.5 * PI - horizonAngle
        // So: adjV = (altitudeAngle + horizonAngle) / (0.5 * PI)
        float adjV = (altitudeAngle + horizonAngle) / (0.5 * PI);
        
        // Generation mapping:
        // if v < 0.5: adjV = -(1 - 2*v)^2  => v = 0.5 * (1 - sqrt(-adjV))
        // if v >= 0.5: adjV = (2*v - 1)^2  => v = 0.5 * (1 + sqrt(adjV))
        float v;
        if (adjV < 0.0) {
            v = 0.5 * (1.0 - sqrt(clamp(-adjV, 0.0, 1.0)));
        } else {
            v = 0.5 * (1.0 + sqrt(clamp(adjV, 0.0, 1.0)));
        }
        
        // Generation: azimuthAngle = (u - 0.5) * 2 * PI  =>  u = azimuthAngle / (2*PI) + 0.5
        float u = azimuthAngle / (2.0 * PI) + 0.5;
        
        float2 uv = float2(u, v);
        
        // Sky LUT value multiplied by outer space illuminance
        // Sky LUT already contains atmospheric scattering, so we use unattenuated illuminance
        float3 sky_color = sky_view_lut.SampleLevel(linear_sampler, uv, 0).rgb;
        float3 sky_luminance = sky_color * directional_light.get_outer_space_illuminance();
        
        return sky_luminance;
    }
    
    // Sample sky with sun disk (atmospheric scattering + sun disk)
    // Used only for primary rays that miss all geometry
    float3 sample_sky_with_sun_disk(float3 rayDir)
    {
        float3 sky_luminance = sample_sky_lut(rayDir);
        float3 sun_disk = get_sun_disk_luminance(rayDir);
        return sky_luminance + sun_disk;
    }
    
    // Sample directional light with NEE (Next Event Estimation)
    // Returns: xyz = radiance contribution, w = pdf
    float sample_depth_texture_sun_visibility(float3 shadow_position)
    {
        float2 texel_size = 1.0 / float2(max(static_shadow.resolution, uint2(1, 1)));
        float visibility = 0.0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                float2 sample_uv = clamp(
                    shadow_position.xy + float2(x, y) * texel_size,
                    float2(0.0),
                    float2(1.0)
                );
                float baked_depth = static_shadow.depth_texture.SampleLevel(static_shadow.depth_sampler, sample_uv, 0);
                visibility += (shadow_position.z <= baked_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }
        }

        return visibility / 9.0;
    }

    float sample_decompressed_sst_sun_visibility(float3 shadow_position)
    {
        if (static_shadow.sst_enabled == 0)
            return sample_depth_texture_sun_visibility(shadow_position);

        float2 texel_size = 1.0 / float2(max(static_shadow.resolution, uint2(1, 1)));
        float visibility = 0.0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                float2 sample_uv = clamp(
                    shadow_position.xy + float2(x, y) * texel_size,
                    float2(0.0),
                    float2(1.0)
                );
                float baked_depth = static_shadow.sst_decompressed_texture.SampleLevel(static_shadow.depth_sampler, sample_uv, 0);
                visibility += (shadow_position.z <= baked_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }
        }

        return visibility / 9.0;
    }

    float sample_sst_sun_visibility(float3 shadow_position)
    {
        if (static_shadow.sst_enabled == 0 ||
            static_shadow.sst_node_count == 0 ||
            static_shadow.sst_tile_size == 0 ||
            static_shadow.sst_tile_grid.x == 0 ||
            static_shadow.sst_tile_grid.y == 0)
        {
            return sample_packed_sst_sun_visibility(shadow_position);
        }

        uint2 resolution = max(static_shadow.resolution, uint2(1, 1));
        uint pixel_x = min((uint)(shadow_position.x * float(resolution.x)), resolution.x - 1);
        uint pixel_y = min((uint)(shadow_position.y * float(resolution.y)), resolution.y - 1);
        uint tile_x = min(pixel_x / static_shadow.sst_tile_size, static_shadow.sst_tile_grid.x - 1);
        uint tile_y = min(pixel_y / static_shadow.sst_tile_size, static_shadow.sst_tile_grid.y - 1);
        uint root_index = static_shadow.sst_tile_roots[tile_y * static_shadow.sst_tile_grid.x + tile_x];
        if (root_index >= static_shadow.sst_node_count)
            return sample_depth_texture_sun_visibility(shadow_position);

        uint tile_origin_x = tile_x * static_shadow.sst_tile_size;
        uint tile_origin_y = tile_y * static_shadow.sst_tile_size;
        uint tile_extent_x = max(min(static_shadow.sst_tile_size, resolution.x - tile_origin_x), 1);
        uint tile_extent_y = max(min(static_shadow.sst_tile_size, resolution.y - tile_origin_y), 1);
        float local_x = ((float(pixel_x) + 0.5) - float(tile_origin_x)) / float(tile_extent_x);
        float local_y = ((float(pixel_y) + 0.5) - float(tile_origin_y)) / float(tile_extent_y);
        uint tree_bits = is_power_of_two_uint(static_shadow.sst_tile_size) ? log2_power_of_two_uint(static_shadow.sst_tile_size) : 0;

        uint node_index = root_index;
        uint max_steps = min(max(static_shadow.sst_max_traversal_steps, 1), 32);
        [loop]
        for (uint i = 0; i < max_steps; i++)
        {
            if (node_index >= static_shadow.sst_node_count)
                return sample_depth_texture_sun_visibility(shadow_position);

            SSTNode node = static_shadow.sst_nodes[node_index];
            if ((node.flags & SST_NODE_FLAG_PLANE) != 0)
            {
                float encoded_depth = saturate(
                    node.plane_a * (local_x - 0.5) +
                    node.plane_b * (local_y - 0.5) +
                    node.plane_c
                );
                return (shadow_position.z <= encoded_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }

            uint quadrant = decode_tile_morton_quadrant(
                pixel_x,
                pixel_y,
                tile_origin_x,
                tile_origin_y,
                tile_extent_x,
                tile_extent_y,
                local_x,
                local_y,
                tree_bits,
                i
            );
            node_index = node.child_base + quadrant;
            local_x = local_x * 2.0 - ((quadrant & 1) != 0 ? 1.0 : 0.0);
            local_y = local_y * 2.0 - ((quadrant & 2) != 0 ? 1.0 : 0.0);
        }

        return sample_depth_texture_sun_visibility(shadow_position);
    }

    float decode_unorm_bits(uint value, uint max_value)
    {
        return float(value) / float(max_value);
    }

    float decode_snorm_bits(uint value, uint max_value, float value_range)
    {
        return ((float(value) / float(max_value)) * 2.0 - 1.0) * value_range;
    }

    uint decode_packed_branch_offset(uint word0, uint word1, uint level, uint quadrant)
    {
        if (level >= static_shadow.sst_branch_10bit_start_level)
        {
            if (quadrant == 0)
                return (word0 >> 2) & 0x3ff;
            if (quadrant == 1)
                return (word0 >> 12) & 0x3ff;
            if (quadrant == 2)
                return (word0 >> 22) & 0x3ff;
            return word1 & 0x3ff;
        }

        if (quadrant == 0)
            return (word0 >> 2) & 0x1fff;
        if (quadrant == 1)
            return (word0 >> 15) & 0x1fff;
        if (quadrant == 2)
            return ((word0 >> 28) & 0xf) | ((word1 & 0x1ff) << 4);
        return (word1 >> 9) & 0x1fff;
    }

    bool is_power_of_two_uint(uint value)
    {
        return value != 0 && (value & (value - 1)) == 0;
    }

    uint log2_power_of_two_uint(uint value)
    {
        uint bits = 0;
        [loop]
        for (uint i = 0; i < 16; i++)
        {
            if (value <= 1)
                break;
            value >>= 1;
            bits++;
        }
        return bits;
    }

    uint decode_tile_morton_quadrant(
        uint pixel_x,
        uint pixel_y,
        uint tile_origin_x,
        uint tile_origin_y,
        uint tile_extent_x,
        uint tile_extent_y,
        float local_x,
        float local_y,
        uint tree_bits,
        uint level)
    {
        uint fallback_quadrant = (local_x >= 0.5 ? 1 : 0) + (local_y >= 0.5 ? 2 : 0);
        uint tile_size = static_shadow.sst_tile_size;
        if (tree_bits == 0 || tile_extent_x != tile_size || tile_extent_y != tile_size)
            return fallback_quadrant;

        if (level >= tree_bits)
            return fallback_quadrant;

        uint bit_index = tree_bits - 1 - level;
        uint local_pixel_x = min(pixel_x - tile_origin_x, tile_size - 1);
        uint local_pixel_y = min(pixel_y - tile_origin_y, tile_size - 1);
        return ((local_pixel_x >> bit_index) & 1) | (((local_pixel_y >> bit_index) & 1) << 1);
    }

    float sample_packed_sst_sun_visibility(float3 shadow_position)
    {
        if (static_shadow.sst_enabled == 0 ||
            static_shadow.sst_node_count == 0 ||
            static_shadow.sst_tile_size == 0 ||
            static_shadow.sst_tile_grid.x == 0 ||
            static_shadow.sst_tile_grid.y == 0)
        {
            return sample_depth_texture_sun_visibility(shadow_position);
        }

        uint2 resolution = max(static_shadow.resolution, uint2(1, 1));
        uint pixel_x = min((uint)(shadow_position.x * float(resolution.x)), resolution.x - 1);
        uint pixel_y = min((uint)(shadow_position.y * float(resolution.y)), resolution.y - 1);
        uint tile_x = min(pixel_x / static_shadow.sst_tile_size, static_shadow.sst_tile_grid.x - 1);
        uint tile_y = min(pixel_y / static_shadow.sst_tile_size, static_shadow.sst_tile_grid.y - 1);
        uint root_index = static_shadow.sst_tile_roots[tile_y * static_shadow.sst_tile_grid.x + tile_x];
        if (root_index >= static_shadow.sst_node_count)
            return sample_depth_texture_sun_visibility(shadow_position);

        uint tile_origin_x = tile_x * static_shadow.sst_tile_size;
        uint tile_origin_y = tile_y * static_shadow.sst_tile_size;
        uint tile_extent_x = max(min(static_shadow.sst_tile_size, resolution.x - tile_origin_x), 1);
        uint tile_extent_y = max(min(static_shadow.sst_tile_size, resolution.y - tile_origin_y), 1);
        float local_x = ((float(pixel_x) + 0.5) - float(tile_origin_x)) / float(tile_extent_x);
        float local_y = ((float(pixel_y) + 0.5) - float(tile_origin_y)) / float(tile_extent_y);
        uint tree_bits = is_power_of_two_uint(static_shadow.sst_tile_size) ? log2_power_of_two_uint(static_shadow.sst_tile_size) : 0;

        uint node_index = root_index;
        uint max_steps = min(max(static_shadow.sst_max_traversal_steps, 1), 32);
        [loop]
        for (uint i = 0; i < max_steps; i++)
        {
            if (node_index >= static_shadow.sst_node_count)
                return sample_depth_texture_sun_visibility(shadow_position);

            uint2 packed_node = static_shadow.sst_packed_nodes[node_index];
            uint control = packed_node.x & 0x3;
            if (control == 1)
            {
                float encoded_depth = decode_unorm_bits((packed_node.x >> 2) & 0x3fffffff, 0x3fffffff);
                return (shadow_position.z <= encoded_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }
            if (control == 2)
            {
                uint qz = (packed_node.x >> 2) & 0x3fffffff;
                uint qx = packed_node.y & 0xffff;
                uint qy = (packed_node.y >> 16) & 0xffff;
                float a = decode_snorm_bits(qx, 0xffff, 2.0);
                float b = decode_snorm_bits(qy, 0xffff, 2.0);
                float c = decode_unorm_bits(qz, 0x3fffffff);
                float encoded_depth = saturate(a * (local_x - 0.5) + b * (local_y - 0.5) + c);
                return (shadow_position.z <= encoded_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }

            uint quadrant = decode_tile_morton_quadrant(
                pixel_x,
                pixel_y,
                tile_origin_x,
                tile_origin_y,
                tile_extent_x,
                tile_extent_y,
                local_x,
                local_y,
                tree_bits,
                i
            );
            uint relative_offset = decode_packed_branch_offset(packed_node.x, packed_node.y, i, quadrant);
            if (relative_offset == 0)
                return sample_depth_texture_sun_visibility(shadow_position);
            node_index += relative_offset;
            local_x = local_x * 2.0 - ((quadrant & 1) != 0 ? 1.0 : 0.0);
            local_y = local_y * 2.0 - ((quadrant & 2) != 0 ? 1.0 : 0.0);
        }

        return sample_depth_texture_sun_visibility(shadow_position);
    }

    float sample_compact_sst_sun_visibility(float3 shadow_position)
    {
        if (static_shadow.sst_enabled == 0 ||
            static_shadow.sst_compact_word_count == 0 ||
            static_shadow.sst_tile_size == 0 ||
            static_shadow.sst_tile_grid.x == 0 ||
            static_shadow.sst_tile_grid.y == 0)
        {
            return sample_depth_texture_sun_visibility(shadow_position);
        }

        uint2 resolution = max(static_shadow.resolution, uint2(1, 1));
        uint pixel_x = min((uint)(shadow_position.x * float(resolution.x)), resolution.x - 1);
        uint pixel_y = min((uint)(shadow_position.y * float(resolution.y)), resolution.y - 1);
        uint tile_x = min(pixel_x / static_shadow.sst_tile_size, static_shadow.sst_tile_grid.x - 1);
        uint tile_y = min(pixel_y / static_shadow.sst_tile_size, static_shadow.sst_tile_grid.y - 1);
        uint node_word_offset = static_shadow.sst_compact_roots[tile_y * static_shadow.sst_tile_grid.x + tile_x];
        if (node_word_offset >= static_shadow.sst_compact_word_count)
            return sample_packed_sst_sun_visibility(shadow_position);

        uint tile_origin_x = tile_x * static_shadow.sst_tile_size;
        uint tile_origin_y = tile_y * static_shadow.sst_tile_size;
        uint tile_extent_x = max(min(static_shadow.sst_tile_size, resolution.x - tile_origin_x), 1);
        uint tile_extent_y = max(min(static_shadow.sst_tile_size, resolution.y - tile_origin_y), 1);
        float local_x = ((float(pixel_x) + 0.5) - float(tile_origin_x)) / float(tile_extent_x);
        float local_y = ((float(pixel_y) + 0.5) - float(tile_origin_y)) / float(tile_extent_y);
        uint tree_bits = is_power_of_two_uint(static_shadow.sst_tile_size) ? log2_power_of_two_uint(static_shadow.sst_tile_size) : 0;

        uint max_steps = min(max(static_shadow.sst_max_traversal_steps, 1), 32);
        [loop]
        for (uint i = 0; i < max_steps; i++)
        {
            if (node_word_offset >= static_shadow.sst_compact_word_count)
                return sample_packed_sst_sun_visibility(shadow_position);

            uint word0 = static_shadow.sst_compact_words[node_word_offset];
            uint control = word0 & 0x3;
            if (control == 1)
            {
                float encoded_depth = decode_unorm_bits((word0 >> 2) & 0x3fffffff, 0x3fffffff);
                return (shadow_position.z <= encoded_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }

            if (node_word_offset + 1 >= static_shadow.sst_compact_word_count)
                return sample_packed_sst_sun_visibility(shadow_position);

            uint word1 = static_shadow.sst_compact_words[node_word_offset + 1];
            if (control == 2)
            {
                uint qz = (word0 >> 2) & 0x3fffffff;
                uint qx = word1 & 0xffff;
                uint qy = (word1 >> 16) & 0xffff;
                float a = decode_snorm_bits(qx, 0xffff, 2.0);
                float b = decode_snorm_bits(qy, 0xffff, 2.0);
                float c = decode_unorm_bits(qz, 0x3fffffff);
                float encoded_depth = saturate(a * (local_x - 0.5) + b * (local_y - 0.5) + c);
                return (shadow_position.z <= encoded_depth + static_shadow.depth_bias) ? 1.0 : 0.0;
            }

            uint quadrant = decode_tile_morton_quadrant(
                pixel_x,
                pixel_y,
                tile_origin_x,
                tile_origin_y,
                tile_extent_x,
                tile_extent_y,
                local_x,
                local_y,
                tree_bits,
                i
            );
            uint relative_offset = decode_packed_branch_offset(word0, word1, i, quadrant);
            if (relative_offset == 0)
                return sample_packed_sst_sun_visibility(shadow_position);
            node_word_offset += relative_offset;
            local_x = local_x * 2.0 - ((quadrant & 1) != 0 ? 1.0 : 0.0);
            local_y = local_y * 2.0 - ((quadrant & 2) != 0 ? 1.0 : 0.0);
        }

        return sample_packed_sst_sun_visibility(shadow_position);
    }

    float sample_compact_sst_sun_visibility_pcf3(float3 shadow_position)
    {
        float2 texel_size = 1.0 / float2(max(static_shadow.resolution, uint2(1, 1)));
        float visibility = 0.0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                float3 sample_position = shadow_position;
                sample_position.xy = clamp(
                    shadow_position.xy + float2(x, y) * texel_size,
                    float2(0.0),
                    float2(1.0)
                );
                visibility += sample_compact_sst_sun_visibility(sample_position);
            }
        }
        return visibility / 9.0;
    }

    float sample_static_sun_visibility(ShadingData sd, float3 light_dir)
    {
        Ray shadow_ray = Ray(sd.compute_new_ray_origin(), light_dir);
        if (static_shadow.enabled == 0 || static_shadow.shadow_mode == SHADOW_MODE_REALTIME)
            return get_visibility(shadow_ray);

        float4 shadow_position_h = mul(static_shadow.world_to_shadow, float4(shadow_ray.origin, 1.0));
        float3 shadow_position = shadow_position_h.xyz / max(abs(shadow_position_h.w), 1e-6);

        if (shadow_position.x < 0.0 || shadow_position.x > 1.0 ||
            shadow_position.y < 0.0 || shadow_position.y > 1.0 ||
            shadow_position.z < 0.0 || shadow_position.z > 1.0)
        {
            return get_visibility(shadow_ray);
        }

        if (static_shadow.shadow_mode == SHADOW_MODE_DECOMPRESSED_SST)
            return sample_decompressed_sst_sun_visibility(shadow_position);

        if (static_shadow.shadow_mode == SHADOW_MODE_COMPACT_SST_PCF3)
            return sample_compact_sst_sun_visibility_pcf3(shadow_position);

        if (static_shadow.shadow_mode == SHADOW_MODE_COMPACT_SST)
            return sample_compact_sst_sun_visibility(shadow_position);

        if (static_shadow.shadow_mode == SHADOW_MODE_PACKED_SST)
            return sample_packed_sst_sun_visibility(shadow_position);

        if (static_shadow.shadow_mode == SHADOW_MODE_SST)
            return sample_sst_sun_visibility(shadow_position);

        return sample_depth_texture_sun_visibility(shadow_position);
    }

    float3 sample_directional_irradiance(ShadingData sd, float2 rand_sample)
    {
        float4 sample_result = sample_cone_direction(
            directional_light.direction,
            directional_light.cos_half_angle,
            rand_sample);
        float3 light_direction = sample_result.xyz;
        float visibility = sample_static_sun_visibility(sd, light_direction);
        float n_dot_l = max(0.0, dot(sd.shading_frame.n, light_direction));
        float3 illuminance =
            directional_light.get_outer_space_illuminance() * get_sun_transmittance_at_view();
        return visibility * illuminance * n_dot_l;
    }

    float4 sample_directional_light_with_visibility(
        ShadingData sd,
        float2 rand_sample,
        float visibility_override,
        uint use_visibility_override)
    {
        // Sample direction within sun disk cone
        float4 sample_result = sample_cone_direction(
            directional_light.direction, 
            directional_light.cos_half_angle, 
            rand_sample
        );
        float3 L_dir = sample_result.xyz;
        float pdf = sample_result.w;
        
        // Visibility test
        float visibility = use_visibility_override != 0
            ? saturate(visibility_override)
            : sample_static_sun_visibility(sd, L_dir);
        
        // Lambertian BRDF evaluation
        float NdotL = max(0.0, dot(sd.shading_frame.n, L_dir));
        float3 f = sd.diffuse * INV_PI;
        
        // For directional light with cone sampling:
        // - Sun radiance = illuminance / solid_angle (where solid_angle = 2π(1-cos_half_angle))
        // - pdf = 1 / solid_angle
        // - Li / pdf = (illuminance / solid_angle) / (1 / solid_angle) = illuminance
        // So we DON'T divide by pdf for directional lights!
        // 
        // Use ground-level illuminance which includes atmospheric transmittance
        // (light is attenuated as it travels through the atmosphere to reach the surface)
        float3 illuminance = directional_light.get_outer_space_illuminance() * get_sun_transmittance_at_view();
        float3 radiance = visibility * f * illuminance * NdotL;
        
        return float4(radiance, pdf);
    }

    float4 sample_directional_light(ShadingData sd, float2 rand_sample)
    {
        return sample_directional_light_with_visibility(sd, rand_sample, 1.0, 0);
    }

    float3 get_sun_transmittance_at_view()
    {
        float height = length(viewPos);
        float3 up = viewPos / height;
        float sunCosZenithAngle = dot(normalize(directional_light.direction), up);

        float2 uv = float2(
            clamp(0.5 + 0.5 * sunCosZenithAngle, 0.0, 1.0),
            clamp((height - groundRadiusMM) / (atmosphereRadiusMM - groundRadiusMM), 0.0, 1.0)
        );

        return transmittance_lut.SampleLevel(transmittance_sampler, uv, 0).rgb;
    }

    float get_linear_depth(float3 world_position)
    {
        float4 view = mul(camera.view_matrix, float4(world_position, 1.0f));
        return -view.z;
    }

    HitInfo intersect(Ray ray)
    {
        HitInfo hit = {};

        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed())
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                // Get candidate hit info
                uint instance_id = q.CandidateInstanceIndex();
                uint primitive_index = q.CandidatePrimitiveIndex();
                float2 bary = q.CandidateTriangleBarycentrics();
                
                // Get material and vertex for alpha test
                InstanceDesc instance_desc = g_scene.instance_descs[instance_id];
                MaterialDesc material = load_material_desc(instance_desc.material_id);
                
                // Skip alpha test for opaque materials
                if (material.alpha_mode == ALPHA_MODE_OPAQUE)
                {
                    q.CommitNonOpaqueTriangleHit();
                    continue;
                }
                
                // Get UV coordinates for texture sampling
                Vertex v = get_vertex(instance_id, primitive_index, bary);
                float2 uv = v.uv;
                
                // Get world position for stochastic alpha
                float3 world_pos = v.position;
                
                // Sample opacity from texture
                float opacity = get_opacity(material, uv);
                
                // Evaluate alpha test
                if (eval_alpha_test(material, opacity, world_pos, frame_index))
                {
                    q.CommitNonOpaqueTriangleHit();
                }
            }
        }
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hit = {
                true,
                q.CommittedTriangleBarycentrics(),
                q.CommittedInstanceIndex(),
                q.CommittedPrimitiveIndex(),
            };
        }
        return hit;
    }

    HitInfo intersect(Ray ray, out float t)
    {
        HitInfo hit = {};

        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed())
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                // Get candidate hit info
                uint instance_id = q.CandidateInstanceIndex();
                uint primitive_index = q.CandidatePrimitiveIndex();
                float2 bary = q.CandidateTriangleBarycentrics();
                
                // Get material and vertex for alpha test
                InstanceDesc instance_desc = g_scene.instance_descs[instance_id];
                MaterialDesc material = load_material_desc(instance_desc.material_id);
                
                // Skip alpha test for opaque materials
                if (material.alpha_mode == ALPHA_MODE_OPAQUE)
                {
                    q.CommitNonOpaqueTriangleHit();
                    continue;
                }
                
                // Get UV coordinates for texture sampling
                Vertex v = get_vertex(instance_id, primitive_index, bary);
                float2 uv = v.uv;
                
                // Get world position for stochastic alpha
                float3 world_pos = v.position;
                
                // Sample opacity from texture
                float opacity = get_opacity(material, uv);
                
                // Evaluate alpha test
                if (eval_alpha_test(material, opacity, world_pos, frame_index))
                {
                    q.CommitNonOpaqueTriangleHit();
                }
            }
        }
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hit = {
                true,
                q.CommittedTriangleBarycentrics(),
                q.CommittedInstanceIndex(),
                q.CommittedPrimitiveIndex(),
            };
            t = q.CommittedRayT();
        }
        return hit;
    }

    float get_visibility(Ray ray)
    {
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;

        rayQuery.TraceRayInline(g_scene.tlas, RAY_FLAG_NONE, 0xff, ray.to_ray_desc());

        while (rayQuery.Proceed())
        {
            if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                // Get candidate hit info
                uint instance_id = rayQuery.CandidateInstanceIndex();
                uint primitive_index = rayQuery.CandidatePrimitiveIndex();
                float2 bary = rayQuery.CandidateTriangleBarycentrics();
                
                // Get material for alpha test
                InstanceDesc instance_desc = g_scene.instance_descs[instance_id];
                MaterialDesc material = load_material_desc(instance_desc.material_id);
                
                // Opaque materials block visibility
                if (material.alpha_mode == ALPHA_MODE_OPAQUE)
                {
                    rayQuery.CommitNonOpaqueTriangleHit();
                    continue;
                }
                
                // Get UV coordinates for texture sampling
                Vertex v = get_vertex(instance_id, primitive_index, bary);
                float2 uv = v.uv;
                
                // Get world position for stochastic alpha
                float3 world_pos = v.position;
                
                // Sample opacity from texture
                float opacity = get_opacity(material, uv);
                
                // Evaluate alpha test - if passed, ray is blocked
                if (eval_alpha_test(material, opacity, world_pos, frame_index))
                {
                    rayQuery.CommitNonOpaqueTriangleHit();
                }
            }
        }

        bool visible = (rayQuery.CommittedStatus() == COMMITTED_NOTHING);
        return visible ? 1.0 : 0.0;
    }
};
