import common;
import camera;
import software_rt;

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
    // Software RT BVH buffers (replaces hardware RaytracingAccelerationStructure)
    StructuredBuffer<uint> tlas_nodes;           // raw uint buffer, 8 uints per BVH node
    StructuredBuffer<uint> tlas_instance_order;
    StructuredBuffer<uint> blas_nodes;           // raw uint buffer, 8 uints per BVH node
    StructuredBuffer<uint> blas_offsets;
    StructuredBuffer<uint> blas_prim_indices;
    StructuredBuffer<uint> blas_prim_offsets;
    StructuredBuffer<float4x4> inverse_transforms;  // for ray transform to object space

    StructuredBuffer<MaterialDesc> material_descs;
    StructuredBuffer<float> raw_material_descs;    // 16 floats per MaterialDesc (avoids struct padding)
    
    StructuredBuffer<MeshDesc> mesh_descs;
    StructuredBuffer<InstanceDesc> instance_descs;
    StructuredBuffer<Vertex> vertices;
    StructuredBuffer<uint> indices;

    // Raw buffer views for software RT (avoids struct padding issues on Metal)
    StructuredBuffer<float> raw_vertices;         // 8 floats per vertex
    StructuredBuffer<uint> raw_mesh_descs;         // 4 uints per MeshDesc
    StructuredBuffer<uint> raw_instance_descs;     // 4 uints per InstanceDesc

    StructuredBuffer<float4x4> transforms;
    StructuredBuffer<float4x4> inverse_transpose_transforms;

    Texture2D<float4> transmittance_lut;
    SamplerState transmittance_sampler;

    Texture2D<float4> sky_view_lut;
    SamplerState linear_sampler;
    float3 sun_direction;
    
    DirectionalLight directional_light;

    Camera camera;

    uint instance_count;
    uint frame_index;
    uint pad1;
    uint pad2;

    // Load MaterialDesc directly from StructuredBuffer (for texture handles)
    MaterialDesc load_material_desc(uint index)
    {
        return material_descs[index];
    }

    // Load basic material data from raw float buffer, patching over struct-loaded MaterialDesc
    // This avoids struct padding issues on Metal for the scalar/vector fields
    // Raw layout: 16 floats per material
    // [0-2] base_color, [3] flags(uint), [4-6] emissive, [7] shading_model(uint),
    // [8] roughness, [9] metallic, [10] texture_flags(uint), [11] alpha_mode(uint),
    // [12] alpha_cutoff, [13-15] specular_color
    MaterialDesc load_material_desc_safe(uint index)
    {
        MaterialDesc mat = material_descs[index];  // get texture handles from struct buffer
        // Override scalar/vector fields from raw buffer to avoid padding issues
        uint base = index * 16;
        mat.base_color = float3(raw_material_descs[base + 0], raw_material_descs[base + 1], raw_material_descs[base + 2]);
        mat.flags = asuint(raw_material_descs[base + 3]);
        mat.emissive = float3(raw_material_descs[base + 4], raw_material_descs[base + 5], raw_material_descs[base + 6]);
        mat.shading_model = asuint(raw_material_descs[base + 7]);
        mat.roughness = raw_material_descs[base + 8];
        mat.metallic = raw_material_descs[base + 9];
        mat.texture_flags = asuint(raw_material_descs[base + 10]);
        mat.alpha_mode = asuint(raw_material_descs[base + 11]);
        mat.alpha_cutoff = raw_material_descs[base + 12];
        mat.specular_color = float3(raw_material_descs[base + 13], raw_material_descs[base + 14], raw_material_descs[base + 15]);
        return mat;
    }


    MaterialDesc get_material(HitInfo hit_info)
    {
        // Use raw buffer to avoid struct padding issues on Metal
        uint inst_base = hit_info.instance_id * 4;
        uint material_id = raw_instance_descs[inst_base + 1];
        return load_material_desc_safe(material_id);
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

    // Helper: read InstanceDesc from raw buffer (4 uints per instance)
    void load_raw_instance(uint instance_id, out uint mesh_id, out uint material_id, out uint transform_id, out uint instance_flags)
    {
        uint base = instance_id * 4;
        mesh_id = raw_instance_descs[base + 0];
        material_id = raw_instance_descs[base + 1];
        transform_id = raw_instance_descs[base + 2];
        instance_flags = raw_instance_descs[base + 3];
    }

    // Helper: read MeshDesc from raw buffer (4 uints per mesh)
    void load_raw_mesh(uint mesh_id, out uint vertex_count, out uint index_count, out uint vertex_offset, out uint index_offset)
    {
        uint base = mesh_id * 4;
        vertex_count = raw_mesh_descs[base + 0];
        index_count = raw_mesh_descs[base + 1];
        vertex_offset = raw_mesh_descs[base + 2];
        index_offset = raw_mesh_descs[base + 3];
    }

    // Helper: read Vertex from raw float buffer (8 floats per vertex: pos.xyz, normal.xyz, uv.xy)
    Vertex load_raw_vertex(uint vertex_index)
    {
        uint base = vertex_index * 8;
        Vertex v;
        v.position = float3(raw_vertices[base + 0], raw_vertices[base + 1], raw_vertices[base + 2]);
        v.normal   = float3(raw_vertices[base + 3], raw_vertices[base + 4], raw_vertices[base + 5]);
        v.uv       = float2(raw_vertices[base + 6], raw_vertices[base + 7]);
        return v;
    }

    Vertex get_vertex(HitInfo hit_info)
    {
        return get_vertex(hit_info.instance_id, hit_info.primitive_index, hit_info.bary);
    }

    Vertex get_vertex(uint instance_id, uint primitive_index, float2 bary)
    {
        // Use raw buffers to avoid struct padding issues on Metal
        uint mesh_id, material_id, transform_id, instance_flags;
        load_raw_instance(instance_id, mesh_id, material_id, transform_id, instance_flags);
        
        uint vertex_count, index_count, vertex_offset, index_offset;
        load_raw_mesh(mesh_id, vertex_count, index_count, vertex_offset, index_offset);
        
        float4x4 transform = g_scene.transforms[transform_id];
        float4x4 inverse_transpose_transform = g_scene.inverse_transpose_transforms[transform_id];
        uint i0 = g_scene.indices[index_offset + primitive_index * 3 + 0];
        uint i1 = g_scene.indices[index_offset + primitive_index * 3 + 1];
        uint i2 = g_scene.indices[index_offset + primitive_index * 3 + 2];
        Vertex v0 = load_raw_vertex(vertex_offset + i0);
        Vertex v1 = load_raw_vertex(vertex_offset + i1);
        Vertex v2 = load_raw_vertex(vertex_offset + i2);
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

        // Use raw buffers to avoid struct padding issues on Metal
        uint mesh_id, material_id_unused, transform_id, instance_flags;
        load_raw_instance(instance_id, mesh_id, material_id_unused, transform_id, instance_flags);
        
        uint vertex_count, index_count, vertex_offset, index_offset;
        load_raw_mesh(mesh_id, vertex_count, index_count, vertex_offset, index_offset);
        
        float4x4 transform = g_scene.transforms[transform_id];
        float4x4 inverse_transpose_transform = g_scene.inverse_transpose_transforms[transform_id];
        uint i0 = g_scene.indices[index_offset + primitive_index * 3 + 0];
        uint i1 = g_scene.indices[index_offset + primitive_index * 3 + 1];
        uint i2 = g_scene.indices[index_offset + primitive_index * 3 + 2];
        Vertex v0 = load_raw_vertex(vertex_offset + i0);
        Vertex v1 = load_raw_vertex(vertex_offset + i1);
        Vertex v2 = load_raw_vertex(vertex_offset + i2);
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
        bool double_sided = true; // material.is_double_sided();
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
    float4 sample_directional_light(ShadingData sd, float2 rand_sample)
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
        Ray shadow_ray = Ray(sd.compute_new_ray_origin(), L_dir);
        float visibility = get_visibility(shadow_ray);
        
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
        float4 clip = mul(camera.view_proj_matrix_no_jitter, float4(world_position, 1.0f));
        return clip.z / clip.w;
    }

    // Helper to build SWRTSceneBuffers from Scene fields
    SWRTSceneBuffers _build_swrt_scene()
    {
        SWRTSceneBuffers s;
        s.tlas_nodes = tlas_nodes;
        s.tlas_instance_order = tlas_instance_order;
        s.blas_nodes = blas_nodes;
        s.blas_offsets = blas_offsets;
        s.blas_prim_indices = blas_prim_indices;
        s.blas_prim_offsets = blas_prim_offsets;
        s.raw_vertices = raw_vertices;
        s.indices = indices;
        s.raw_mesh_descs = raw_mesh_descs;
        s.raw_instance_descs = raw_instance_descs;
        s.transforms = transforms;
        s.inverse_transforms = inverse_transforms;
        s.instance_count = instance_count;
        return s;
    }

    HitInfo intersect(Ray ray)
    {
        HitInfo hit = {};

        SWRTSceneBuffers swrt_scene = _build_swrt_scene();

        SWRT_RayQuery rq;
        swrt_trace_ray_inline(rq, ray.origin, ray.dir, ray.t_min, ray.t_max);
        swrt_proceed(rq, swrt_scene);

        if (swrt_committed_status(rq) == SWRT_COMMITTED_TRIANGLE_HIT)
        {
            hit = {
                true,
                swrt_committed_triangle_barycentrics(rq),
                swrt_committed_instance_index(rq),
                swrt_committed_primitive_index(rq),
            };
        }
        return hit;
    }

    HitInfo intersect(Ray ray, out float t)
    {
        HitInfo hit = {};

        SWRTSceneBuffers swrt_scene = _build_swrt_scene();

        SWRT_RayQuery rq;
        swrt_trace_ray_inline(rq, ray.origin, ray.dir, ray.t_min, ray.t_max);
        swrt_proceed(rq, swrt_scene);

        if (swrt_committed_status(rq) == SWRT_COMMITTED_TRIANGLE_HIT)
        {
            hit = {
                true,
                swrt_committed_triangle_barycentrics(rq),
                swrt_committed_instance_index(rq),
                swrt_committed_primitive_index(rq),
            };
            t = swrt_committed_ray_t(rq);
        }
        return hit;
    }

    float get_visibility(Ray ray)
    {
        SWRTSceneBuffers swrt_scene = _build_swrt_scene();
        return swrt_get_visibility(ray.origin, ray.dir, ray.t_min, ray.t_max, swrt_scene);
    }
};
