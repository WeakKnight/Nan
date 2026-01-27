import common;

struct Camera {
    float4x4 view_matrix;
    float4x4 proj_matrix;
    float4x4 view_proj_matrix;
    float4x4 inv_view_proj_matrix;
    float4x4 proj_matrix_no_jitter;
    float4x4 view_proj_matrix_no_jitter;
    float4x4 prev_view_proj_matrix;
    float4x4 prev_view_proj_matrix_no_jitter;

    float3 position;
    float jitter_x;
    float3 image_u;
    float jitter_y;
    float3 image_v;
    float near_clip_plane;
    float3 image_w;
    float far_clip_plane;
    float prev_jitter_x;
    float prev_jitter_y;

    /** Computes a camera ray for a given pixel assuming a pinhole camera model.
    The camera jitter is taken into account to compute the sample position on the image plane.
    \param[in] pixel Pixel coordinates with origin in top-left.
    \param[in] frameDim Image plane dimensions in pixels.
    \param[in] applyJitter true if jitter should be applied else false.
    \return Returns the camera ray.
    */
    Ray compute_ray_pinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        Ray ray;

        // Compute the normalized ray direction assuming a pinhole camera.
        ray.origin = position;
        ray.dir = normalize(compute_non_normalized_ray_dir_pinhole(pixel, frameDim, applyJitter));

        float invCos = 1.f / dot(normalize(image_w), ray.dir);
        ray.t_min = near_clip_plane * invCos;
        ray.t_max = far_clip_plane * invCos;

        return ray;
    }

    /** Computes the primary ray's direction, non-normalized assuming pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter True if jitter should be applied else false.
        \return Returns the non-normalized ray direction
    */
    float3 compute_non_normalized_ray_dir_pinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim;
        if (applyJitter) 
        {
            p += float2(-jitter_x, jitter_y);
        }
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the non-normalized ray direction assuming a pinhole camera.
        return ndc.x * image_u + ndc.y * image_v + image_w;
    }
};