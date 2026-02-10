"""
Motion vector reprojection test.

Verifies that get_motion() produces correct 2.5D motion vectors by:
  A) Confirming near-zero motion when camera is static.
  B) Comparing GPU motion vectors against CPU-computed expected values.
  C) Tracking a surface point across frames with camera movement and
     accumulating color via reprojection to demonstrate stability.

Usage:
    python test_motion.py
"""

import slangpy as spy
import numpy as np
from pathlib import Path
from scene_node import SceneNode
from scene import Scene
from camera import Camera, CameraController
from event_dispatcher import SyncEventDispatcher

PROJECT_DIR = Path(__file__).parent

WIDTH = 256
HEIGHT = 256
NUM_WARMUP_FRAMES = 2    # Frames with static camera
NUM_MOVING_FRAMES = 20   # Frames with camera movement
CAMERA_STEP = 0.02       # Per-frame camera translation (rightward)
TEST_PIXEL = (WIDTH // 2, HEIGHT // 2)  # Center pixel


def create_texture(device, w, h, fmt=spy.Format.rgba32_float):
    return device.create_texture(
        format=fmt, width=w, height=h,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )


def read_pixel(texture, px, py):
    """Read back a single pixel as numpy array (float32 x 4)."""
    bmp = texture.to_bitmap()
    arr = np.array(bmp, copy=False).reshape(texture.height, texture.width, -1)
    return arr[py, px].astype(np.float32)


def mat4_to_numpy(mat):
    """Convert slangpy float4x4 to numpy (4,4) float64 array."""
    return mat.to_numpy().astype(np.float64)


def numpy_world_to_screen_uv(view_proj, X):
    """CPU reference: project world position to [0,1] screen UV (DX convention)."""
    m = mat4_to_numpy(view_proj)
    p = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    clip = m @ p
    ndc = clip[:2] / clip[3]
    uv = ndc * np.array([0.5, -0.5]) + 0.5
    return uv


def numpy_view_z(view_mat, X):
    """CPU reference: compute linear view-space Z."""
    m = mat4_to_numpy(view_mat)
    p = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    return (m @ p)[2]


def main():
    print("=" * 60)
    print("Motion Vector Reprojection Test")
    print("=" * 60)

    # ---- setup ----
    device = spy.Device(
        enable_debug_layers=False,
        enable_print=True,
        compiler_options={
            "include_paths": [PROJECT_DIR],
            "defines": {"USE_RAYTRACING_PIPELINE": "0", "HEADLESS_MODE": "1"},
        },
    )

    scene_node = SceneNode.demo()
    event_dispatcher = SyncEventDispatcher()
    scene = Scene(device, scene_node, event_dispatcher)

    program = device.load_program("test_motion.slang", ["compute_main"])
    pipeline = device.create_compute_pipeline(program)

    world_pos_tex = create_texture(device, WIDTH, HEIGHT)
    motion_tex = create_texture(device, WIDTH, HEIGHT)
    color_tex = create_texture(device, WIDTH, HEIGHT)

    def dispatch_frame(frame_idx):
        """Run one frame of the motion test shader."""
        scene.camera.begin_frame(WIDTH, HEIGHT)
        scene.update()

        cmd = device.create_command_encoder()
        with cmd.begin_compute_pass() as p:
            so = p.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(so)
            cursor.g_world_pos = world_pos_tex
            cursor.g_motion = motion_tex
            cursor.g_color = color_tex
            cursor.g_frame = frame_idx
            scene.bind(cursor.g_scene)
            p.dispatch(thread_count=[WIDTH, HEIGHT, 1])
        device.submit_command_buffer(cmd.finish())
        device.wait()

    # ------------------------------------------------------------------
    # Verification A: static camera – motion should be ~zero
    # ------------------------------------------------------------------
    print("\n--- Verification A: Static camera ---")
    for f in range(NUM_WARMUP_FRAMES):
        dispatch_frame(f)

    wp = read_pixel(world_pos_tex, *TEST_PIXEL)
    mv = read_pixel(motion_tex, *TEST_PIXEL)
    print(f"  Test pixel {TEST_PIXEL}: world_pos = {wp[:3]}, hit = {wp[3]:.0f}")
    print(f"  Motion vector: xy = ({mv[0]:.6f}, {mv[1]:.6f}) px, dz = {mv[2]:.6f}")

    static_ok = True
    if wp[3] > 0.5:
        if abs(mv[0]) > 0.01 or abs(mv[1]) > 0.01:
            print("  [WARN] Motion > 0.01 px with static camera!")
            static_ok = False
        else:
            print("  [PASS] Motion near zero with static camera.")
    else:
        print("  [SKIP] Test pixel missed geometry (sky).")

    # ------------------------------------------------------------------
    # Verification B: moving camera – compare GPU vs CPU motion
    # ------------------------------------------------------------------
    print("\n--- Verification B: Moving camera (CPU vs GPU) ---")

    # First frame at current position (to establish prev matrices)
    dispatch_frame(NUM_WARMUP_FRAMES)
    wp_before = read_pixel(world_pos_tex, *TEST_PIXEL)
    X = wp_before[:3]

    # Move camera rightward.
    # IMPORTANT: only set position/target, do NOT call recompute() here.
    # begin_frame (inside dispatch_frame) will call recompute() once, which
    # correctly saves the previous frame's matrices as prev and computes
    # new matrices at the moved position.
    cam = scene.camera
    fwd = spy.math.normalize(cam.target - cam.position)
    right = spy.math.normalize(spy.math.cross(fwd, spy.float3(0, 1, 0)))
    cam.position = cam.position + right * CAMERA_STEP
    cam.target = cam.target + right * CAMERA_STEP

    # Dispatch next frame (camera has moved, begin_frame handles matrices)
    dispatch_frame(NUM_WARMUP_FRAMES + 1)

    wp_after = read_pixel(world_pos_tex, *TEST_PIXEL)
    mv_gpu = read_pixel(motion_tex, *TEST_PIXEL)

    # Find the world position that the test pixel sees NOW (after move)
    X_now = wp_after[:3]

    if wp_after[3] > 0.5:
        # CPU reference: compute expected motion for the point X_now (static, so Xprev = X_now)
        uv_curr = numpy_world_to_screen_uv(scene.camera.view_proj_matrix_no_jitter, X_now)
        uv_prev = numpy_world_to_screen_uv(scene.camera.prev_view_proj_matrix_no_jitter, X_now)
        expected_xy = (uv_prev - uv_curr) * np.array([WIDTH, HEIGHT])

        vz_curr = numpy_view_z(scene.camera.view_matrix, X_now)
        vz_prev = numpy_view_z(scene.camera.prev_view_matrix, X_now)
        expected_z = vz_prev - vz_curr

        print(f"  World pos (current pixel): {X_now}")
        print(f"  GPU  motion: xy = ({mv_gpu[0]:.4f}, {mv_gpu[1]:.4f}) px, dz = {mv_gpu[2]:.6f}")
        print(f"  CPU  motion: xy = ({expected_xy[0]:.4f}, {expected_xy[1]:.4f}) px, dz = {expected_z:.6f}")

        err_xy = np.linalg.norm(mv_gpu[:2] - expected_xy)
        err_z = abs(mv_gpu[2] - expected_z)
        print(f"  Error: |xy| = {err_xy:.6f} px, |z| = {err_z:.6f}")

        if err_xy < 0.5 and err_z < 0.01:
            print("  [PASS] GPU motion matches CPU reference.")
        else:
            print("  [FAIL] GPU motion differs from CPU reference!")
    else:
        print("  [SKIP] Test pixel missed geometry after move.")

    # ------------------------------------------------------------------
    # Verification C: reprojection accumulation stability
    # ------------------------------------------------------------------
    print("\n--- Verification C: Reprojection accumulation ---")

    # Reset camera to demo position
    scene_node2 = SceneNode.demo()
    cam = scene_node2.camera
    scene.camera.position = cam.position
    scene.camera.target = cam.target
    scene.camera.up = cam.up
    scene.camera.fov = cam.fov
    scene.camera._has_prev_matrices = False
    scene.camera.frame_index = 0

    # Warm-up frame to establish prev matrices
    dispatch_frame(0)

    accumulated_color = np.zeros(3, dtype=np.float64)
    sample_count = 0
    # Track the current floating-point pixel position
    tracked_px = np.array([TEST_PIXEL[0] + 0.5, TEST_PIXEL[1] + 0.5], dtype=np.float64)
    colors_per_frame = []

    for f in range(1, NUM_MOVING_FRAMES + 1):
        # Move camera rightward each frame (no manual recompute - begin_frame handles it)
        fwd = spy.math.normalize(scene.camera.target - scene.camera.position)
        right = spy.math.normalize(spy.math.cross(fwd, spy.float3(0, 1, 0)))
        scene.camera.position = scene.camera.position + right * CAMERA_STEP
        scene.camera.target = scene.camera.target + right * CAMERA_STEP

        dispatch_frame(f)

        # Read motion at tracked pixel (integer coords)
        ipx = int(np.clip(np.round(tracked_px[0] - 0.5), 0, WIDTH - 1))
        ipy = int(np.clip(np.round(tracked_px[1] - 0.5), 0, HEIGHT - 1))

        mv = read_pixel(motion_tex, ipx, ipy)
        col = read_pixel(color_tex, ipx, ipy)
        wp = read_pixel(world_pos_tex, ipx, ipy)

        if wp[3] > 0.5:
            accumulated_color += col[:3].astype(np.float64)
            sample_count += 1
            colors_per_frame.append(col[:3].copy())
        else:
            colors_per_frame.append(np.zeros(3))

        # Advance tracked pixel using motion vector (motion = prevUV - currUV in pixels)
        # To go from current frame to next frame, we DON'T apply this motion;
        # the motion tells us where the current point WAS in the previous frame.
        # For tracking: after the next frame's camera moves, the motion vector at the
        # new pixel position will tell us the shift. We use the CURRENT motion to
        # shift our tracking: the point at (tracked_px) in this frame was at
        # (tracked_px + motion.xy) in the previous frame. So for next frame,
        # the point will move by roughly -motion.xy (opposite direction).
        # Actually: motion = prevUV - currUV, so current pixel -> prev pixel by +motion.
        # For next frame, the same world point will be at currPixel - motion (approximately).
        tracked_px[0] -= mv[0]
        tracked_px[1] -= mv[1]

    if sample_count > 0:
        avg_color = accumulated_color / sample_count
        print(f"  Tracked pixel across {NUM_MOVING_FRAMES} frames ({sample_count} valid samples)")
        print(f"  Final tracked position: ({tracked_px[0]:.1f}, {tracked_px[1]:.1f})")
        print(f"  Accumulated average color: ({avg_color[0]:.4f}, {avg_color[1]:.4f}, {avg_color[2]:.4f})")

        # Check stability: variance of colors should be small relative to mean
        if sample_count >= 3:
            colors = np.array(colors_per_frame[:sample_count])
            variance = np.var(colors, axis=0)
            mean = np.mean(colors, axis=0)
            # coefficient of variation (relative stddev)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = np.where(mean > 1e-6, np.sqrt(variance) / mean, 0.0)
            avg_cv = np.mean(cv)
            print(f"  Color coefficient of variation (avg): {avg_cv:.4f}")
            if avg_cv < 1.0:
                print("  [PASS] Color accumulation is stable across frames.")
            else:
                print("  [WARN] High color variance – reprojection may be drifting.")
    else:
        print("  [SKIP] No valid samples collected.")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
