import slangpy as spy
from slangpy.math import Handedness

def halton(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0
    while index > 0:
        factor /= base
        result += factor * (index % base)
        index //= base
    return result

class HaltonSamplePattern:
    def __init__(self, sample_count: int):
        self.sample_count = sample_count
        self.current_sample = 0

    def next(self):
        value = spy.float2(halton(self.current_sample, 2), halton(self.current_sample, 3))
        self.current_sample += 1
        if self.sample_count != 0:
            self.current_sample = self.current_sample % self.sample_count
        return spy.math.frac(value)

class Camera:
    def __init__(self):
        super().__init__()
        self.width = 100
        self.height = 100
        self.aspect_ratio = 1.0
        self.position = spy.float3(1, 1, 1)
        self.target = spy.float3(0, 0, 0)
        self.up = spy.float3(0, 1, 0)
        self.fov = 70.0
        self.near_clip_plane = 0.03
        self.far_clip_plane = 1000
        self.focal_distance = 10000
        self.sample_pattern = HaltonSamplePattern(16)
        self.jitter = spy.float2(0, 0)
        self.prev_jitter = spy.float2(0, 0)
        self._has_prev_matrices = False
        self.frame_index = 0
        self.recompute()

    def recompute(self):
        previous_view_proj_matrix_no_jitter = getattr(self, "view_proj_matrix_no_jitter", None) if self._has_prev_matrices else None
        previous_view_proj_matrix = getattr(self, "view_proj_matrix", None) if self._has_prev_matrices else None

        self.aspect_ratio = float(self.width) / float(self.height)

        self.fwd = spy.math.normalize(self.target - self.position)
        self.right = spy.math.normalize(spy.math.cross(self.fwd, self.up))
        self.up = spy.float3(0, 1, 0)
        # self.up = spy.math.normalize(spy.math.cross(self.right, self.fwd))

        fov = spy.math.radians(self.fov)

        self.image_w = self.fwd * self.focal_distance;
        self.image_u = spy.math.normalize(spy.math.cross(self.image_w, self.up));
        self.image_v = spy.math.normalize(spy.math.cross(self.image_u, self.image_w));
        ulen = self.focal_distance * spy.math.tan(fov * 0.5) * self.aspect_ratio;
        self.image_u *= ulen;
        vlen = self.focal_distance * spy.math.tan(fov * 0.5);
        self.image_v *= vlen;

        self.proj_matrix = spy.math.perspective(fov, self.aspect_ratio, self.near_clip_plane, self.far_clip_plane)
        self.view_matrix = spy.math.matrix_from_look_at(self.position, self.target, self.up, Handedness.right_handed)
        
        self.view_proj_matrix_no_jitter = spy.math.mul(self.proj_matrix, self.view_matrix)
        self.proj_matrix_no_jitter = self.proj_matrix;

        jitter_matrix = spy.math.matrix_from_translation(spy.math.float3(2 * self.jitter.x, 2 * self.jitter.y, 0))
        self.proj_matrix = spy.math.mul(jitter_matrix, self.proj_matrix)
        self.view_proj_matrix = spy.math.mul(self.proj_matrix, self.view_matrix)
        self.inv_view_proj_matrix = spy.math.inverse(self.view_proj_matrix)

        if previous_view_proj_matrix_no_jitter is None:
            self.prev_view_proj_matrix_no_jitter = self.view_proj_matrix_no_jitter
        else:
            self.prev_view_proj_matrix_no_jitter = previous_view_proj_matrix_no_jitter
        if previous_view_proj_matrix is None:
            self.prev_view_proj_matrix = self.view_proj_matrix
        else:
            self.prev_view_proj_matrix = previous_view_proj_matrix
        self._has_prev_matrices = True

    def begin_frame(self, w, h):
        self.width = w
        self.height = h
        self.prev_jitter = spy.float2(self.jitter.x, self.jitter.y)
        jitter_sample = self.sample_pattern.next()
        self.jitter = jitter_sample * spy.float2(1.0 / w, 1.0 / h)
        if self.frame_index == 0:
            self._has_prev_matrices = False
        self.recompute()
        self.frame_index += 1

    def bind(self, cursor: spy.ShaderCursor):
        cursor["view_matrix"] = self.view_matrix
        cursor["proj_matrix"] = self.proj_matrix
        cursor["view_proj_matrix"] = self.view_proj_matrix
        cursor["inv_view_proj_matrix"] = self.inv_view_proj_matrix
        cursor["proj_matrix_no_jitter"] = self.proj_matrix_no_jitter
        cursor["view_proj_matrix_no_jitter"] = self.view_proj_matrix_no_jitter
        cursor["prev_view_proj_matrix"] = self.prev_view_proj_matrix
        cursor["prev_view_proj_matrix_no_jitter"] = self.prev_view_proj_matrix_no_jitter

        cursor["position"] = self.position
        cursor["image_u"] = self.image_u
        cursor["image_v"] = self.image_v
        cursor["image_w"] = self.image_w
        cursor["jitter_x"] = self.jitter.x;
        cursor["jitter_y"] = self.jitter.y;
        cursor["prev_jitter_x"] = self.prev_jitter.x;
        cursor["prev_jitter_y"] = self.prev_jitter.y;
        cursor["near_clip_plane"] = self.near_clip_plane
        cursor["far_clip_plane"] = self.far_clip_plane

class CameraController:
    MOVE_KEYS = {
        spy.KeyCode.a: spy.float3(-1, 0, 0),
        spy.KeyCode.d: spy.float3(1, 0, 0),
        spy.KeyCode.e: spy.float3(0, 1, 0),
        spy.KeyCode.q: spy.float3(0, -1, 0),
        spy.KeyCode.w: spy.float3(0, 0, 1),
        spy.KeyCode.s: spy.float3(0, 0, -1),
    }
    MOVE_SHIFT_FACTOR = 10.0
    MOVE_SPEED_PERCENT_MIN = 20.0
    MOVE_SPEED_PERCENT_MAX = 2000.0
    MOVE_SPEED_PERCENT_DEFAULT = 100.0

    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self.mouse_down = False
        self.mouse_pos = spy.float2()
        self.key_state = {k: False for k in CameraController.MOVE_KEYS.keys()}
        self.shift_down = False

        self.move_delta = spy.float3()
        self.rotate_delta = spy.float2()

        self.move_speed = 1.0
        self.move_speed_percent = CameraController.MOVE_SPEED_PERCENT_DEFAULT
        self.rotate_speed = 0.002

        self.move_test = False

    def set_move_speed_percent(self, value: float) -> None:
        self.move_speed_percent = max(
            CameraController.MOVE_SPEED_PERCENT_MIN,
            min(CameraController.MOVE_SPEED_PERCENT_MAX, float(value)),
        )

    def update(self, dt: float, frame: int):
        changed = False
        position = self.camera.position
        fwd = self.camera.fwd
        up = self.camera.up
        right = self.camera.right

        if frame > 1 and self.move_test:
            if frame % 120 < 60:
                self.move_delta = spy.float3(1.0, 0.0, 0)
            else:
                self.move_delta = spy.float3(-1.0, 0.0, 0)

        # Move
        if spy.math.length(self.move_delta) > 0:
            offset = right * self.move_delta.x
            offset += up * self.move_delta.y
            offset += fwd * self.move_delta.z
            factor = CameraController.MOVE_SHIFT_FACTOR if self.shift_down else 1.0
            speed_scale = self.move_speed_percent / 100.0
            offset *= self.move_speed * speed_scale * factor * dt
            position += offset
            changed = True

        # Rotate
        if spy.math.length(self.rotate_delta) > 0:
            yaw = spy.math.atan2(fwd.z, fwd.x)
            pitch = spy.math.asin(fwd.y)
            yaw += self.rotate_speed * self.rotate_delta.x
            pitch -= self.rotate_speed * self.rotate_delta.y
            fwd = spy.float3(
                spy.math.cos(yaw) * spy.math.cos(pitch),
                spy.math.sin(pitch),
                spy.math.sin(yaw) * spy.math.cos(pitch),
            )
            self.rotate_delta = spy.float2()
            changed = True

        if changed:
            self.camera.position = position
            self.camera.target = position + fwd
            self.camera.up = spy.float3(0, 1, 0)
            self.camera.recompute()

        return changed

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press() or event.is_key_release():
            down = event.is_key_press()
            if event.key in CameraController.MOVE_KEYS:
                self.key_state[event.key] = down
            elif event.key == spy.KeyCode.left_shift:
                self.shift_down = down
        self.move_delta = spy.float3()
        for key, state in self.key_state.items():
            if state:
                self.move_delta += CameraController.MOVE_KEYS[key]

    def on_mouse_event(self, event: spy.MouseEvent):
        self.rotate_delta = spy.float2()
        if event.is_button_down() and event.button == spy.MouseButton.left:
            self.mouse_down = True
        if event.is_button_up() and event.button == spy.MouseButton.left:
            self.mouse_down = False
        if event.is_move():
            mouse_delta = event.pos - self.mouse_pos
            if self.mouse_down:
                self.rotate_delta = mouse_delta
            self.mouse_pos = event.pos
