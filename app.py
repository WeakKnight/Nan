from slangpy import Device


from event_dispatcher.sync_dispatcher import SyncEventDispatcher


from slangpy.ui import Context


import slangpy as spy
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from camera import CameraController
from scene import Scene
from scene_node import SceneNode
from renderer import Renderer
from render_data import RenderData
from extensions import ExtensionManager
# install pip package event-dispatching https://pypi.org/project/event-dispatching/
import event_dispatcher 

PROJECT_DIR = Path(__file__).parent
SCENE_CAMERA_CONFIG_FILENAME = "nan_camera.json"


@dataclass
class AppConfig:
    width: int = 1920
    height: int = 1080
    headless: bool = False
    headless_frame_count: int = 64
    headless_output: Path = field(default_factory=lambda: Path("headless_output.png"))
    vsync: bool = False
    srgb_output: bool = True
    camera_move_test: bool = False
    scene_path: str | None = None
    enabled_extensions: tuple[str, ...] = ()
    static_shadow_mode: int | None = None
    static_shadow_resolution: int | tuple[int, int] | None = None
    sst_encoder_backend: str = "auto"
    static_shadow_mask_mode: str = "off"
    static_shadow_mask_threshold: float = 0.02
    static_shadow_mask_bootstrap_passes: int = 2
    startup_profile: bool = False

class App:
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()

        self.config: AppConfig = config or AppConfig()
        self.headless: bool = self.config.headless
        profile_origin = time.perf_counter()
        profile_previous = profile_origin

        def profile_mark(label: str) -> None:
            nonlocal profile_previous
            now = time.perf_counter()
            if self.config.startup_profile:
                print(
                    f"[SurfaceProbeProfile] {label}: "
                    f"{now - profile_previous:.3f}s "
                    f"(app_total={now - profile_origin:.3f}s)",
                    flush=True,
                )
            profile_previous = now

        self.window: None | spy.Window = None
        if not self.headless:
            self.window = spy.Window(
                width=self.config.width,
                height=self.config.height,
                title="Nan",
                resizable=True,
            )
        profile_mark("window_create")
        self.shader_defines: dict[str, str] = {"USE_RAYTRACING_PIPELINE": "0"}
        if self.headless:
            self.shader_defines["HEADLESS_MODE"] = "1"
        else:
            self.shader_defines["HEADLESS_MODE"] = "0"

        self.device: Device = spy.Device(
            # type=spy.DeviceType.cuda,
            enable_debug_layers=False,
            enable_print=True,
            compiler_options={
                "include_paths": [PROJECT_DIR],
                "defines": self.shader_defines,
            },
        )
        self.surface: spy.Surface | None = None
        if not self.headless and self.window is not None:
            self.surface = self.device.create_surface(self.window)
            self.surface.configure(
                width=self.window.width,
                height=self.window.height,
                vsync=self.config.vsync,
            )
        profile_mark("device_and_surface_create")

        self.output_texture: spy.Texture | None = None  # type: ignore (will be set immediately)
        self.render_data: RenderData = RenderData(self.device)
        self._scene_camera_config_path: Path | None = None
        self._scene_camera_status_text: spy.ui.Text | None = None
        self._scene_camera_status: str = "Scene camera: not loaded"

        if self.window is not None:
            self.window.on_keyboard_event = self.on_keyboard_event
            self.window.on_mouse_event = self.on_mouse_event
            self.window.on_resize = self.on_resize

        self.scene_node: SceneNode = self._load_scene(self.config.scene_path)
        profile_mark("scene_asset_import")
        self._apply_scene_camera_config()

        self.event_dispatcher: SyncEventDispatcher = event_dispatcher.SyncEventDispatcher()
        self.scene: Scene = Scene(self.device, self.scene_node, self.event_dispatcher)
        profile_mark("scene_gpu_resources_and_acceleration_structures")
        self.extensions: ExtensionManager = ExtensionManager(self.config.enabled_extensions)
        self.extensions.initialize(self)
        self.scene.extensions = self.extensions
        profile_mark("extensions_initialize")

        self.camera_controller: CameraController = CameraController(self.scene_node.camera)
        self.camera_controller.move_test = self.config.camera_move_test

        self.ui: Context | None = spy.ui.Context(self.device) if not self.headless else None
        self.renderer: Renderer | None = None
        profile_mark("camera_and_ui_initialize")

        self.render_doc_is_available = (
            spy.renderdoc.is_available() and not self.headless
        )
        if self.render_doc_is_available:
            print("RenderDoc Avaliable")
        self.should_capture = False

    def _load_scene(self, scene_path: Optional[str]) -> SceneNode:
        """Load scene from path, choosing loader based on file extension."""
        if scene_path is None:
            # Default scene: Cornell box
            return SceneNode.demo()
        
        path = Path(scene_path)
        self._scene_camera_config_path = path.resolve().parent / SCENE_CAMERA_CONFIG_FILENAME
        if path.suffix.lower() == ".json":
            return SceneNode.load_json(str(path), axis_conversion="z_up_to_y_up")
        else:
            return SceneNode.load_asset(str(path), scale=0.1)

    def _scene_camera_key(self) -> str | None:
        if self.config.scene_path is None:
            return None
        return Path(self.config.scene_path).name

    @staticmethod
    def _float3_to_list(value) -> list[float]:
        return [float(value[0]), float(value[1]), float(value[2])]

    @staticmethod
    def _read_float3(data, key: str) -> spy.float3:
        values = data.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"camera.{key} must be a 3-float array")
        parsed = [float(values[0]), float(values[1]), float(values[2])]
        if not all(math.isfinite(v) for v in parsed):
            raise ValueError(f"camera.{key} contains non-finite values")
        return spy.float3(parsed[0], parsed[1], parsed[2])

    def _camera_config_payload(self) -> dict:
        camera = self.scene_node.camera
        return {
            "version": 1,
            "scene": self._scene_camera_key(),
            "camera": {
                "position": self._float3_to_list(camera.position),
                "target": self._float3_to_list(camera.target),
                "up": self._float3_to_list(camera.up),
                "fov": float(camera.fov),
                "near_clip_plane": float(camera.near_clip_plane),
                "far_clip_plane": float(camera.far_clip_plane),
                "focal_distance": float(camera.focal_distance),
            },
        }

    def _set_scene_camera_status(self, text: str) -> None:
        self._scene_camera_status = text
        if self._scene_camera_status_text is not None:
            self._scene_camera_status_text.text = text

    def _save_scene_camera_config(self) -> None:
        if self._scene_camera_config_path is None:
            self._set_scene_camera_status("Scene camera: no scene path")
            return
        try:
            self._scene_camera_config_path.parent.mkdir(parents=True, exist_ok=True)
            self._scene_camera_config_path.write_text(
                json.dumps(self._camera_config_payload(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self._set_scene_camera_status(f"Scene camera: save failed ({exc})")
            print(f"[App] Failed to save scene camera config: {exc}")
            return
        self._set_scene_camera_status(f"Scene camera: saved {self._scene_camera_config_path.name}")
        print(f"[App] Scene camera saved to {self._scene_camera_config_path}")

    def _apply_scene_camera_config(self) -> bool:
        if self._scene_camera_config_path is None:
            self._set_scene_camera_status("Scene camera: no scene path")
            return False
        if not self._scene_camera_config_path.exists():
            self._set_scene_camera_status(f"Scene camera: no {self._scene_camera_config_path.name}")
            return False

        try:
            data = json.loads(self._scene_camera_config_path.read_text(encoding="utf-8"))
            if data.get("scene") != self._scene_camera_key():
                self._set_scene_camera_status("Scene camera: config belongs to another scene")
                return False
            camera_data = data.get("camera")
            if not isinstance(camera_data, dict):
                raise ValueError("camera block is missing")

            camera = self.scene_node.camera
            camera.position = self._read_float3(camera_data, "position")
            camera.target = self._read_float3(camera_data, "target")
            camera.up = self._read_float3(camera_data, "up")
            camera.fov = float(camera_data.get("fov", camera.fov))
            camera.near_clip_plane = float(camera_data.get("near_clip_plane", camera.near_clip_plane))
            camera.far_clip_plane = float(camera_data.get("far_clip_plane", camera.far_clip_plane))
            camera.focal_distance = float(camera_data.get("focal_distance", camera.focal_distance))
            if not all(
                math.isfinite(v)
                for v in (
                    camera.fov,
                    camera.near_clip_plane,
                    camera.far_clip_plane,
                    camera.focal_distance,
                )
            ):
                raise ValueError("camera scalar values must be finite")
            if camera.fov <= 0.0 or camera.fov >= 179.0:
                raise ValueError("camera.fov must be in (0, 179)")
            if camera.near_clip_plane <= 0.0:
                raise ValueError("camera.near_clip_plane must be positive")
            if camera.far_clip_plane <= camera.near_clip_plane:
                raise ValueError("camera.far_clip_plane must be greater than near_clip_plane")
            if camera.focal_distance <= 0.0:
                raise ValueError("camera.focal_distance must be positive")
            camera.recompute()
        except Exception as exc:
            self._set_scene_camera_status(f"Scene camera: load failed ({exc})")
            print(f"[App] Failed to load scene camera config: {exc}")
            return False

        self._set_scene_camera_status(f"Scene camera: loaded {self._scene_camera_config_path.name}")
        print(f"[App] Scene camera loaded from {self._scene_camera_config_path}")
        return True

    def _reload_scene_camera_config(self) -> None:
        if self._apply_scene_camera_config():
            self.event_dispatcher.dispatch("camera_move")

    def set_renderer(self, renderer: Renderer):
        self.renderer = renderer
        self.renderer.initialize(self.device, self.scene)
        if self.ui is not None:
            ui_window = spy.ui.Window(
                self.ui.screen, "Settings", spy.float2(10, 10), spy.float2(420, 380)
            )

            # def render_doc_capture_btn():
            #     self.should_capture = True
            #     print("Btn Pressed")
            #
            # # if self.render_doc_is_available:
            # spy.ui.Button(ui_window, 'RenderDoc Capture', callback=render_doc_capture_btn)
            spy.ui.Button(ui_window, "Save Scene Camera", callback=self._save_scene_camera_config)
            spy.ui.Button(ui_window, "Reload Scene Camera", callback=self._reload_scene_camera_config)
            self._scene_camera_status_text = spy.ui.Text(ui_window, self._scene_camera_status)

            def on_camera_speed_changed(value: float) -> None:
                self.camera_controller.set_move_speed_percent(value)

            spy.ui.SliderFloat(
                ui_window,
                "Camera Speed %",
                min=CameraController.MOVE_SPEED_PERCENT_MIN,
                max=CameraController.MOVE_SPEED_PERCENT_MAX,
                value=self.camera_controller.move_speed_percent,
                callback=on_camera_speed_changed,
            )
            self.scene.setup_ui(
                self.ui,
                ui_window,
                include_static_shadow_controls=self.extensions.has("static_shadow_sst"),
            )
            self.extensions.setup_ui(self, self.ui, ui_window)
            self.renderer.setup_ui(self.ui, ui_window)

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.headless or self.window is None:
            return
        if event.type == spy.KeyboardEventType.key_press:
            self.event_dispatcher.dispatch('key_press', event.key)
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.f1:
                if self.output_texture:
                    spy.tev.show_async(self.output_texture)
            elif event.key == spy.KeyCode.f2:
                if self.output_texture:
                    bitmap = self.output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")
            elif event.key == spy.KeyCode.f11 and self.render_doc_is_available:
                self.should_capture = True

        if not self.ui.handle_keyboard_event(event):
            self.camera_controller.on_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.headless or self.ui is None:
            return
        if not self.ui.handle_mouse_event(event):
            self.camera_controller.on_mouse_event(event)

    def on_resize(self, width: int, height: int):
        if self.headless or self.surface is None:
            return
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=self.config.vsync)
        else:
            self.surface.unconfigure()

    def main_loop(self):
        self.extensions.before_main_loop(self)
        if self.headless:
            self._headless_loop()
        else:
            self._interactive_loop()
        self.device.wait()

    def _configure_static_shadow_if_requested(self) -> None:
        if self.config.static_shadow_mode is None:
            return

        self.configure_static_shadow_mode(
            self.config.static_shadow_mode,
            self.config.static_shadow_resolution,
        )

    def configure_static_shadow_mode(self, requested_mode: int, resolution: int | tuple[int, int] | None = None) -> None:
        extension = self.extensions.get("static_shadow_sst")
        if extension is None:
            raise RuntimeError(
                "Static shadow mode requires the 'static_shadow_sst' extension. "
                "Enable it with --enable-extension static_shadow_sst."
            )
        extension.configure_static_shadow_mode(requested_mode, resolution)

    def render_headless_to_file(self, output_path: Path, frame_count: int | None = None) -> None:
        if not self.headless:
            raise RuntimeError("render_headless_to_file requires a headless App")

        self.config.headless_output = Path(output_path)
        if frame_count is not None:
            self.config.headless_frame_count = max(1, int(frame_count))
        self.render_data.clear()
        self._reset_headless_temporal_state()
        if self.renderer is not None and hasattr(self.renderer, "reset_accumulator"):
            self.renderer.reset_accumulator = True
        if self.renderer is not None and hasattr(self.renderer, "reset_texture_space"):
            self.renderer.reset_texture_space = True
            if hasattr(self.renderer, "texture_iteration"):
                self.renderer.texture_iteration = 0
        self._headless_loop()

    def _reset_headless_temporal_state(self) -> None:
        self.scene._frame_index = 0
        camera = self.scene.camera
        camera.frame_index = 0
        camera.sample_pattern.current_sample = 0
        camera.jitter = spy.float2(0, 0)
        camera.prev_jitter = spy.float2(0, 0)
        camera._has_prev_matrices = False
        camera.recompute()

    def _ensure_output_texture(self, width: int, height: int) -> None:
        if (
            self.output_texture is None
            or self.output_texture.width != width
            or self.output_texture.height != height
        ):
            self.output_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=width,
                height=height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="output_texture",
            )

    def _interactive_loop(self) -> None:
        if self.window is None or self.surface is None:
            return
        frame = 0
        timer = spy.Timer()
        while not self.window.should_close():
            dt = timer.elapsed_s()
            timer.reset()

            self.window.process_events()

            if self.camera_controller.update(dt, frame):
                self.event_dispatcher.dispatch("camera_move")

            if not self.surface.config:
                continue
            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            self._ensure_output_texture(surface_texture.width, surface_texture.height)

            if self.should_capture:
                spy.renderdoc.start_frame_capture(self.device, self.window)
                print("Start Capture")

            self.scene.camera.begin_frame(
                self.output_texture.width, self.output_texture.height
            )

            command_encoder = self.device.create_command_encoder()
            self.scene.update()
            self.extensions.before_render(
                command_encoder,
                self.output_texture,
                frame,
                self.device,
                self.scene,
                self.render_data,
            )
            if self.renderer is not None:
                self.renderer.render(
                    command_encoder,
                    self.output_texture,
                    frame,
                    self.device,
                    self.scene,
                    self.render_data,
                )
            self.extensions.after_render(
                command_encoder,
                self.output_texture,
                frame,
                self.device,
                self.scene,
                self.render_data,
            )
            command_encoder.blit(surface_texture, self.output_texture)

            if self.ui is not None:
                window_size = (self.window.width, self.window.height)
                self.ui.begin_frame(*window_size)
                self.ui.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

            if self.should_capture:
                spy.renderdoc.end_frame_capture()
                print("End Capture")
                self.should_capture = False

            self.device.flush_print()
            frame += 1

    def _headless_loop(self) -> None:
        frame_count = max(0, self.config.headless_frame_count)
        if frame_count <= 0:
            return

        self._ensure_output_texture(self.config.width, self.config.height)

        timer = spy.Timer()

        for frame in range(frame_count):
            dt = timer.elapsed_s()
            timer.reset()

            if self.camera_controller.update(dt, frame):
                self.event_dispatcher.dispatch("camera_move")

            self.scene.camera.begin_frame(
                self.output_texture.width, self.output_texture.height
            )

            command_encoder = self.device.create_command_encoder()
            self.scene.update()
            self.extensions.before_render(
                command_encoder,
                self.output_texture,
                frame,
                self.device,
                self.scene,
                self.render_data,
            )
            if self.renderer is not None:
                self.renderer.render(
                    command_encoder,
                    self.output_texture,
                    frame,
                    self.device,
                    self.scene,
                    self.render_data,
                )
            self.extensions.after_render(
                command_encoder,
                self.output_texture,
                frame,
                self.device,
                self.scene,
                self.render_data,
            )

            self.device.submit_command_buffer(command_encoder.finish())
            self.device.flush_print()

        self._save_output_texture()

    def _save_output_texture(self) -> None:
        if self.output_texture is None:
            return

        output_path = self.config.headless_output
        if not output_path:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bitmap = self.output_texture.to_bitmap()
        if self.config.srgb_output:
            bitmap = bitmap.convert(
                spy.Bitmap.PixelFormat.rgb,
                spy.Bitmap.ComponentType.uint8,
                srgb_gamma=True,
            )

        future = bitmap.write_async(str(output_path))
        if future is not None:
            wait = getattr(future, "wait", None)
            if callable(wait):
                wait()
            else:
                result = getattr(future, "result", None)
                if callable(result):
                    result()
        print(f"Headless output saved to {output_path}")
