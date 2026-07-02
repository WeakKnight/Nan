from slangpy import Device


from event_dispatcher.sync_dispatcher import SyncEventDispatcher


from slangpy.ui import Context


import slangpy as spy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from camera import CameraController
from scene import Scene
from scene_node import SceneNode
from renderer import Renderer
from render_data import RenderData
# install pip package event-dispatching https://pypi.org/project/event-dispatching/
import event_dispatcher 

PROJECT_DIR = Path(__file__).parent


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

class App:
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()

        self.config: AppConfig = config or AppConfig()
        self.headless: bool = self.config.headless

        self.window: None | spy.Window = None
        if not self.headless:
            self.window = spy.Window(
                width=self.config.width,
                height=self.config.height,
                title="Nan",
                resizable=True,
            )
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

        self.output_texture: spy.Texture | None = None  # type: ignore (will be set immediately)
        self.render_data: RenderData = RenderData(self.device)

        if self.window is not None:
            self.window.on_keyboard_event = self.on_keyboard_event
            self.window.on_mouse_event = self.on_mouse_event
            self.window.on_resize = self.on_resize

        self.scene_node: SceneNode = self._load_scene(self.config.scene_path)

        self.event_dispatcher: SyncEventDispatcher = event_dispatcher.SyncEventDispatcher()
        self.scene: Scene = Scene(self.device, self.scene_node, self.event_dispatcher)

        self.camera_controller: CameraController = CameraController(self.scene_node.camera)
        self.camera_controller.move_test = self.config.camera_move_test

        self.ui: Context | None = spy.ui.Context(self.device) if not self.headless else None
        self.renderer: Renderer | None = None

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
        if path.suffix.lower() == ".json":
            return SceneNode.load_json(str(path), axis_conversion="z_up_to_y_up")
        else:
            return SceneNode.load_asset(str(path), scale=0.1)

    def set_renderer(self, renderer: Renderer):
        self.renderer = renderer
        self.renderer.initialize(self.device, self.scene)
        if self.ui is not None:
            ui_window = spy.ui.Window(
                self.ui.screen, "Settings", spy.float2(10, 10), spy.float2(360, 180)
            )

            # def render_doc_capture_btn():
            #     self.should_capture = True
            #     print("Btn Pressed")
            #
            # # if self.render_doc_is_available:
            # spy.ui.Button(ui_window, 'RenderDoc Capture', callback=render_doc_capture_btn)
            self.scene.setup_ui(self.ui, ui_window)
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
        if self.headless:
            self._headless_loop()
        else:
            self._interactive_loop()
        self.device.wait()

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
            if self.renderer is not None:
                self.renderer.render(
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
            if self.renderer is not None:
                self.renderer.render(
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
