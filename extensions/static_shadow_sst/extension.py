from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import slangpy as spy

from extensions.base import ExtensionContext
from scene import Scene
from .screen_shadow_mask import ScreenShadowMaskPass


@dataclass
class StaticShadowSSTSettings:
    mode: int | None = None
    resolution: int | tuple[int, int] | None = None
    encoder_backend: str = "auto"
    screen_mask_mode: str = "off"
    screen_mask_threshold: float = 0.02
    screen_mask_bootstrap_passes: int = 2


class StaticShadowMaskProvider:
    def __init__(self, device: spy.Device, settings: StaticShadowSSTSettings):
        self.device = device
        self.settings = settings
        self._pass: ScreenShadowMaskPass | None = None

    def create(
        self,
        command_encoder: spy.CommandEncoder,
        scene: Scene,
        render_data: Any,
        width: int,
        height: int,
    ) -> spy.Texture | None:
        if self.settings.screen_mask_mode == "off":
            return None
        if not scene.static_shadow_enabled or scene.static_shadow_mode == Scene.SHADOW_MODE_REALTIME:
            return None
        if self._pass is None:
            self._pass = ScreenShadowMaskPass(self.device)
        return self._pass.execute(
            command_encoder,
            scene,
            render_data,
            width,
            height,
            self.settings.screen_mask_mode,
            self.settings.screen_mask_threshold,
            self.settings.screen_mask_bootstrap_passes,
        )


class StaticShadowSSTExtension:
    name = "static_shadow_sst"

    def __init__(self):
        self.app: Any | None = None
        self.context: ExtensionContext | None = None
        self.settings = StaticShadowSSTSettings()
        self.screen_shadow_mask_provider: StaticShadowMaskProvider | None = None
        self._shadow_mask_mode_combo: spy.ui.ComboBox | None = None

    def initialize(self, context: ExtensionContext) -> None:
        self.context = context
        self.app = context.app
        app = context.app
        self.settings = StaticShadowSSTSettings(
            mode=app.config.static_shadow_mode,
            resolution=app.config.static_shadow_resolution,
            encoder_backend=app.config.sst_encoder_backend,
            screen_mask_mode=app.config.static_shadow_mask_mode,
            screen_mask_threshold=max(0.0, float(app.config.static_shadow_mask_threshold)),
            screen_mask_bootstrap_passes=max(0, min(4, int(app.config.static_shadow_mask_bootstrap_passes))),
        )

        app.scene.set_sst_encoder_backend(self.settings.encoder_backend)
        if self.settings.resolution is not None:
            app.scene.set_static_shadow_resolution(self.settings.resolution)
        self.screen_shadow_mask_provider = StaticShadowMaskProvider(app.device, self.settings)
        context.register_service("screen_shadow_mask_provider", self.screen_shadow_mask_provider)

    def before_main_loop(self, app: Any) -> None:
        if self.settings.mode is None:
            return
        self.configure_static_shadow_mode(self.settings.mode, self.settings.resolution)

    def configure_static_shadow_mode(
        self,
        requested_mode: int,
        resolution: int | tuple[int, int] | None = None,
    ) -> None:
        if self.app is None:
            raise RuntimeError("static_shadow_sst extension is not initialized")

        scene = self.app.scene
        requested_mode = max(
            Scene.SHADOW_MODE_REALTIME,
            min(Scene.SHADOW_MODE_DECOMPRESSED_SST, int(requested_mode)),
        )
        if resolution is not None:
            scene.set_static_shadow_resolution(resolution)
        if requested_mode == Scene.SHADOW_MODE_REALTIME:
            scene.static_shadow_mode = Scene.SHADOW_MODE_REALTIME
            return

        needs_sst = requested_mode in (
            Scene.SHADOW_MODE_SST,
            Scene.SHADOW_MODE_PACKED_SST,
            Scene.SHADOW_MODE_COMPACT_SST,
            Scene.SHADOW_MODE_COMPACT_SST_PCF3,
            Scene.SHADOW_MODE_DECOMPRESSED_SST,
        )
        scene.static_shadow_auto_encode_sst = needs_sst

        print(
            "[StaticShadowSST] Baking static shadow before render: "
            f"mode={requested_mode} size={scene.static_shadow_size[0]}x{scene.static_shadow_size[1]}"
        )
        scene.bake_static_shadow_depth_map()
        if needs_sst and not scene.sst_enabled:
            scene.encode_sparse_shadow_tree()

        if needs_sst and not scene.sst_enabled:
            print("[StaticShadowSST] Requested SST shadow mode but SST encode failed; falling back to depth texture")
            requested_mode = Scene.SHADOW_MODE_DEPTH_TEXTURE

        scene.static_shadow_mode = requested_mode
        scene._sync_shadow_mode_ui()
        scene._update_static_shadow_status()

    def setup_ui(self, app: Any, ui_context: spy.ui.Context, ui_window: spy.ui.Window) -> None:
        def on_shadow_mask_mode_changed(value=None):
            if self._shadow_mask_mode_combo is None:
                return
            index = self._shadow_mask_mode_combo.value if value is None else int(value)
            modes = ScreenShadowMaskPass.MODES
            if 0 <= index < len(modes):
                self.settings.screen_mask_mode = modes[index]
                self._reset_renderer_accumulation(app)

        self._shadow_mask_mode_combo = spy.ui.ComboBox(
            ui_window,
            "Shadow Mask",
            items=["Off", "Full", "Adaptive", "Adaptive Wave"],
            value=self._shadow_mask_mode_index(),
            callback=on_shadow_mask_mode_changed,
        )

        def on_shadow_mask_threshold_changed(value):
            self.settings.screen_mask_threshold = max(0.0, float(value))
            self._reset_renderer_accumulation(app)

        spy.ui.SliderFloat(
            ui_window,
            "Shadow Mask Threshold",
            min=0.0,
            max=0.25,
            value=self.settings.screen_mask_threshold,
            callback=on_shadow_mask_threshold_changed,
        )

    def _shadow_mask_mode_index(self) -> int:
        try:
            return ScreenShadowMaskPass.MODES.index(self.settings.screen_mask_mode)
        except ValueError:
            self.settings.screen_mask_mode = "off"
            return 0

    @staticmethod
    def _reset_renderer_accumulation(app: Any) -> None:
        if app.renderer is not None and hasattr(app.renderer, "reset_accumulator"):
            app.renderer.reset_accumulator = True
