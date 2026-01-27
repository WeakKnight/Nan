import slangpy as spy
from scene import Scene
from typing import Protocol
from render_data import RenderData

class Renderer(Protocol):
    def initialize(self, device: spy.Device, scene: Scene):
        ...

    def render(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
        frame: int,
        device: spy.Device,
        scene: Scene,
        render_data: RenderData,
    ):
        ...

    def setup_ui(self, ui_context: spy.ui.Context, ui_window: spy.ui.Window):
        ...
