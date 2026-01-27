import slangpy as spy
from typing import Optional
from render_data import RenderData


class Accumulator:
    def __init__(self, device: spy.Device, resource_key: Optional[str] = None):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("accumulator.slang", ["compute_main"])
        self.kernel = self.device.create_compute_kernel(self.program)
        self.resource_key = resource_key or f"accumulator.history.{id(self)}"
        self.label = "accumulator"

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        render_data: RenderData,
        input: spy.Texture,
        output: spy.Texture,
        reset: bool = False,
    ):
        accumulator_texture = render_data.get_texture(
            self.resource_key,
            width=input.width,
            height=input.height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=self.label,
        )
        self.kernel.dispatch(
            thread_count=[input.width, input.height, 1],
            vars={
                "g_accumulator": {
                    "input": input,
                    "output": output,
                    "accumulator": accumulator_texture,
                    "reset": reset,
                }
            },
            command_encoder=command_encoder,
        )
