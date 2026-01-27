import slangpy as spy
class ToneMapper:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("tone_mapper.slang", ["compute_main"])
        self.kernel = self.device.create_compute_kernel(self.program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,
    ):
        self.kernel.dispatch(
            thread_count=[input.width, input.height, 1],
            vars={
                "g_tone_mapper": {
                    "input": input,
                    "output": output,
                }
            },
            command_encoder=command_encoder,
        )