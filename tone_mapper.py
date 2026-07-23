import time

import slangpy as spy
class ToneMapper:
    def __init__(
        self,
        device: spy.Device,
        *,
        profile_sink: list[tuple[str, float]] | None = None,
    ):
        super().__init__()
        self.device = device
        stage_start = time.perf_counter() if profile_sink is not None else 0.0
        self.program = self.device.load_program("tone_mapper.slang", ["compute_main"])
        if profile_sink is not None:
            now = time.perf_counter()
            profile_sink.append(("tonemap_program_load", now - stage_start))
            stage_start = now
        self.kernel = self.device.create_compute_kernel(self.program)
        if profile_sink is not None:
            profile_sink.append(
                ("tonemap_kernel_create", time.perf_counter() - stage_start)
            )
        self.exposure = 1.0

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
                    "exposure": self.exposure,
                }
            },
            command_encoder=command_encoder,
        )
