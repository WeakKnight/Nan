import slangpy as spy
import numpy as np
from scene import Scene

class PathTracer:
    def __init__(self, device: spy.Device, scene: Scene):
        super().__init__()
        self.device = device
        self.scene = scene
        self.program = self.device.load_program("path_tracer.slang", ["compute_main"])
        self.pipeline = self.device.create_compute_pipeline(self.program)
        self.default_static_shadow_mask = self.device.create_texture(
            format=spy.Format.r32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            label="default_static_shadow_mask",
            data=np.ones((1, 1), dtype=np.float32),
        )

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
        frame: int,
        static_shadow_mask: spy.Texture | None = None,
        use_static_shadow_mask: bool = False,
    ):
        w = output.width
        h = output.height

        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_frame = frame
            cursor.g_static_shadow_mask = static_shadow_mask or self.default_static_shadow_mask
            cursor.g_use_static_shadow_mask = 1 if use_static_shadow_mask and static_shadow_mask is not None else 0
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[w, h, 1])
