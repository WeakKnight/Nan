import slangpy as spy


class StaticShadowDepthMap:
    def __init__(self, device: spy.Device):
        self.device = device
        self.program = self.device.load_program("static_shadow_depth_map.slang", ["compute_main"])
        self.pipeline = self.device.create_compute_pipeline(self.program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        scene,
        output: spy.Texture,
        output_second: spy.Texture,
        shadow_to_world: spy.float4x4,
    ):
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_output_second = output_second
            cursor.g_shadow_to_world = shadow_to_world
            scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[output.width, output.height, 1])
