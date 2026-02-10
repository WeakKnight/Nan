import slangpy as spy
from scene import Scene

class PathTracer:
    def __init__(self, device: spy.Device, scene: Scene):
        super().__init__()
        self.device = device
        self.scene = scene
        self.program = self.device.load_program("path_tracer.slang", ["compute_main"])
        self.pipeline = self.device.create_compute_pipeline(self.program)

    def execute(self, command_encoder: spy.CommandEncoder, output: spy.Texture, frame: int,
                motion: spy.Texture | None = None, viewz: spy.Texture | None = None):
        w = output.width
        h = output.height

        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_frame = frame
            if motion is not None:
                cursor.g_motion = motion
            if viewz is not None:
                cursor.g_viewz = viewz
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[w, h, 1])
