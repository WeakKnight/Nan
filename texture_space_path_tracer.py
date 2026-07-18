import slangpy as spy

from mesh_colors import MeshColorsLayout
from scene import Scene


class TextureSpacePathTracer:
    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        layout: MeshColorsLayout,
        *,
        samples_per_texel: int = 1,
        max_bounces: int = 3,
    ):
        self.device = device
        self.scene = scene
        self.layout = layout
        self.samples_per_texel = max(1, int(samples_per_texel))
        self.max_bounces = max(1, int(max_bounces))
        self.program = device.load_program(
            "texture_space_path_tracer.slang", ["compute_main"]
        )
        self.pipeline = device.create_compute_pipeline(self.program)
        (
            self.face_infos_buffer,
            self.instance_infos_buffer,
            self.side_infos_buffer,
        ) = layout.create_gpu_buffers(device)
        self.adjacency_infos_buffer = layout.create_adjacency_gpu_buffer(device)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Buffer,
        iteration: int,
        *,
        reset: bool = False,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            for instance_index, instance_info in enumerate(self.layout.instance_infos):
                if instance_info.texel_count <= 0:
                    continue
                shader_object = pass_encoder.bind_pipeline(self.pipeline)
                cursor = spy.ShaderCursor(shader_object)
                cursor.g_mesh_colors = output
                cursor.g_mesh_colors_face_infos = self.face_infos_buffer
                cursor.g_mesh_colors_instance_infos = self.instance_infos_buffer
                cursor.g_mesh_colors_side_infos = self.side_infos_buffer
                cursor.g_instance_index = instance_index
                cursor.g_iteration = max(0, int(iteration))
                cursor.g_samples_per_texel = self.samples_per_texel
                cursor.g_max_bounces = self.max_bounces
                cursor.g_reset = 1 if reset else 0
                self.scene.bind(cursor.g_scene)
                pass_encoder.dispatch(
                    thread_count=[instance_info.texel_count, 1, 1]
                )
