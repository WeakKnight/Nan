import slangpy as spy

from mesh_colors import MeshColorsLayout
from scene import Scene
from texture_space_path_tracer import TextureSpacePathTracer


class MeshColorsResolve:
    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        layout: MeshColorsLayout,
        texture_space_path_tracer: TextureSpacePathTracer,
    ):
        self.device = device
        self.scene = scene
        self.layout = layout
        self.texture_space_path_tracer = texture_space_path_tracer
        self.program = device.load_program(
            "mesh_colors_resolve.slang", ["compute_main"]
        )
        self.pipeline = device.create_compute_pipeline(self.program)
        self.frozen_program = device.load_program(
            "mesh_colors_frozen_resolve.slang", ["compute_main"]
        )
        self.frozen_pipeline = device.create_compute_pipeline(self.frozen_program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        mesh_colors: spy.Buffer,
        output: spy.Texture,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_mesh_colors = mesh_colors
            cursor.g_mesh_colors_face_infos = (
                self.texture_space_path_tracer.face_infos_buffer
            )
            cursor.g_mesh_colors_instance_infos = (
                self.texture_space_path_tracer.instance_infos_buffer
            )
            cursor.g_mesh_colors_side_infos = (
                self.texture_space_path_tracer.side_infos_buffer
            )
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[output.width, output.height, 1])

    def execute_frozen(
        self,
        command_encoder: spy.CommandEncoder,
        mesh_colors: spy.Buffer,
        output: spy.Texture,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.frozen_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_mesh_colors = mesh_colors
            cursor.g_mesh_colors_face_infos = (
                self.texture_space_path_tracer.face_infos_buffer
            )
            cursor.g_mesh_colors_instance_infos = (
                self.texture_space_path_tracer.instance_infos_buffer
            )
            cursor.g_mesh_colors_side_infos = (
                self.texture_space_path_tracer.side_infos_buffer
            )
            self.scene.bind(cursor.g_scene)
            pass_encoder.dispatch(thread_count=[output.width, output.height, 1])
