import slangpy as spy


class MeshColorsRGB9E5Packer:
    """Pack writable float3 Mesh Colors irradiance into one uint per texel."""

    def __init__(self, device: spy.Device):
        self.program = device.load_program(
            "mesh_colors_rgb9e5_pack.slang", ["compute_main"]
        )
        self.pipeline = device.create_compute_pipeline(self.program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        source: spy.Buffer,
        destination: spy.Buffer,
        texel_count: int,
    ) -> None:
        texel_count = max(0, int(texel_count))
        if texel_count == 0:
            return
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_source = source
            cursor.g_destination = destination
            cursor.g_texel_count = texel_count
            pass_encoder.dispatch(thread_count=[texel_count, 1, 1])
