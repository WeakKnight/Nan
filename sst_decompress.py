import numpy as np
import slangpy as spy


class SSTDecompressor:
    def __init__(self, device: spy.Device):
        self.device = device
        self.program = self.device.load_program("sst_decompress.slang", ["compute_main"])
        self.pipeline = self.device.create_compute_pipeline(self.program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
        compact_words: spy.Buffer,
        compact_roots: spy.Buffer,
        resolution: int | tuple[int, int],
        tile_grid: tuple[int, int],
        tile_size: int,
        max_traversal_steps: int,
        compact_word_count: int,
        branch_10bit_start_level: int,
    ) -> None:
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_output = output
            cursor.g_sst_compact_words = compact_words
            cursor.g_sst_compact_roots = compact_roots
            if isinstance(resolution, (tuple, list)):
                width, height = int(resolution[0]), int(resolution[1])
            else:
                width = height = int(resolution)
            cursor.g_resolution = spy.uint2(max(1, width), max(1, height))
            cursor.g_tile_grid = spy.uint2(int(tile_grid[0]), int(tile_grid[1]))
            cursor.g_tile_size = int(tile_size)
            cursor.g_max_traversal_steps = int(max_traversal_steps)
            cursor.g_compact_word_count = int(compact_word_count)
            cursor.g_branch_10bit_start_level = int(branch_10bit_start_level)
            pass_encoder.dispatch(thread_count=[output.width, output.height, 1])


def make_sst_buffer(device: spy.Device, data: np.ndarray, label: str) -> spy.Buffer:
    return device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label=label,
        data=np.asarray(data, dtype=np.uint32),
    )
