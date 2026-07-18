import math
from pathlib import Path
import unittest

import numpy as np
import slangpy as spy

from mesh_colors_rgb9e5 import MeshColorsRGB9E5Packer


PROJECT_DIR = Path(__file__).parent
RGB9E5_MAX = 65_408.0


def _pack_rgb9e5_reference(value) -> int:
    color = np.asarray(value, dtype=np.float64).copy()
    color = np.nan_to_num(
        color, nan=0.0, posinf=RGB9E5_MAX, neginf=0.0
    )
    color = np.clip(color, 0.0, RGB9E5_MAX)
    maximum = float(np.max(color))
    if maximum < 2.0**-25:
        return 0
    exponent = max(-15, math.floor(math.log2(maximum)) + 1)
    shared_exponent = exponent + 15
    scale = 2.0 ** (shared_exponent - 24)
    mantissa = np.floor(color / scale + 0.5).astype(np.uint32)
    if int(np.max(mantissa)) == 512 and shared_exponent < 31:
        shared_exponent += 1
        scale *= 2.0
        mantissa = np.floor(color / scale + 0.5).astype(np.uint32)
    mantissa = np.minimum(mantissa, 511)
    return int(
        mantissa[0]
        | (mantissa[1] << np.uint32(9))
        | (mantissa[2] << np.uint32(18))
        | (np.uint32(shared_exponent) << np.uint32(27))
    )


def _unpack_rgb9e5_reference(packed: int) -> np.ndarray:
    scale = 2.0 ** ((int(packed) >> 27) - 24)
    return np.asarray(
        [
            int(packed) & 0x1FF,
            (int(packed) >> 9) & 0x1FF,
            (int(packed) >> 18) & 0x1FF,
        ],
        dtype=np.float32,
    ) * scale


class MeshColorsRGB9E5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [PROJECT_DIR]},
        )

    def test_gpu_pack_matches_reference_and_shader_decode(self):
        values = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.25, 0.5, 1.0],
                [100.0, 10.0, 1.0],
                [RGB9E5_MAX, RGB9E5_MAX, RGB9E5_MAX],
                [-1.0, np.nan, np.inf],
                [2.0**-24, 2.0**-20, 2.0**-16],
            ],
            dtype=np.float32,
        )
        payload = np.zeros((len(values), 4), dtype=np.float32)
        payload[:, :3] = values
        source = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
            data=payload,
        )
        packed = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
            struct_size=4,
            element_count=len(values),
        )
        packer = MeshColorsRGB9E5Packer(self.device)
        command_encoder = self.device.create_command_encoder()
        packer.execute(command_encoder, source, packed, len(values))
        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait_for_idle()

        actual_packed = (
            np.asarray(packed.to_numpy()).view(np.uint32).reshape(-1)
        )
        expected_packed = np.asarray(
            [_pack_rgb9e5_reference(value) for value in values],
            dtype=np.uint32,
        )
        np.testing.assert_array_equal(actual_packed, expected_packed)

        program = self.device.load_program(
            "test_mesh_colors_rgb9e5.slang", ["compute_main"]
        )
        pipeline = self.device.create_compute_pipeline(program)
        decoded = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
            struct_size=16,
            element_count=len(values),
        )
        command_encoder = self.device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_input = packed
            cursor.g_output = decoded
            pass_encoder.dispatch(thread_count=[len(values), 1, 1])
        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait_for_idle()

        actual_decoded = (
            np.asarray(decoded.to_numpy()).view(np.float32).reshape(-1, 4)
        )[:, :3]
        expected_decoded = np.stack(
            [_unpack_rgb9e5_reference(value) for value in expected_packed]
        )
        np.testing.assert_array_equal(actual_decoded, expected_decoded)

    def test_quantization_error_is_bounded_by_shared_exponent_step(self):
        values = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [1.0, 2.0, 3.0],
                [10.0, 100.0, 1000.0],
                [0.001, 0.5, 12.0],
            ],
            dtype=np.float64,
        )
        for value in values:
            packed = _pack_rgb9e5_reference(value)
            decoded = _unpack_rgb9e5_reference(packed)
            step = 2.0 ** ((packed >> 27) - 24)
            np.testing.assert_array_less(
                np.abs(decoded - value), step * 0.501
            )


if __name__ == "__main__":
    unittest.main()
