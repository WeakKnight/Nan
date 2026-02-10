import slangpy as spy
from typing import Optional
from render_data import RenderData


class TemporalAccumulator:
    """TAA-based temporal accumulator with neighborhood clamping and bicubic history sampling.

    Manages a ping-pong history texture pair. Each frame the previous output
    becomes the history input for the next frame.
    """

    def __init__(self, device: spy.Device, resource_key: Optional[str] = None):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("temporal_accumulator.slang", ["compute_main"])
        self.pipeline = self.device.create_compute_pipeline(self.program)
        self.resource_key = resource_key or f"temporal_accumulator.{id(self)}"

        # Linear clamp sampler for bicubic history fetch
        self.linear_sampler: spy.Sampler = device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        )

        # Ping-pong state
        self._history_key_a = f"{self.resource_key}.history_a"
        self._history_key_b = f"{self.resource_key}.history_b"
        self._use_a_as_history = True  # A = history (read), B = output (write)

        self.min_alpha: float = 0.1
        self.enable_clamp: bool = True
        self.sigma_scale: float = 2.0
        self.history_sharpness: float = 0.66
        self.max_convergence: float = 10.0

    def _get_history_and_output_keys(self):
        if self._use_a_as_history:
            return self._history_key_a, self._history_key_b
        else:
            return self._history_key_b, self._history_key_a

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        render_data: RenderData,
        input_color: spy.Texture,
        input_motion: spy.Texture,
        reset: bool = False,
    ) -> spy.Texture:
        """Run TAA and return the output texture (ping-pong managed internally).

        The returned texture is valid until the next call to execute().
        Pass it directly to downstream passes (e.g. tone mapper).
        """
        w = input_color.width
        h = input_color.height

        history_key, output_key = self._get_history_and_output_keys()

        # Ensure both ping-pong textures exist
        history_tex = render_data.get_texture(
            history_key,
            width=w,
            height=h,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"{self.resource_key}_history",
        )
        taa_output_tex = render_data.get_texture(
            output_key,
            width=w,
            height=h,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"{self.resource_key}_output",
        )

        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)

            cursor.g_input_color = input_color
            cursor.g_input_motion = input_motion
            cursor.g_history = history_tex
            cursor.g_output = taa_output_tex
            cursor.g_linear_sampler = self.linear_sampler

            cursor.g_rect_size = spy.float2(float(w), float(h))
            cursor.g_inv_rect_size = spy.float2(1.0 / w, 1.0 / h)
            cursor.g_rect_size_prev = spy.float2(float(w), float(h))
            cursor.g_inv_render_size = spy.float2(1.0 / w, 1.0 / h)
            cursor.g_min_alpha = self.min_alpha
            cursor.g_reset = reset
            cursor.g_enable_clamp = self.enable_clamp
            cursor.g_sigma_scale = self.sigma_scale
            cursor.g_history_sharpness = self.history_sharpness
            cursor.g_max_convergence = self.max_convergence

            pass_encoder.dispatch(thread_count=[w, h, 1])

        # Swap ping-pong: next frame, current output becomes history
        self._use_a_as_history = not self._use_a_as_history

        return taa_output_tex
