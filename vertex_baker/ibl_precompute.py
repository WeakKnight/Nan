from __future__ import annotations

import math
import time
from pathlib import Path

import slangpy as spy


SHADER_PATH = Path(__file__).resolve().with_name("ibl_precompute.slang")


class EnvironmentIBL:
    SKY_SIZE = 512
    SPECULAR_SIZE = 256
    SPECULAR_MIPS = 5
    DFG_SIZE = 128
    SPECULAR_OCCLUSION_SIZE = 16

    def __init__(
        self,
        device: spy.Device,
        path: str | Path,
        shader_session: spy.SlangSession | None = None,
    ) -> None:
        self.device = device
        self.shader_session = shader_session or device.slang_session
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Environment map not found: {self.path}")

        started = time.perf_counter()
        loader = spy.TextureLoader(device)
        self.equirect = loader.load_texture(
            str(self.path),
            options={"load_as_srgb": False, "generate_mips": False},
        )
        self.equirect_sampler = device.create_sampler(
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        self.cube_sampler = device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        cube_usage = (
            spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target
        )
        self.sky = device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.rgba16_float,
            width=self.SKY_SIZE,
            height=self.SKY_SIZE,
            array_length=6,
            mip_count=spy.ALL_MIPS,
            usage=cube_usage,
            label="baker_ibl_sky",
        )
        self.specular = device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.rgba16_float,
            width=self.SPECULAR_SIZE,
            height=self.SPECULAR_SIZE,
            array_length=6,
            mip_count=self.SPECULAR_MIPS,
            usage=cube_usage,
            label="baker_ibl_specular",
        )
        self.dfg = device.create_texture(
            format=spy.Format.rg16_float,
            width=self.DFG_SIZE,
            height=self.DFG_SIZE,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="baker_ibl_dfg",
        )
        self.specular_occlusion = device.create_texture(
            format=spy.Format.r32_float,
            width=self.SPECULAR_OCCLUSION_SIZE * self.SPECULAR_OCCLUSION_SIZE,
            height=self.SPECULAR_OCCLUSION_SIZE,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="baker_pmr_specular_occlusion",
        )
        self.sh = device.create_buffer(
            size=9 * 16,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="baker_ibl_sh9",
        )

        self._module = self.shader_session.load_module(str(SHADER_PATH))
        self._pipelines = {
            name: device.create_compute_pipeline(self.shader_session.load_program(str(SHADER_PATH), [name]))
            for name in (
                "equirect_to_cube_main",
                "downsample_cube_main",
                "prefilter_specular_main",
                "integrate_dfg_main",
                "integrate_specular_occlusion_main",
                "project_sh_main",
            )
        }
        self._generate()
        self.generation_milliseconds = (time.perf_counter() - started) * 1000.0
        print(
            f"Generated IBL from {self.path.name}: "
            f"{self.SKY_SIZE}px sky, {self.SPECULAR_SIZE}px/{self.SPECULAR_MIPS} mip specular, "
            f"SH9 + {self.DFG_SIZE}px DFG + PMR SO{self.SPECULAR_OCCLUSION_SIZE} in "
            f"{self.generation_milliseconds:.1f} ms"
        )

    def _dispatch_face(
        self,
        encoder: spy.CommandEncoder,
        pipeline: spy.ComputePipeline,
        output: spy.TextureView,
        size: int,
        face: int,
        **bindings,
    ) -> None:
        with encoder.begin_compute_pass() as pass_encoder:
            cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(pipeline))
            cursor.g_env_output = output
            cursor.g_env_face = face
            for name, value in bindings.items():
                cursor[name] = value
            pass_encoder.dispatch(thread_count=[size, size, 1])

    def _generate(self) -> None:
        encoder = self.device.create_command_encoder()
        for face in range(6):
            self._dispatch_face(
                encoder,
                self._pipelines["equirect_to_cube_main"],
                self.sky.create_view(layer=0, layer_count=6, mip=0, mip_count=1),
                self.SKY_SIZE,
                face,
                g_env_equirect=self.equirect,
                g_env_sampler=self.equirect_sampler,
            )
        for mip in range(1, self.sky.mip_count):
            size = max(1, self.SKY_SIZE >> mip)
            for face in range(6):
                self._dispatch_face(
                    encoder,
                    self._pipelines["downsample_cube_main"],
                    self.sky.create_view(layer=0, layer_count=6, mip=mip, mip_count=1),
                    size,
                    face,
                    g_env_sky=self.sky,
                    g_env_sampler=self.cube_sampler,
                    g_env_source_lod=float(mip - 1),
                )

        for mip in range(self.SPECULAR_MIPS):
            size = max(1, self.SPECULAR_SIZE >> mip)
            normalized_lod = float(mip) / float(self.SPECULAR_MIPS - 1)
            roughness = 1.0 - math.sqrt(max(1.0 - normalized_lod, 0.0))
            for face in range(6):
                self._dispatch_face(
                    encoder,
                    self._pipelines["prefilter_specular_main"],
                    self.specular.create_view(layer=0, layer_count=6, mip=mip, mip_count=1),
                    size,
                    face,
                    g_env_sky=self.sky,
                    g_env_sampler=self.cube_sampler,
                    g_env_roughness=roughness,
                    g_env_source_size=float(self.SKY_SIZE),
                )

        with encoder.begin_compute_pass() as pass_encoder:
            cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(self._pipelines["integrate_dfg_main"]))
            cursor.g_dfg_output = self.dfg
            pass_encoder.dispatch(thread_count=[self.DFG_SIZE, self.DFG_SIZE, 1])
        with encoder.begin_compute_pass() as pass_encoder:
            cursor = spy.ShaderCursor(
                pass_encoder.bind_pipeline(self._pipelines["integrate_specular_occlusion_main"])
            )
            cursor.g_specular_occlusion_output = self.specular_occlusion
            pass_encoder.dispatch(
                thread_count=[
                    self.SPECULAR_OCCLUSION_SIZE * self.SPECULAR_OCCLUSION_SIZE,
                    self.SPECULAR_OCCLUSION_SIZE,
                    1,
                ]
            )
        with encoder.begin_compute_pass() as pass_encoder:
            cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(self._pipelines["project_sh_main"]))
            cursor.g_env_equirect = self.equirect
            cursor.g_sh_output = self.sh
            pass_encoder.dispatch(thread_count=[9 * 64, 1, 1])
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
