from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import slangpy as spy

from render_data import RenderData
from surface_probe_resources import SURFACE_PROBE_RADIAL_MOMENT_SIZE


SURFACE_PROBE_SAMPLE_COUNT_SIZE = 4
SURFACE_PROBE_SELF_HIT_SIZE = 4


class SurfaceProbeFieldSemantic(str, Enum):
    DIFFUSE_IRRADIANCE_RGB = "diffuse_irradiance_rgb"
    DIFFUSE_PRT_L2_SCALAR = "diffuse_prt_l2_scalar"


class SurfaceProbeFieldStorage(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"

    @property
    def scalar_size(self) -> int:
        return 4 if self is SurfaceProbeFieldStorage.FLOAT32 else 2


class SurfaceProbeAttachmentSemantic(str, Enum):
    SELF_HIT_COUNTS = "self_hit_counts"
    RADIAL_MOMENTS_4X4 = "radial_moments_4x4"


@dataclass(frozen=True)
class SurfaceProbeAttachmentDesc:
    semantic: SurfaceProbeAttachmentSemantic
    stride: int

    def __post_init__(self) -> None:
        if self.stride <= 0 or self.stride % 4 != 0:
            raise ValueError(
                "Probe attachment stride must be positive and 4-byte aligned"
            )


@dataclass(frozen=True)
class SurfaceProbeFieldDesc:
    semantic: SurfaceProbeFieldSemantic
    coefficient_count: int
    channel_count: int
    working_storage: SurfaceProbeFieldStorage
    version: int = 1

    def __post_init__(self) -> None:
        if self.coefficient_count <= 0:
            raise ValueError("Probe field coefficient count must be positive")
        if self.channel_count <= 0:
            raise ValueError("Probe field channel count must be positive")
        if self.version <= 0:
            raise ValueError("Probe field version must be positive")

    @property
    def value_stride(self) -> int:
        return (
            self.coefficient_count
            * self.channel_count
            * self.working_storage.scalar_size
        )

    @property
    def bytes_per_probe(self) -> int:
        # Progressive validity/convergence is common state, not part of the
        # field encoding itself.
        return self.value_stride + SURFACE_PROBE_SAMPLE_COUNT_SIZE


DIFFUSE_IRRADIANCE_RGB_FIELD = SurfaceProbeFieldDesc(
    semantic=SurfaceProbeFieldSemantic.DIFFUSE_IRRADIANCE_RGB,
    coefficient_count=1,
    channel_count=3,
    working_storage=SurfaceProbeFieldStorage.FLOAT32,
)

SELF_HIT_COUNTS_ATTACHMENT = SurfaceProbeAttachmentDesc(
    SurfaceProbeAttachmentSemantic.SELF_HIT_COUNTS,
    SURFACE_PROBE_SELF_HIT_SIZE,
)
RADIAL_MOMENTS_4X4_ATTACHMENT = SurfaceProbeAttachmentDesc(
    SurfaceProbeAttachmentSemantic.RADIAL_MOMENTS_4X4,
    SURFACE_PROBE_RADIAL_MOMENT_SIZE,
)
DIFFUSE_IRRADIANCE_ATTACHMENTS = (
    SELF_HIT_COUNTS_ATTACHMENT,
    RADIAL_MOMENTS_4X4_ATTACHMENT,
)


@dataclass(frozen=True)
class SurfaceProbeFieldBuffers:
    desc: SurfaceProbeFieldDesc
    values: spy.Buffer
    sample_counts: spy.Buffer


@dataclass(frozen=True)
class SurfaceProbeAttachments:
    buffers: dict[SurfaceProbeAttachmentSemantic, spy.Buffer]

    def require(self, semantic: SurfaceProbeAttachmentSemantic) -> spy.Buffer:
        try:
            return self.buffers[semantic]
        except KeyError as error:
            raise ValueError(
                f"Surface Probe attachment is not available: {semantic.value}"
            ) from error

    @property
    def self_hit_counts(self) -> spy.Buffer:
        return self.require(SurfaceProbeAttachmentSemantic.SELF_HIT_COUNTS)

    @property
    def radial_moments(self) -> spy.Buffer:
        return self.require(
            SurfaceProbeAttachmentSemantic.RADIAL_MOMENTS_4X4
        )


class SurfaceProbeFieldBaker(Protocol):
    """Host contract implemented by a field-specific GPU baker."""

    field_desc: SurfaceProbeFieldDesc

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        field: SurfaceProbeFieldBuffers,
        attachments: SurfaceProbeAttachments,
        iteration: int,
        *,
        reset: bool = False,
    ) -> None: ...


@dataclass(frozen=True)
class SurfaceProbeRuntimeBuffers:
    field: SurfaceProbeFieldBuffers
    attachments: SurfaceProbeAttachments

    @classmethod
    def acquire(
        cls,
        render_data: RenderData,
        probe_count: int,
        *,
        field_desc: SurfaceProbeFieldDesc = DIFFUSE_IRRADIANCE_RGB_FIELD,
        attachment_descs: tuple[
            SurfaceProbeAttachmentDesc, ...
        ] = DIFFUSE_IRRADIANCE_ATTACHMENTS,
        resource_prefix: str = "surface_probe_renderer",
    ) -> "SurfaceProbeRuntimeBuffers":
        if field_desc.working_storage is not SurfaceProbeFieldStorage.FLOAT32:
            raise ValueError(
                "Progressive Surface Probe fields currently require FP32"
            )
        probe_count = max(1, int(probe_count))
        usage = (
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource
        )
        semantic = field_desc.semantic.value
        field_key = (
            f"{resource_prefix}.field.{semantic}.v{field_desc.version}"
        )
        values = render_data.get_buffer(
            f"{field_key}.values",
            usage=usage,
            size=field_desc.value_stride * probe_count,
            label=f"surface_probe_{semantic}_values",
        )
        sample_counts = render_data.get_buffer(
            f"{field_key}.sample_counts",
            usage=usage,
            struct_size=SURFACE_PROBE_SAMPLE_COUNT_SIZE,
            element_count=probe_count,
            label="surface_probe_sample_counts",
        )
        attachment_buffers: dict[
            SurfaceProbeAttachmentSemantic, spy.Buffer
        ] = {}
        for attachment_desc in attachment_descs:
            semantic_key = attachment_desc.semantic
            if semantic_key in attachment_buffers:
                raise ValueError(
                    f"Duplicate Surface Probe attachment: {semantic_key.value}"
                )
            attachment_buffers[semantic_key] = render_data.get_buffer(
                f"{field_key}.attachment.{semantic_key.value}",
                usage=usage,
                struct_size=attachment_desc.stride,
                element_count=probe_count,
                label=f"surface_probe_{semantic_key.value}",
            )
        return cls(
            field=SurfaceProbeFieldBuffers(
                desc=field_desc,
                values=values,
                sample_counts=sample_counts,
            ),
            attachments=SurfaceProbeAttachments(
                buffers=attachment_buffers,
            ),
        )
