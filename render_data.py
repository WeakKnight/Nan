import slangpy as spy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _freeze_kwargs(kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    if not kwargs:
        return ()
    return tuple(sorted(kwargs.items()))


@dataclass
class _TextureRecord:
    texture: spy.Texture
    width: int
    height: int
    format: spy.Format
    usage: spy.TextureUsage
    extra_params: Tuple[Tuple[str, Any], ...]


@dataclass
class _BufferRecord:
    buffer: spy.Buffer
    usage: spy.BufferUsage
    size: Optional[int]
    struct_size: Optional[int]
    element_count: Optional[int]
    extra_params: Tuple[Tuple[str, Any], ...]


class RenderData:
    """Frame-scoped cache for render pass resources.

    Render passes can request textures or buffers by name. RenderData will
    lazily create the resource if it does not exist yet or if the requested
    descriptor does not match the cached instance. Otherwise the cached
    resource is returned.
    """

    def __init__(self, device: spy.Device):
        self.device = device
        self._textures: Dict[str, _TextureRecord] = {}
        self._buffers: Dict[str, _BufferRecord] = {}

    def get_texture(
        self,
        name: str,
        *,
        width: int,
        height: int,
        format: spy.Format,
        usage: spy.TextureUsage,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> spy.Texture:
        """Return a texture matching the requested descriptor.

        If a cached texture exists with the same descriptor it is returned,
        otherwise a new texture is created and cached.
        """
        frozen = _freeze_kwargs(kwargs)
        record = self._textures.get(name)
        if (
            record is not None
            and record.width == width
            and record.height == height
            and record.format == format
            and record.usage == usage
            and record.extra_params == frozen
        ):
            return record.texture

        # print(width)
        # print(height)
        texture = self.device.create_texture(
            format=format,
            width=width,
            height=height,
            usage=usage,
            label=label or name,
            **kwargs,
        )
        self._textures[name] = _TextureRecord(
            texture=texture,
            width=width,
            height=height,
            format=format,
            usage=usage,
            extra_params=frozen,
        )
        return texture

    def get_buffer(
        self,
        name: str,
        *,
        usage: spy.BufferUsage,
        size: Optional[int] = None,
        struct_size: Optional[int] = None,
        element_count: Optional[int] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> spy.Buffer:
        """Return a buffer matching the requested descriptor."""
        if size is None and (struct_size is None or element_count is None):
            raise ValueError(
                "get_buffer requires either 'size' or both 'struct_size' and 'element_count'."
            )

        frozen = _freeze_kwargs(kwargs)
        record = self._buffers.get(name)
        if (
            record is not None
            and record.usage == usage
            and record.size == size
            and record.struct_size == struct_size
            and record.element_count == element_count
            and record.extra_params == frozen
        ):
            return record.buffer

        create_kwargs: Dict[str, Any] = {"usage": usage, "label": label or name}
        create_kwargs.update(kwargs)
        if size is not None:
            create_kwargs["size"] = size
        if struct_size is not None:
            create_kwargs["struct_size"] = struct_size
        if element_count is not None:
            create_kwargs["element_count"] = element_count

        buffer = self.device.create_buffer(**create_kwargs)
        self._buffers[name] = _BufferRecord(
            buffer=buffer,
            usage=usage,
            size=size,
            struct_size=struct_size,
            element_count=element_count,
            extra_params=frozen,
        )
        return buffer

    def release_texture(self, name: str) -> None:
        self._textures.pop(name, None)

    def release_buffer(self, name: str) -> None:
        self._buffers.pop(name, None)

    def clear(self) -> None:
        """Drop all cached resources."""
        self._textures.clear()
        self._buffers.clear()

