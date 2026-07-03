from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ExtensionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtensionContext:
    app: Any
    manager: Any
    name: str

    @property
    def device(self) -> Any:
        return self.app.device

    @property
    def scene(self) -> Any:
        return self.app.scene

    def register_service(self, service_name: str, service: Any) -> None:
        self.manager.register_service(service_name, service, owner=self.name)

    def get_service(self, service_name: str) -> Any | None:
        return self.manager.get_service(service_name)


@runtime_checkable
class RenderExtension(Protocol):
    name: str

    def initialize(self, context: ExtensionContext) -> None:
        ...
