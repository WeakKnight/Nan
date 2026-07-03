from __future__ import annotations

from typing import Any

from .base import ExtensionContext, ExtensionError

BUILTIN_EXTENSIONS = {
    "static_shadow_sst": "extensions.static_shadow_sst:StaticShadowSSTExtension",
}


def normalize_extension_name(value: str) -> str:
    raw = value.strip()
    if ":" in raw:
        return raw
    name = raw.lower().replace("-", "_")
    aliases = {
        "static_shadow": "static_shadow_sst",
        "sst": "static_shadow_sst",
        "static_sst": "static_shadow_sst",
    }
    return aliases.get(name, name)


class ExtensionManager:
    def __init__(self, names: tuple[str, ...] = ()):
        self._names = tuple(dict.fromkeys(normalize_extension_name(name) for name in names if name.strip()))
        self._extensions: dict[str, Any] = {}
        self._services: dict[str, Any] = {}
        self._service_owners: dict[str, str] = {}
        self._initialized = False

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def initialize(self, app: Any) -> None:
        if self._initialized:
            raise ExtensionError("ExtensionManager.initialize() called more than once")
        for name in self._names:
            if name in self._extensions:
                raise ExtensionError(f"Extension '{name}' is already loaded")
            extension = self._create_extension(name)
            extension_name = normalize_extension_name(getattr(extension, "name", name))
            if ":" not in name and extension_name != name:
                raise ExtensionError(
                    f"Extension factory for '{name}' returned extension named '{extension_name}'"
                )
            if extension_name in self._extensions:
                raise ExtensionError(f"Extension '{extension_name}' is already loaded")
            context = ExtensionContext(app=app, manager=self, name=extension_name)
            try:
                extension.initialize(context)
            except Exception as exc:
                raise ExtensionError(f"Failed to initialize extension '{extension_name}': {exc}") from exc
            self._extensions[extension_name] = extension
        self._initialized = True

    def _create_extension(self, name: str) -> Any:
        target = BUILTIN_EXTENSIONS.get(name)
        if target is None and ":" in name:
            target = name
        if target is None:
            raise ExtensionError(f"Unknown extension '{name}'")

        module_name, class_name = target.split(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        extension_class = getattr(module, class_name)
        return extension_class()

    def register_service(self, service_name: str, service: Any, owner: str) -> None:
        service_name = service_name.strip()
        if not service_name:
            raise ExtensionError("Cannot register an empty extension service name")
        existing_owner = self._service_owners.get(service_name)
        if existing_owner is not None:
            raise ExtensionError(
                f"Extension service '{service_name}' is already registered by '{existing_owner}'"
            )
        self._services[service_name] = service
        self._service_owners[service_name] = owner

    def get_service(self, service_name: str) -> Any | None:
        return self._services.get(service_name)

    def has(self, name: str) -> bool:
        return normalize_extension_name(name) in self._extensions

    def get(self, name: str) -> Any | None:
        return self._extensions.get(normalize_extension_name(name))

    def before_main_loop(self, app: Any) -> None:
        self._call_hook("before_main_loop", app)

    def setup_ui(self, app: Any, ui_context: Any, ui_window: Any) -> None:
        self._call_hook("setup_ui", app, ui_context, ui_window)

    def before_render(
        self,
        command_encoder: Any,
        output: Any,
        frame: int,
        device: Any,
        scene: Any,
        render_data: Any,
    ) -> None:
        self._call_hook("before_render", command_encoder, output, frame, device, scene, render_data)

    def after_render(
        self,
        command_encoder: Any,
        output: Any,
        frame: int,
        device: Any,
        scene: Any,
        render_data: Any,
    ) -> None:
        self._call_hook("after_render", command_encoder, output, frame, device, scene, render_data)

    def create_screen_shadow_mask(
        self,
        command_encoder: Any,
        scene: Any,
        render_data: Any,
        width: int,
        height: int,
    ) -> Any | None:
        provider = self.get_service("screen_shadow_mask_provider")
        if provider is None:
            return None
        return provider.create(
            command_encoder,
            scene,
            render_data,
            width,
            height,
        )

    def _call_hook(self, hook_name: str, *args: Any) -> None:
        for name, extension in self._extensions.items():
            hook = getattr(extension, hook_name, None)
            if hook is None:
                continue
            try:
                hook(*args)
            except Exception as exc:
                raise ExtensionError(f"Extension '{name}' failed in {hook_name}(): {exc}") from exc
