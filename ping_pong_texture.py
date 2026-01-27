import slangpy as spy
from render_data import RenderData


class PingPongTexture:
    def __init__(self, format: spy.Format, label: str):
        self.ping: spy.Texture = None
        self.pong: spy.Texture = None
        self.format = format
        self.label = label
        self._ping_key = f"{label}.ping"
        self._pong_key = f"{label}.pong"

    def validate(self, render_data: RenderData, w: int, h: int):
        if (
            self.ping is not None
            and self.pong is not None
            and self.ping.width == w
            and self.ping.height == h
            and self.pong.width == w
            and self.pong.height == h
        ):
            return

        self.ping = render_data.get_texture(
            self._ping_key,
            width=w,
            height=h,
            format=self.format,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"{self.label}_ping",
        )
        self.pong = render_data.get_texture(
            self._pong_key,
            width=w,
            height=h,
            format=self.format,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"{self.label}_pong",
        )

    def swap(self):
        self.ping, self.pong = self.pong, self.ping
        self._ping_key, self._pong_key = self._pong_key, self._ping_key
