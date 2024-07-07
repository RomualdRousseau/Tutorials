import pyray as pr

from tutorial1.constants import FRAME_RATE
from tutorial1.math.linalg import EPS, clamp
from tutorial1.utils.types import Widget


class OpenVertical:

    def __init__(self, widget: Widget, duration: float, color: pr.Color):
        self.duration = duration
        self.widget = widget
        self.color = color

    def get_bound(self) -> pr.Rectangle:
        t = self.timer / (self.duration + EPS)
        height = int(clamp(self.widget.get_bound().height * t, 0, self.widget.get_bound().height))
        return pr.Rectangle(
            self.widget.get_bound().x,
            self.widget.get_bound().y + (self.widget.get_bound().height - height) / 2,
            self.widget.get_bound().width,
            height,
        )

    def is_playing(self, latency: float = 0.1) -> bool:
        return self.timer < self.duration + latency

    def reset(self) -> None:
        self.timer = 0.0

    def update(self, dt: float):
        self.timer = self.timer + min(dt, 1 / FRAME_RATE)
        if not self.is_playing():
            self.widget.update(dt)

    def draw(self):
        if not self.is_playing():
            self.widget.draw()
        else:
            bound = self.get_bound()
            if bound.height > 0.0:
                pr.draw_rectangle_rec(bound, self.color)
