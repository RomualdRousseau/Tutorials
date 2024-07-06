import pyray as pr

from tutorial1.constants import FRAME_RATE, WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.math.linalg import EPS, clamp


class FadeInOut:

    def __init__(self, min: int, max: int, duration: float, color: pr.Color) -> None:
        self.duration = duration
        self.timer = 0
        self.min = min
        self.max = max
        self.color = color

    def is_playing(self, latency: float = 0.1) -> bool:
        return self.timer < self.duration + latency
    
    def reset(self) -> None:
        self.timer = 0

    def update(self, dt: float) -> None:
        self.timer = self.timer + min(dt, 1 / FRAME_RATE)

    def draw(self) -> None:
        t = self.timer / (self.duration + EPS)
        alpha = float(clamp(self.max * (1.0 - t) + self.min * t, 0, 255))
        if alpha > 0.0:
            pr.draw_rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, pr.fade(self.color, alpha / 255.0))
