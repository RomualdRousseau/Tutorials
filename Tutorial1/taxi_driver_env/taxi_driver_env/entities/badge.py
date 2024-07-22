import numpy as np
import pyray as pr
from taxi_driver_env.math.geom import Point

COLOR = pr.Color(255, 255, 255, 255)
MAX_LIFE = 60


class Badge:
    def __init__(self, pos: Point, text: str) -> None:
        self.pos = pos
        self.text = text
        self.life = MAX_LIFE

    def is_alive(self) -> bool:
        return self.life > 0

    def reset(self) -> None:
        self.life = MAX_LIFE

    def hit(self, damage: int) -> None:
        pass

    def update(self, dt: float) -> None:
        self.pos = Point(np.array([self.pos.xy[0], self.pos.xy[1] - 1]))
        self.life = max(0, self.life - 1)

    def draw(self, layer: int) -> None:
        if layer != 1:
            return

        pr.draw_text(
            self.text,
            int(self.pos.xy[0]),
            int(self.pos.xy[1]),
            1,
            pr.color_alpha(COLOR, (self.life / MAX_LIFE) * (COLOR.a / 256)),
        )
