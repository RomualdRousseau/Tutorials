import pyray as pr

from tutorial1.math.geom import Point

COLOR = pr.Color(255, 255, 255, 128)
RADIUS = 3
MAX_LIFE = 30


class Explosion:
    def __init__(self, pos: Point) -> None:
        self.pos = pos
        self.life = MAX_LIFE

    def is_alive(self) -> bool:
        return self.life > 0

    def reset(self) -> None:
        self.life = MAX_LIFE

    def hit(self, damage: int) -> None:
        pass

    def update(self, dt: float) -> None:
        self.life = max(0, self.life - 1)

    def draw(self, layer: int) -> None:
        if layer != 1:
            return

        pr.draw_circle_v(self.pos.to_vec(), self.life * RADIUS / MAX_LIFE, pr.color_alpha(COLOR, self.life * 128 / MAX_LIFE))
