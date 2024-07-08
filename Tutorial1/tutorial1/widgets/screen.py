import pyray as pr

import tutorial1.resources as res
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH


class Screen:
    def __init__(self, texture_name: str):
        self.texture_name = texture_name

    def get_bound(self) -> pr.Rectangle:
        return pr.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    def reset(self) -> None:
        self.surface = res.load_texture(self.texture_name)

    def update(self, _: float) -> None:
        pass

    def draw(self):
        pr.draw_texture_pro(
            self.surface,
            pr.Rectangle(0, 0, self.surface.width, self.surface.height),
            pr.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
            pr.Vector2(0, 0),
            0,
            pr.WHITE,  # type: ignore
        )
