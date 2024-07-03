import pyray as pr

import tutorial1.resources as res
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH

WAIT_TIME = 5  # s

_timer = 0.0


def reset() -> None:
    global _timer  # noqa: PLW0603
    _timer = 0.0


def update(dt: float) -> str:
    global _timer  # noqa: PLW0603
    _timer += dt
    return "loading" if pr.get_key_pressed() != 0 or _timer > WAIT_TIME else "title"


def draw() -> None:
    tex = res.load_texture("title")
    pr.draw_texture_pro(
        tex,
        pr.Rectangle(0, 0, tex.width, tex.height),
        pr.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
        pr.Vector2(0, 0),
        0,
        pr.WHITE,  # type: ignore
    )
