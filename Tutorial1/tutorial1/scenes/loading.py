from threading import Thread

import pyray as pr

import tutorial1.resources as res
import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import world

_thread = Thread(target=world.reset)


def reset() -> None:
    _thread.start()


def update(_: float) -> str:
    return "gameloop" if not _thread.is_alive() else "loading"


def draw() -> None:
    tex = res.load_texture("loading")
    pr.draw_texture_pro(
        tex,
        pr.Rectangle(0, 0, tex.width, tex.height),
        pr.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
        pr.Vector2(0, 0),
        0,
        pr.WHITE,  # type: ignore
    )
    prx.draw_text("Loading ...", pr.Vector2(20, WINDOW_HEIGHT - 30), 20, pr.WHITE, align="right")  # type: ignore
