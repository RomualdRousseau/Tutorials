from threading import Thread

import pyray as pr

import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT
from tutorial1.entities import world

_thread = Thread(target=world.reset)


def reset() -> None:
    _thread.start()


def update(_: float) -> str:
    return "gameloop" if not _thread.is_alive() else "loading"


def draw() -> None:
    pr.clear_background(pr.BLUE)  # type: ignore
    prx.draw_text("Loading ...", pr.Vector2(0, WINDOW_HEIGHT * 0.5 - 11), 20, pr.WHITE, align="center")  # type: ignore
