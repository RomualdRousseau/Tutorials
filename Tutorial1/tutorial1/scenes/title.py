import pyray as pr

import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT


def reset() -> None:
    pr.trace_log(pr.TraceLogLevel.LOG_DEBUG, "TITLE: reset")


def update(_: float) -> str:
    return "gameloop" if pr.get_key_pressed() != 0 else "title"


def draw() -> None:
    pr.clear_background(pr.BLUE)  # type: ignore
    prx.draw_text("Press any key ...", pr.Vector2(0, WINDOW_HEIGHT * 0.5 - 11), 20, pr.WHITE, align="center")  # type: ignore
