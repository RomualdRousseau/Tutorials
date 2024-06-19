import pyray as pr

from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH


def reset() -> None:
    pr.trace_log(pr.TraceLogLevel.LOG_DEBUG, "TITLE: reset")


def update(_: float) -> str:
    return "gameloop" if pr.get_key_pressed() != 0 else "title"


def draw() -> None:
    pr.clear_background(pr.BLUE)  # type: ignore
    pr.draw_text("Press any key ...", WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2, 20, pr.WHITE)  # type: ignore
