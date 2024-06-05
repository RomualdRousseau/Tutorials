import pyray as pr


def update(_: float) -> str:
    return "gameloop" if pr.get_key_pressed() != 0 else "title"


def draw() -> None:
    pr.clear_background(pr.BLUE)  # type: ignore
    pr.draw_text("Press any key ...", 200, 300, 20, pr.WHITE)  # type: ignore
