import pyray as pr
from taxi_driver_env.constants import WINDOW_HEIGHT, WINDOW_WIDTH

SCREEN = pr.Rectangle(0, 0, WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)


def init_gamepad():
    for i in range(5):
        pr.trace_log(pr.TraceLogLevel.LOG_INFO, f"GAMEPAD: id: {i} - {pr.get_gamepad_name(i)}")


def draw_text(
    text: str,
    pos: pr.Vector2,
    font_size: int,
    color: pr.Color,
    align: str = "left",
    shadow: bool = False,
):
    assert align in ["left", "right", "center"]
    if align == "left":
        x = int(pos.x)
    elif align == "right":
        len = pr.measure_text(text, font_size)
        x = int(WINDOW_WIDTH - 1 - len - pos.x)
    else:
        len = pr.measure_text(text, font_size)
        x = int(WINDOW_WIDTH / 2 - len / 2 - pos.x)
    if shadow:
        pr.draw_text(text, x + 2, int(pos.y) + 2, font_size, pr.BLACK)  # type: ignore
    pr.draw_text(text, x, int(pos.y), font_size, color)


def draw_line(
    start: pr.Vector2,
    end: pr.Vector2,
    thick: float,
    color: pr.Color,
    rounded: bool,
) -> None:
    aabb = pr.Rectangle(
        min(start.x, end.x),
        min(start.y, end.y),
        abs(end.x - start.x),
        abs(end.y - start.y),
    )
    if pr.get_collision_rec(SCREEN, aabb) is None:
        return

    pr.draw_line_ex(start, end, thick, color)
    if rounded:
        pr.draw_circle_v(start, thick / 2, color)
        pr.draw_circle_v(end, thick / 2, color)


def draw_dashed_line(
    start: pr.Vector2,
    end: pr.Vector2,
    thick: float,
    color: pr.Color,
    dashed: tuple[int, pr.Color],
    rounded: bool,
) -> None:
    aabb = pr.Rectangle(
        min(start.x, end.x),
        min(start.y, end.y),
        abs(end.x - start.x),
        abs(end.y - start.y),
    )
    if pr.get_collision_rec(SCREEN, aabb) is None:
        return

    u = pr.vector2_subtract(end, start)
    u_l = pr.vector2_length(u)
    if u_l == 0:
        return

    step, color2 = dashed

    pr.draw_line_ex(start, end, thick, color2)
    if rounded:
        pr.draw_circle_v(start, thick / 2, color)

    v = pr.vector2_scale(u, 1 / u_l)
    p1 = start
    flip = True
    for _ in range(0, int(u_l - step), step):
        p2 = pr.vector2_add(p1, pr.vector2_scale(v, step))
        if flip:
            pr.draw_line_ex(p1, p2, thick, color)
        p1 = p2
        flip = not flip
    if flip:
        pr.draw_line_ex(p1, end, thick, color)

    if rounded:
        pr.draw_circle_v(end, thick / 2, color if flip else color2)
