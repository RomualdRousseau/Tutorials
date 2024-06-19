import pyray as pr

from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH

SCREEN = pr.Rectangle(0, 0, WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)


def draw_text_shadow(text: str, pos: pr.Vector2, font_size: int, color: pr.Color):
    pr.draw_text(text, int(pos.x + 1), int(pos.y + 1), font_size, pr.BLACK)  # type: ignore
    pr.draw_text(text, int(pos.x), int(pos.y), font_size, color)


def draw_line(
    start: pr.Vector2,
    end: pr.Vector2,
    thick: float,
    color: pr.Color,
    rounded: bool,
) -> None:
    aabb = pr.Rectangle(start.x, start.y, end.x, end.y)
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
    aabb = pr.Rectangle(start.x, start.y, end.x, end.y)
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
