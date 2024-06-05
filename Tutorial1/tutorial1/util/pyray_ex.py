import pyray as pr


def draw_line(
    start: pr.Vector2,
    end: pr.Vector2,
    thick: float,
    color: pr.Color,
    rounded: bool,
) -> None:
    if rounded and thick > 1:
        pr.draw_circle_v(start, thick / 2, color)
    pr.draw_line_ex(start, end, thick, color)
    if rounded and thick > 1:
        pr.draw_circle_v(end, thick / 2, color)


def draw_dashed_line(
    start: pr.Vector2,
    end: pr.Vector2,
    thick: float,
    color: pr.Color,
    dashed: tuple[int, pr.Color],
    rounded: bool,
) -> None:
    u = pr.vector2_subtract(end, start)
    l = pr.vector2_length(u)
    if l == 0:
        return

    if rounded and thick > 1:
        pr.draw_circle_v(start, thick / 2, color)

    step, color2 = dashed
    v = pr.vector2_scale(u, 1 / l)
    p1 = start
    flip = True
    for _ in range(0, int(l - step), step):
        p2 = pr.vector2_add(p1, pr.vector2_scale(v, step))
        pr.draw_line_ex(p1, p2, thick, color if flip else color2)
        p1 = p2
        flip = not flip

    pr.draw_line_ex(p1, end, thick, color if flip else color2)
    if rounded and thick > 1:
        pr.draw_circle_v(end, thick / 2, color if flip else color2)
