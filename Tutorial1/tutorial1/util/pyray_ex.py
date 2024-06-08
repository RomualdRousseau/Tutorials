import pyray as pr


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
    u = pr.vector2_subtract(end, start)
    l = pr.vector2_length(u)
    if l == 0:
        return

    step, color2 = dashed
    
    pr.draw_line_ex(start, end, thick, color2)
    if rounded:
        pr.draw_circle_v(start, thick / 2, color)

    v = pr.vector2_scale(u, 1 / l)
    p1 = start
    flip = True
    for _ in range(0, int(l - step), step):
        p2 = pr.vector2_add(p1, pr.vector2_scale(v, step))
        if flip:
            pr.draw_line_ex(p1, p2, thick, color)
        p1 = p2
        flip = not flip   
    if flip:
        pr.draw_line_ex(p1, end, thick, color)
        
    if rounded:
        pr.draw_circle_v(end, thick / 2, color if flip else color2)
