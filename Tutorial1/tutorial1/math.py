from dataclasses import dataclass
from functools import partial
from math import sqrt
from random import randrange
from typing import Optional

import pyray as pr

@dataclass
class Point:
    x: float
    y: float

    def draw(self, thick: float, color: pr.Color):
        pr.draw_circle_v(self.to_vec(), thick, color)

    def to_vec(self):
        return pr.Vector2(self.x, self.y)

    def almost(self, other, eps = 0.0001):
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps


@dataclass
class Segment:
    start: Point
    end: Point

    def draw(self, thick: float, color: pr.Color):
        pr.draw_line_ex(self.start.to_vec(), self.end.to_vec(), thick, color)

    def almost(self, other, eps = 0.0001):
        return (
            self.start.almost(other.start, eps)
            and self.end.almost(other.end, eps)
            or self.end.almost(other.start, eps)
            and self.start.almost(other.end, eps)
        )

    def __eq__(self, other):
        return (
            self.start == other.start
            and self.end == other.end
            or self.end == other.start
            and self.start == other.end
        )


def randcoord(stop: int) -> int:
    return randrange(0, stop)


def det(a: float, b: float, c: float, d: float) -> float:
    return a * d - b * c


def norm(x: float, y: float) -> tuple[float, float]:
    l = 1 / sqrt(x**2 + y**2)
    return x * l, y * l


def lerp(x1: float, x2: float, t: float):
    return x1 * (1 - t) + x2 * t


def within(x: float, dom: tuple[float, float] = (0, 1), strict: bool = True) -> bool:
    return dom[0] < x < dom[1] if strict else dom[0] <= x <= dom[1]


def distance(p1: Point, p2: Point) -> float:
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def intersect(seg1: Segment, seg2: Segment, strict: bool = True) -> Optional[Point]:
    x1, y1 = seg1.start.x, seg1.start.y
    x2, y2 = seg1.end.x, seg1.end.y
    x3, y3 = seg2.start.x, seg2.start.y
    x4, y4 = seg2.end.x, seg2.end.y

    in_seg = partial(within, dom=(0, 1), strict=strict)

    match det(x1 - x2, y1 - y2, x3 - x4, y3 - y4):
        case d if d != 0:
            match -det(x1 - x2, y1 - y2, x1 - x3, y1 - y3) / d:
                case u if in_seg(u):
                    match det(x1 - x3, y1 - y3, x3 - x4, y3 - y4) / d:
                        case t if in_seg(t):
                            return Point(int(lerp(x1, x2, t)), int(lerp(y1, y2, t)))
                        case _:
                            return None
                case _:
                    return None
        case _:
            return None


def in_polygon(point: Point, polygon: list[Point]) -> bool:
    x, y = point.x, point.y
    N = len(polygon)
    inside = False

    p1 = polygon[0]
    for i in range(1, N + 1):
        p2 = polygon[i % N]
        if y > min(p1.y, p2.y):
            if y <= max(p1.y, p2.y):
                if x <= max(p1.x, p2.x):
                    if p1.y != p2.y:
                        xinters = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if p1.x == p2.x or x < xinters:
                        inside = not inside
        p1 = p2
    return inside
