from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import math
from typing import Optional

import pyray as pr
import numpy as np

import tutorial1.util.pyray_ex as prx


@dataclass(unsafe_hash=True)
class Point:
    x: float
    y: float

    def draw(self, thick: float, color: pr.Color):
        pr.draw_circle_v(self.to_vec(), thick, color)

    def to_vec(self):
        return pr.Vector2(self.x, self.y)

    def to_np(self):
        return np.array([self.x, self.y])

    def almost(self, other: Point, eps=0.0001):
        return abs(self.x - other.x) <= eps and abs(self.y - other.y) <= eps


@dataclass(unsafe_hash=True)
class Segment:
    start: Point
    end: Point

    def draw(
        self,
        thick: float,
        color: pr.Color,
        dashed: tuple[int, pr.Color] | None = None,
        rounded: bool = False,
    ):
        if dashed:
            prx.draw_dashed_line(
                self.start.to_vec(), self.end.to_vec(), thick, color, dashed, rounded
            )
        else:
            prx.draw_line(self.start.to_vec(), self.end.to_vec(), thick, color, rounded)

    def almost(self, other: Segment, eps=0.0001):
        return (
            self.start.almost(other.start, eps)
            and self.end.almost(other.end, eps)
            or self.end.almost(other.start, eps)
            and self.start.almost(other.end, eps)
        )

    def __eq__(self, other):
        return isinstance(other, Segment) and (
            self.start == other.start
            and self.end == other.end
            or self.end == other.start
            and self.start == other.end
        )


def distance(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(p2.to_np() - p1.to_np()))

def angle(seg: Segment) -> float:
    x1, y1 = seg.start.x, seg.start.y
    x2, y2 = seg.end.x, seg.end.y
    return np.arctan2(y2 - y1, x2 - x1)

def intersect(seg1: Segment, seg2: Segment, strict: bool = True) -> Optional[Point]:
    x1, y1 = seg1.start.x, seg1.start.y
    x2, y2 = seg1.end.x, seg1.end.y
    x3, y3 = seg2.start.x, seg2.start.y
    x4, y4 = seg2.end.x, seg2.end.y

    atol = -1e-08 if strict else 0.0
    within_0_and_1 = partial(np.isclose, b=0.5, rtol=1.0, atol=atol)

    match np.linalg.det([[x1 - x2, y1 - y2], [x3 - x4, y3 - y4]]):
        case d if d != 0:
            match -np.linalg.det([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]]) / d:
                case u if within_0_and_1(u):
                    match np.linalg.det([[x1 - x3, y1 - y3], [x3 - x4, y3 - y4]]) / d:
                        case t if within_0_and_1(t):
                            return Point(
                                float(np.interp(t, [0, 1], [x1, x2])),
                                float(np.interp(t, [0, 1], [y1, y2])),
                            )
                        case _:
                            return None
                case _:
                    return None
        case _:
            return None

def point_on_segment(p: Point, seg: Segment) -> bool:
    px, py = p.x, p.y
    sx, sy = seg.start.x, seg.start.y
    ex, ey = seg.end.x, seg.end.y

    if not (min(sx, ex) <= px <= max(sx, ex) and min(sy, ey) <= py <= max(sy, ey)):
        return False

    return (ex - sx) * (py - sy) == (ey - sy) * (px - sx)


def point_in_polygon(point: Point, polygon: list[Point], strict: bool = True) -> bool:
    x, y = point.x, point.y
    N = len(polygon)
    inside = False

    p1 = polygon[0]
    for i in range(1, N + 1):
        p2 = polygon[i % N]

        if strict and point_on_segment(point, Segment(p1, p2)):
            return False

        if min(p1.y, p2.y) < y <= max(p1.y, p2.y) and x <= max(p1.x, p2.x):
            xinters = np.interp(y, [p1.y, p2.y], [p1.x, p2.x], period=np.inf)
            if p1.x == p2.x or x < xinters:
                inside = not inside
        p1 = p2
    return inside


def distance_point_segment(p: Point, seg: Segment) -> float:
    u = p.to_np() - seg.start.to_np()
    v = seg.end.to_np() - seg.start.to_np()
    l = np.linalg.norm(v)
    v = v / l
    match np.dot(u, v):
        case i if 0 <= i <= l:
            i = seg.start.to_np() + v * i
            return float(np.linalg.norm(p.to_np() - i))
        case _:
            return np.Infinity
