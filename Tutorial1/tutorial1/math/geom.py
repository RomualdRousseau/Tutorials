from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional

import pyray as pr
import numpy as np

from tutorial1.util.funcs import curry
import tutorial1.util.pyray_ex as prx


@dataclass
class Point:
    xy: np.ndarray

    def draw(self, thick: float, color: pr.Color):
        pr.draw_circle_v(self.to_vec(), thick, color)

    def to_vec(self):
        return pr.Vector2(*self.xy)

    def almost(self, other: Point, eps=0.0001):
        return np.allclose(self.xy, other.xy, 0, eps)

    def __eq__(self, other):
        return isinstance(other, Point) and np.all(self.xy == other.xy)


@dataclass
class Segment:
    start: Point
    end: Point

    @property
    def length(self):
        return distance(self.start, self.end)

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
    return float(np.linalg.norm(p2.xy - p1.xy))


def angle(seg: Segment) -> float:
    x1, y1 = seg.start.xy
    x2, y2 = seg.end.xy
    return np.arctan2(y2 - y1, x2 - x1)


def intersect(seg1: Segment, seg2: Segment, strict: bool = True) -> Optional[Point]:
    x1, y1 = seg1.start.xy
    x2, y2 = seg1.end.xy
    x3, y3 = seg2.start.xy
    x4, y4 = seg2.end.xy

    atol = -1e-08 if strict else 0.0
    within_0_and_1 = partial(np.isclose, b=0.5, rtol=1.0, atol=atol)

    match np.linalg.det([[x1 - x2, y1 - y2], [x3 - x4, y3 - y4]]):
        case d if d != 0:
            match -np.linalg.det([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]]) / d:
                case u if within_0_and_1(u):
                    match np.linalg.det([[x1 - x3, y1 - y3], [x3 - x4, y3 - y4]]) / d:
                        case t if within_0_and_1(t):
                            return Point(
                                np.array(
                                    [
                                        np.interp(t, [0, 1], [x1, x2]),
                                        np.interp(t, [0, 1], [y1, y2]),
                                    ]
                                )
                            )
                        case _:
                            return None
                case _:
                    return None
        case _:
            return None


def point_on_segment(p: Point, seg: Segment) -> bool:
    px, py = p.xy
    sx, sy = seg.start.xy
    ex, ey = seg.end.xy

    if not (min(sx, ex) <= px <= max(sx, ex) and min(sy, ey) <= py <= max(sy, ey)):
        return False

    return (ex - sx) * (py - sy) == (ey - sy) * (px - sx)


def point_in_polygon(point: Point, polygon: list[Point], strict: bool = True) -> bool:
    x, y = point.xy
    N = len(polygon)
    inside = False

    p1, (p1x, p1y) = polygon[0], polygon[0].xy
    for i in range(1, N + 1):
        p2, (p2x, p2y) = polygon[i % N], polygon[i % N].xy

        if strict and point_on_segment(point, Segment(p1, p2)):
            return False

        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            xinters = np.interp(y, [p1y, p2y], [p1x, p2x], period=np.inf)
            if p1x == p2x or x < xinters:
                inside = not inside
        p1, (p1x, p1y) = p2, (p2x, p2y)
    return inside


def distance_point_segment(p: Point, seg: Segment) -> float:
    u = p.xy - seg.start.xy
    v = seg.end.xy - seg.start.xy
    l = np.linalg.norm(v)
    v = v / l
    match np.dot(u, v):
        case i if 0 <= i <= l:
            i = seg.start.xy + v * i
            return float(np.linalg.norm(p.xy - i))
        case _:
            return np.Infinity


def nearest_point_segment(p: Point, seg: Segment) -> Optional[Point]:
    u = p.xy - seg.start.xy
    v = seg.end.xy - seg.start.xy
    l = np.linalg.norm(v)
    v = v / l
    match np.dot(u, v):
        case i if 0 <= i <= l:
            return Point(seg.start.xy + v * i)
        case _:
            return None


def collision_circle_segment(
    center: Point, radius: float, seg: Segment
) -> Optional[np.ndarray]:
    u = center.xy - seg.start.xy
    v = seg.end.xy - seg.start.xy
    v_l = np.linalg.norm(v)
    v = v / v_l
    match np.dot(u, v):
        case i if 0 <= i <= v_l:
            i = seg.start.xy + v * i
            w = center.xy - i
            match float(np.linalg.norm(w)):
                case w_d if w_d <= radius:
                    return w * (radius - w_d) / w_d
                case _:
                    return None
        case _:
            return None


def cast_ray_segments(
    position: Point, direction: np.ndarray, length: float, segments: Iterable[Segment]
) -> Segment:
    target = Point(length * direction + position.xy)
    ray = Segment(position, target)
    map_not_none = curry(filter)(lambda x: x is not None)
    inter_ray = curry(intersect)(ray)
    closest = curry(distance)(position)
    point = min(map_not_none(map(inter_ray, segments)), key=closest, default=target)
    return Segment(position, point)


def points_to_segments(points: list[Point], closed: bool = True) -> list[Segment]:
    segments = []
    N = len(points)
    p1 = points[0]
    for i in range(1, N + 1 if closed else N):
        p2 = points[i % N]
        segments.append(Segment(p1, p2))
        p1 = p2
    return segments


def break_segment(seg1: Segment, seg2: Segment) -> list[Segment]:
    p = intersect(seg1, seg2, False)
    if p is not None and not seg2.start.almost(p) and not seg2.end.almost(p):
        return [Segment(seg2.start, p), Segment(p, seg2.end)]
    else:
        return [seg2]
