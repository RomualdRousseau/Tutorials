from dataclasses import dataclass

import math

import pyray as pr

from tutorial1.graph import Edge
from tutorial1.math import (
    Point,
    Segment,
    in_polygon,
    intersect,
    norm,
)


@dataclass
class Envelope:
    points: list[Point]

    def segments(self):
        N = len(self.points)
        for i in range(N):
            p1, p2 = self.points[i], self.points[(i + 1) % N]
            yield Segment(p1, p2)

    def draw(self, thick: float, color: pr.Color) -> None:
        for s in self.segments():
            s.draw(thick, color)


def generate_envelope(edge: Edge, thick: int = 10) -> Envelope:
    x1, y1 = edge.start.x, edge.start.y
    x2, y2 = edge.end.x, edge.end.y
    vx, vy = norm(x2 - x1, y2 - y1)

    points = []
    points.append(Point(int(x1 - vy * thick), int(y1 + vx * thick)))
    points.append(Point(int(x2 - vy * thick), int(y2 + vx * thick)))

    b = math.atan2(vx, -vy)
    for i in range(10):
        a = b - math.pi * i / 10
        c, s = math.cos(a), math.sin(a)
        points.append(Point(int(x2 + c * thick), int(y2 + s * thick)))

    points.append(Point(int(x2 + vy * thick), int(y2 - vx * thick)))
    points.append(Point(int(x1 + vy * thick), int(y1 - vx * thick)))

    b = math.atan2(-vx, vy)
    for i in range(10):
        a = b - math.pi * i / 10
        c, s = math.cos(a), math.sin(a)
        points.append(Point(int(x1 + c * thick), int(y1 + s * thick)))

    return Envelope(points)


def union_envelopes(envelopes: list[Envelope]) -> list[Segment]:
    segments = []
    for e in envelopes:
        for s in e.segments():
            inside = False
            s_clipped = Segment(s.start, s.end)
            for other in filter(lambda x: x != e, envelopes):
                clip = 0b00
                clip |= 0b01 if in_polygon(s_clipped.start, other.points) else 0b00
                clip |= 0b10 if in_polygon(s_clipped.end, other.points) else 0b00
                match clip:
                    case 0b01:
                        for ss in other.segments():
                            p = intersect(s, ss, True)
                            if p:
                                s_clipped = Segment(p, s_clipped.end)
                    case 0b10:
                        for ss in other.segments():
                            p = intersect(s, ss, False)
                            if p:
                                s_clipped = Segment(s_clipped.start, p)
                    case 0b11:
                        inside = True

            if (
                not inside
                and all(map(lambda x: not x.almost(s_clipped, 3), segments))
            ):
                segments.append(s_clipped)

    return segments
