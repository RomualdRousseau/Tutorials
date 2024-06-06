from dataclasses import dataclass

import math
import numpy as np

from tutorial1.util.geom import (
    Point,
    Segment,
    point_in_polygon,
    intersect,
)

import tutorial1.util.graph as graph


@dataclass
class Envelope:
    points: list[Point]
    segments: list[Segment]
    skeleton: list[Segment]
    width: int


def generare_from_spatial_graph(
    graph: graph.SpatialGraph, width: int, step: int
) -> tuple[Envelope, list[Point]]:
    envelopes = _merge_envelopes([_generate_envelope(e, width) for e in graph.edges])

    anchors = []
    for i in range(0, 601, step):
        for j in range(0, 601, step):
            anchor = Point(j, i)
            pip = lambda p: point_in_polygon(anchor, p, False)
            if not any(map(lambda e: pip(e.points), envelopes)):
                anchors.append(anchor)

    return _union_envelopes(envelopes), anchors


def _generate_envelope(
    edge: graph.SpatialEdge, width: int, slices: int = 5
) -> Envelope:
    x1, y1 = edge.start.point.x, edge.start.point.y
    x2, y2 = edge.end.point.x, edge.end.point.y
    vx, vy = np.array([x2 - x1, y2 - y1]) / np.linalg.norm([x2 - x1, y2 - y1])

    points = []
    points.append(Point(x1 - vy * width * 0.5, y1 + vx * width * 0.5))
    points.append(Point(x2 - vy * width * 0.5, y2 + vx * width * 0.5))

    b = math.atan2(vx, -vy)
    for i in range(slices):
        a = b - math.pi * i / slices
        c, s = math.cos(a), math.sin(a)
        points.append(Point(x2 + c * width * 0.5, y2 + s * width * 0.5))

    points.append(Point(x2 + vy * width * 0.5, y2 - vx * width * 0.5))
    points.append(Point(x1 + vy * width * 0.5, y1 - vx * width * 0.5))

    b = math.atan2(-vx, vy)
    for i in range(slices):
        a = b - math.pi * i / slices
        c, s = math.cos(a), math.sin(a)
        points.append(Point(x1 + c * width * 0.5, y1 + s * width * 0.5))

    segments = []
    N = len(points)
    p1 = points[0]
    for i in range(1, N + 1):
        p2 = points[i % N]
        segments.append(Segment(p1, p2))
        p1 = p2

    return Envelope(points, segments, [edge.segment], width)


def _merge_envelopes(
    envelopes: list[Envelope], distance: float = 1, iteration: int = 5
) -> list[Envelope]:
    for _ in range(iteration):
        for e in envelopes:
            for p1 in e.points:
                for o in filter(lambda x: x != e, envelopes):
                    for p2 in o.points:
                        if p1.almost(p2, distance):
                            xm, ym = (p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5
                            p1.x, p1.y = xm, ym
                            p2.x, p2.y = xm, ym
    return envelopes


def _union_envelopes(envelopes: list[Envelope]) -> Envelope:
    points = []
    segments = []
    skeleton = [e.skeleton[0] for e in envelopes]
    width = envelopes[0].width

    for e in envelopes:
        for s in e.segments:
            inside = False
            s_clipped = Segment(s.start, s.end)
            for other in filter(lambda x: x != e, envelopes):
                clip = 0b00
                clip |= (
                    0b01 if point_in_polygon(s_clipped.start, other.points) else 0b00
                )
                clip |= 0b10 if point_in_polygon(s_clipped.end, other.points) else 0b00
                match clip:
                    case 0b01:
                        for ss in other.segments:
                            p = intersect(s_clipped, ss, False)
                            if p:
                                s_clipped = Segment(p, s_clipped.end)
                    case 0b10:
                        for ss in other.segments:
                            p = intersect(s_clipped, ss, False)
                            if p:
                                s_clipped = Segment(s_clipped.start, p)
                    case 0b11:
                        inside = True

            if (
                not inside
                and not s_clipped.end.almost(s_clipped.start)
                and all(map(lambda x: not x.almost(s_clipped), segments))
            ):
                segments.append(s_clipped)

    return Envelope(points, segments, skeleton, width)
