from dataclasses import dataclass
from functools import reduce

import numpy as np
import pyray as pr

from tutorial1.constants import VIRTUAL_MARGIN, VIRTUAL_WIDTH
from tutorial1.math.geom import (
    Point,
    Segment,
    break_segment,
    point_in_polygon,
    points_to_segments,
)

import tutorial1.math.graph as graph
from tutorial1.util.funcs import curry, flatmap


@dataclass
class Envelope:
    segments: list[Segment]
    skeleton: list[Segment]
    width: int

    @property
    def points(self) -> list[Point]:
        return list(map(lambda s: s.start, self.segments))


def generare_from_spatial_graph(
    graph: graph.SpatialGraph, width: int
) -> tuple[Envelope, list[Point]]:
    pr.trace_log(pr.TraceLogLevel.LOG_INFO, "ENVELOPE: Generate Envelopes")
    envelopes = [_generate_envelope(e, width) for e in graph.edges]

    pr.trace_log(pr.TraceLogLevel.LOG_INFO, "ENVELOPE: Generate Anchors")
    anchors = _generate_anchors(envelopes)

    pr.trace_log(pr.TraceLogLevel.LOG_INFO, "ENVELOPE: Break Envelopes")
    envelopes = _break_envelopes(envelopes)

    pr.trace_log(pr.TraceLogLevel.LOG_INFO, "ENVELOPE: Union Envelopes")
    envelopes = _union_envelopes(envelopes)

    return envelopes, anchors


def _generate_envelope(
    edge: graph.SpatialEdge, width: int, slices: int = 10
) -> Envelope:
    x1, y1 = edge.start.point.xy
    x2, y2 = edge.end.point.xy
    vx, vy = np.array([x2 - x1, y2 - y1]) / np.linalg.norm([x2 - x1, y2 - y1])

    points = []
    points.append(Point(np.array([x1 - vy * width * 0.5, y1 + vx * width * 0.5])))
    points.append(Point(np.array([x2 - vy * width * 0.5, y2 + vx * width * 0.5])))

    b = np.arctan2(vx, -vy)
    for i in range(slices):
        a = b - np.pi * i / slices
        c, s = np.cos(a), np.sin(a)
        points.append(Point(np.array([x2 + c * width * 0.5, y2 + s * width * 0.5])))

    points.append(Point(np.array([x2 + vy * width * 0.5, y2 - vx * width * 0.5])))
    points.append(Point(np.array([x1 + vy * width * 0.5, y1 - vx * width * 0.5])))

    b = np.arctan2(-vx, vy)
    for i in range(slices):
        a = b - np.pi * i / slices
        c, s = np.cos(a), np.sin(a)
        points.append(Point(np.array([x1 + c * width * 0.5, y1 + s * width * 0.5])))

    segments = points_to_segments(points)

    return Envelope(segments, [edge.segment], width)


def _generate_anchors(envelopes: list[Envelope], step: int = 20):
    width = VIRTUAL_WIDTH + VIRTUAL_MARGIN * 2 + 1
    anchors = []
    for i in range(0, width, step):
        for j in range(0, width, step):
            anchor = Point(np.array([j, i]))
            anchor_in_polygon = lambda e: point_in_polygon(anchor, e.points, False)
            if not any(map(anchor_in_polygon, envelopes)):
                anchors.append(anchor)
    return anchors


def _break_envelopes(envelopes: list[Envelope]) -> list[Envelope]:

    def _break_envelope(e: Envelope, s: Segment) -> Envelope:
        segments = flatmap(curry(break_segment)(s), e.segments)
        return Envelope(segments, e.skeleton, e.width)

    def _break_two_envelopes(e1: Envelope, e2: Envelope) -> tuple[Envelope, Envelope]:
        return (
            reduce(_break_envelope, e2.segments, e1),
            reduce(_break_envelope, e1.segments, e2),
        )

    result = envelopes
    N = len(envelopes)

    for i in range(N):
        head, tail = result[i], []
        for j in range(i + 1, N):
            head, other = _break_two_envelopes(head, result[j])
            tail.append(other)
        result = result[:i] + [head] + tail
    return result


def _union_envelopes(envelopes: list[Envelope]) -> Envelope:
    segments_to_keep = []
    skeleton = [e.skeleton[0] for e in envelopes]
    width = envelopes[0].width

    for e in envelopes:
        for s in e.segments:
            mid = Point(s.start.xy * 0.5 + s.end.xy * 0.5)
            segment_in_polygon = lambda x: point_in_polygon(mid, x.points)
            inside = any(map(segment_in_polygon, filter(lambda x: x != e, envelopes)))
            if not (
                inside
                or s.start.almost(s.end)
                or any(map(lambda x: x.almost(s), segments_to_keep))
            ):
                segments_to_keep.append(s)

    return Envelope(segments_to_keep, skeleton, width)
