from dataclasses import dataclass
from functools import reduce
from typing import Iterable

import numpy as np
from tqdm import tqdm, trange

from tutorial1.constants import VIRTUAL_WIDTH
from tutorial1.math import graph
from tutorial1.math.geom import (
    Point,
    Segment,
    break_segment,
    point_in_polygon,
    points_to_segments,
)
from tutorial1.math.linalg import lst_2_vec, normalize


@dataclass
class Envelope:
    segments: list[Segment]
    skeleton: list[Segment]
    width: int

    @property
    def points(self) -> list[Point]:
        return [s.start for s in self.segments]


def generare_from_spatial_graph(agraph: graph.SpatialGraph, width: int) -> tuple[Envelope, list[Point]]:
    with tqdm(total=4, desc="Generating envelope", ncols=120, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        envelopes = [_generate_envelope(e, width) for e in agraph.edges]
        pbar.update(1)

        anchors = _generate_anchors(envelopes)
        pbar.update(1)

        envelopes = _break_envelopes(envelopes)
        pbar.update(1)

        envelope = _union_envelopes(envelopes)
        pbar.update(1)

    return envelope, anchors


def _generate_envelope(edge: graph.SpatialEdge, width: int, slices: int = 10) -> Envelope:
    x1, y1 = edge.start.point.xy
    x2, y2 = edge.end.point.xy
    vx, vy = normalize(lst_2_vec([x2 - x1, y2 - y1]))

    points = []
    points.append(Point(lst_2_vec([x1 - vy * width * 0.5, y1 + vx * width * 0.5])))
    points.append(Point(lst_2_vec([x2 - vy * width * 0.5, y2 + vx * width * 0.5])))

    b = np.arctan2(vx, -vy)
    for i in range(slices):
        a = b - np.pi * i / slices
        c, s = np.cos(a), np.sin(a)
        points.append(Point(lst_2_vec([x2 + c * width * 0.5, y2 + s * width * 0.5])))

    points.append(Point(lst_2_vec([x2 + vy * width * 0.5, y2 - vx * width * 0.5])))
    points.append(Point(lst_2_vec([x1 + vy * width * 0.5, y1 - vx * width * 0.5])))

    b = np.arctan2(-vx, vy)
    for i in range(slices):
        a = b - np.pi * i / slices
        c, s = np.cos(a), np.sin(a)
        points.append(Point(lst_2_vec([x1 + c * width * 0.5, y1 + s * width * 0.5])))

    segments = points_to_segments(points)

    return Envelope(segments, [edge.segment], width)


def _generate_anchors(envelopes: list[Envelope], step: int = 20):
    def anchors() -> Iterable[Point]:
        for i in range(-VIRTUAL_WIDTH, VIRTUAL_WIDTH + 1, step):
            for j in range(-VIRTUAL_WIDTH, VIRTUAL_WIDTH + 1, step):
                yield Point(lst_2_vec([j, i]))

    anchor_in_polygon = lambda x, y: point_in_polygon(x, y.points)
    anchor_in_polygons = lambda x: not any((anchor_in_polygon(x, y) for y in envelopes))
    return [x for x in anchors() if anchor_in_polygons(x)]


def _break_envelopes(envelopes: list[Envelope]) -> list[Envelope]:
    def break_envelope(e: Envelope, s: Segment) -> Envelope:
        segments: list[Segment] = sum((break_segment(s, x) for x in e.segments), [])
        return Envelope(segments, e.skeleton, e.width)

    def break_two_envelopes(e1: Envelope, e2: Envelope) -> tuple[Envelope, Envelope]:
        return (
            reduce(break_envelope, e2.segments, e1),
            reduce(break_envelope, e1.segments, e2),
        )

    result = envelopes
    n = len(envelopes)

    for i in trange(n, desc="Break", ncols=120, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        head, acc = result[i], []
        for j in range(i + 1, n):
            head, tail = break_two_envelopes(head, result[j])
            acc.append(tail)
        result = [*result[:i], head, *acc]
    return result


def _union_envelopes(envelopes: list[Envelope]) -> Envelope:
    segments_to_keep: list[Segment] = []
    skeleton = [e.skeleton[0] for e in envelopes]
    width = envelopes[0].width

    segment_in_polygon = lambda y: lambda x: point_in_polygon(y.middle, x.points)

    for e in tqdm(envelopes, desc="Union", ncols=120, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        for s in e.segments:
            inside = any(
                map(
                    segment_in_polygon(s),
                    filter(lambda x: x != e, envelopes),
                )
            )
            if not (inside or s.start.almost(s.end) or any((x.almost(s) for x in segments_to_keep))):
                segments_to_keep.append(s)

    return Envelope(segments_to_keep, skeleton, width)
