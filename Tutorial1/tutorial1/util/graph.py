from dataclasses import dataclass
from random import choice

import random
import numpy as np
import pyray as pr

from tutorial1.util.geom import (
    Point,
    Segment,
    distance,
    intersect,
)


EDGE_COLOR = pr.Color(0, 0, 0, 128)
VERTEX_COLOR = pr.Color(0, 0, 255, 128)


@dataclass
class SpatialVertex:
    point: Point

    def draw(self):
        self.point.draw(5.0, VERTEX_COLOR)


@dataclass
class SpatialEdge:
    start: SpatialVertex
    end: SpatialVertex

    @property
    def segment(self):
        return Segment(self.start.point, self.end.point)

    def draw(self):
        self.segment.draw(1.0, EDGE_COLOR)


@dataclass
class SpatialGraph:
    vertice: list[SpatialVertex]
    edges: list[SpatialEdge]

    def draw(self):
        for seg in self.edges:
            seg.draw()
        for vertex in self.vertice:
            vertex.draw()


def generate_random(seed: int):
    random.seed(seed)
    vertice = _generate_vertice(20, 100)
    edges = _generate_edges(vertice, 25, 3)
    return SpatialGraph(vertice, edges)


def _generate_vertice(num: int, min: int) -> list[SpatialVertex]:
    vertice = []
    n = 0
    while n < num:
        v = SpatialVertex(Point(np.array([random.randrange(50, 550), random.randrange(50, 550)])))
        is_valid = lambda x: distance(x.point, v.point) > min
        if all(map(is_valid, vertice)):
            vertice.append(v)
            n += 1
    return vertice


def _generate_edges(vertice: list[SpatialVertex], num: int, k: int = 3):
    edges = []
    n = 0
    while n < num:
        e = _generate_edge(vertice, k)
        is_valid = lambda x: x.segment != e.segment and not intersect(
            x.segment, e.segment
        )
        if all(map(is_valid, edges)):
            edges.append(e)
            n += 1
    return edges


def _generate_edge(vertice: list[SpatialVertex], k: int) -> SpatialEdge:
    start = choice(vertice)
    points_minus_start = filter(lambda x: x != start, vertice)
    distance_to_start = lambda x: distance(start.point, x.point)
    choices = sorted(points_minus_start, key=distance_to_start)
    end = choice(choices[:k])
    return SpatialEdge(start, end)
