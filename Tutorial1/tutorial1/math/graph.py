from __future__ import annotations
from dataclasses import dataclass
import heapq
from random import choice
from queue import PriorityQueue

import random
from typing import Iterable
import numpy as np
import pyray as pr

from tutorial1.constants import VIRTUAL_WIDTH, VIRTUAL_MARGIN
from tutorial1.math.geom import (
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

    def draw(self, color=VERTEX_COLOR):
        self.point.draw(5.0, color)

    def __lt__(self, other):
        return distance(self.point, other.point) < 0

    def __hash__(self) -> int:
        return self.point.__hash__()


@dataclass
class SpatialEdge:
    start: SpatialVertex
    end: SpatialVertex

    @property
    def segment(self):
        return Segment(self.start.point, self.end.point)

    def draw(self, color=EDGE_COLOR):
        self.segment.draw(1.0, color)


@dataclass
class SpatialGraph:
    vertice: list[SpatialVertex]
    edges: list[SpatialEdge]

    def get_edges_from_vextex(
        self, v: SpatialVertex
    ) -> Iterable[tuple[SpatialEdge, SpatialVertex]]:
        is_neighbor = lambda x: x.start == v or x.end == v
        other_vertex = lambda x: (x, x.end if x.start == v else x.start)
        return map(other_vertex, filter(is_neighbor, self.edges))

    def get_shortest_path(
        self, start: SpatialVertex, stop: SpatialVertex
    ) -> SpatialGraph:
        weight = lambda x: x[0]
        distances = {x: 0.0 if x == start else np.inf for x in self.vertice}
        unvisited = sorted(map(lambda x: (distances[x], x), self.vertice), key=weight)
        prev = {}
        
        while unvisited:
            dist_u, u = heapq.heappop(unvisited)
            if u == stop:
                break
            for e, v in self.get_edges_from_vextex(u):
                alt = dist_u + e.segment.length
                if alt < distances[v]:
                    prev[v] = u
                    distances[v] = alt
                    heapq.heappush(unvisited, (alt, v))

        vertice = []
        u = stop
        while prev.get(u) and u != start:
            vertice = [u] + vertice
            u = prev[u]
        vertice = [start] + vertice
        
        edges = []
        for i in range(len(vertice) - 1):
            edges.append(SpatialEdge(vertice[i], vertice[i + 1]))

        return SpatialGraph(vertice, edges)

    def draw(self):
        for seg in self.edges:
            seg.draw()
        for vertex in self.vertice:
            vertex.draw()


def generate_random(seed: int):
    
    def generate_vertice(num: int, min: int) -> list[SpatialVertex]:
        rand = lambda: random.randrange(0, VIRTUAL_WIDTH) + VIRTUAL_MARGIN
        vertice = []
        n = 0
        while n < num:
            v = SpatialVertex(Point(np.array([rand(), rand()])))
            is_valid = lambda x: distance(x.point, v.point) > min
            if all(map(is_valid, vertice)):
                vertice.append(v)
                n += 1
        return vertice

    def generate_edges(vertice: list[SpatialVertex], num: int, k: int = 3):
        edges = []
        n = 0
        while n < num:
            e = generate_edge(vertice, k)
            is_valid = lambda x: x.segment != e.segment and not intersect(
                x.segment, e.segment
            )
            if all(map(is_valid, edges)):
                edges.append(e)
                n += 1
        return edges

    def generate_edge(vertice: list[SpatialVertex], k: int) -> SpatialEdge:
        start = choice(vertice)
        points_minus_start = filter(lambda x: x != start, vertice)
        distance_to_start = lambda x: distance(start.point, x.point)
        choices = sorted(points_minus_start, key=distance_to_start)
        end = choice(choices[:k])
        return SpatialEdge(start, end)

    pr.trace_log(pr.TraceLogLevel.LOG_INFO, f"GRAPH: Generate graph from seed {seed}")
    random.seed(seed)
    pr.trace_log(pr.TraceLogLevel.LOG_INFO, f"GRAPH: Generate vertices")
    vertice = generate_vertice(20, 100)
    pr.trace_log(pr.TraceLogLevel.LOG_INFO, f"GRAPH: Generate edges")
    edges = generate_edges(vertice, 25, 3)
    return SpatialGraph(vertice, edges)
