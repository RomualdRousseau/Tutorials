from dataclasses import dataclass
from functools import partial
from random import choice

from tutorial1.math import (
    Point,
    Segment,
    distance,
    intersect,
    randcoord,
)


class Vertex(Point):
    pass


class Edge(Segment):
    pass


@dataclass
class Graph:
    vertice: list[Vertex]
    edges: list[Edge]


def generate_random():
    vertice = generate_vertice(20, 100)
    edges = generate_edges(vertice, 25, 3)
    return Graph(vertice, edges)


def generate_vertice(num: int, min: int) -> list[Vertex]:
    vertice = []
    n = 0
    while n < num:
        v = Vertex(50 + randcoord(500), 50 + randcoord(500))
        is_valid = lambda x: distance(x, v) > min
        if all(map(is_valid, vertice)):
            vertice.append(v)
            n += 1
    return vertice


def generate_edges(vertice: list[Vertex], num: int, k: int = 3):
    edges = []
    n = 0
    while n < num:
        e = generate_edge(vertice, k)
        is_valid = lambda x: x != e and not intersect(x, e)
        if all(map(is_valid, edges)):
            edges.append(e)
            n += 1
    return edges


def generate_edge(vertice: list[Vertex], k: int) -> Edge:
    start = choice(vertice)
    points_minus_start = filter(lambda x: x != start, vertice)
    distance_to_start = partial(distance, start)
    choices = sorted(points_minus_start, key=distance_to_start)
    end = choice(choices[:k])
    return Edge(start, end)
