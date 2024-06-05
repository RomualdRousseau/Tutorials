import math
import random

import numpy as np
import pyray as pr

import tutorial1.util.envelope as envelope
import tutorial1.util.graph as graph
import tutorial1.util.resources as res
from tutorial1.util.geom import (
    Point,
    Segment,
    angle,
    distance,
    distance_point_segment,
    intersect,
)

GRASS_COLOR = pr.Color(0, 192, 0, 255)
ROAD_COLOR = pr.Color(192, 192, 192, 255)
BORDER1_COLOR = pr.Color(255, 255, 255, 255)
BORDER2_COLOR = pr.Color(255, 0, 0, 255)


borders, anchors = envelope.generare_from_spatial_graph(
    graph.generate_random(5), 10, 20
)

trees = [
    (a, random.random() * math.pi * 2)
    for a in anchors
    if min(map(lambda s: distance_point_segment(a, s), borders.segments)) > 20
    and random.random() < 0.5
]

houses = [
    (
        a,
        angle(sorted(borders.segments, key=lambda s: distance_point_segment(a, s))[0]),
        random.choice(["house1", "house2", "house3"]),
    )
    for a in anchors
    if 10 <= min(map(lambda s: distance_point_segment(a, s), borders.segments)) <= 20
    and random.random() < 0.9
]


def is_alive() -> bool:
    return True


def update(dt: float) -> None:
    pass


def draw(layer: int) -> None:
    if layer == 0:
        pr.clear_background(GRASS_COLOR)
        for s in borders.skeleton:
            s.draw(borders.width + 4, ROAD_COLOR, None, True)
        for s in borders.skeleton:
            s.draw(1, BORDER1_COLOR, (3, ROAD_COLOR), False)
        for s in borders.segments:
            s.draw(2, BORDER1_COLOR, (5, BORDER2_COLOR), True)

    if layer == 2:
        for t, a in trees:
            tex = res.load_texture("tree")
            pr.draw_texture_pro(
                tex,
                pr.Rectangle(0, 0, tex.width, tex.height),
                pr.Rectangle(t.x, t.y, 8, 8),
                pr.Vector2(4, 4),
                a * 180 / math.pi,
                pr.WHITE,  # type: ignore
            )
        for h, a, s in houses:
            tex = res.load_texture(s)
            pr.draw_texture_pro(
                tex,
                pr.Rectangle(0, 0, tex.width, tex.height),
                pr.Rectangle(h.x, h.y, 16, 16),
                pr.Vector2(8, 8),
                a * 180 / math.pi,
                pr.WHITE,  # type: ignore
            )


def cast_rays(
    position: Point,
    direction: pr.Vector2,
    length: float = 100,
    sampling: int = 10,
):
    alpha = math.atan2(direction.y, direction.x)
    for i in range(sampling):
        beta = np.interp(
            i / sampling, [0, 1], [alpha - math.pi / 4, alpha + math.pi / 4]
        )
        point = Point(
            length * math.cos(beta) + position.x,
            length * math.sin(beta) + position.y,
        )
        point = _cast_ray(Segment(position, point))
        yield Segment(position, point)


def _cast_ray(ray: Segment) -> Point:
    d = distance(ray.start, ray.end)
    point = ray.end
    for segment in borders.segments:
        match intersect(ray, segment):
            case p if p:
                match distance(ray.start, p):
                    case dd if dd < d:
                        point = p
                        d = dd
    return point
