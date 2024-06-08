from dataclasses import dataclass

import math
import random
from typing import Iterable, Optional
import numpy as np
import pyray as pr

from tutorial1.util.funcs import curry
from tutorial1.util.geom import (
    Point,
    Segment,
    angle,
    cast_ray_segments,
    collision_circle_segment,
    distance,
    distance_point_segment,
    intersect,
    nearest_point_segment,
)

import tutorial1.util.envelope as envelope
import tutorial1.util.graph as graph
import tutorial1.util.resources as res

ROAD_WIDTH = 10
GRASS_COLOR = pr.Color(0, 192, 0, 255)
ROAD_COLOR = pr.Color(192, 192, 192, 255)
BORDER1_COLOR = pr.Color(255, 255, 255, 255)
BORDER2_COLOR = pr.Color(255, 0, 0, 255)

HOUSE_DENSITY = 0.9
TREE_DENSITY = 0.5

HOUSE_SIZES = {"house1": (10, 10), "house2": (16, 16), "house3": (16, 16)}


@dataclass
class House:
    position: Point
    segment: Segment
    type: str

    def __post_init__(self):
        self.angle = angle(self.segment)
        path_end = nearest_point_segment(self.position, self.segment)
        if path_end:
            self.path = Segment(self.position, path_end)


@dataclass
class Tree:
    position: Point
    angle: float


@dataclass
class World:
    borders: envelope.Envelope
    houses: list[House]
    trees: list[Tree]


def init() -> World:
    borders, anchors = envelope.generare_from_spatial_graph(
        graph.generate_random(5), ROAD_WIDTH, 20
    )
    houses = [
        House(
            a,
            sorted(borders.segments, key=curry(distance_point_segment)(a))[0],
            random.choice(["house1", "house2", "house3"]),
        )
        for a in anchors
        if 10 <= min(map(curry(distance_point_segment)(a), borders.segments)) <= 20
        and random.random() < HOUSE_DENSITY
    ]
    trees = [
        Tree(
            Point(a.x + random.randint(-10, 10), a.y + random.randint(-10, 10)),
            random.random() * math.pi * 2,
        )
        for a in anchors
        if min(map(curry(distance_point_segment)(a), borders.segments)) > 20
        and random.random() < TREE_DENSITY
    ]

    return World(borders, houses, trees)


def is_alive() -> bool:
    return True


def update(dt: float) -> None:
    pass


def draw(layer: int) -> None:
    def draw_bg():
        pr.clear_background(GRASS_COLOR)
        for s in _world.borders.skeleton:
            s.draw(_world.borders.width + 2, ROAD_COLOR, None, True)
        for house in _world.houses:
            house.path.draw(4, ROAD_COLOR)
        for s in _world.borders.segments:
            s.draw(1, BORDER1_COLOR, None, True)
        for s in _world.borders.skeleton:
            s.draw(0.5, BORDER1_COLOR, (3, ROAD_COLOR), False)

    def draw_fg():
        for tree in _world.trees:
            tex = res.load_texture("tree")
            pr.draw_texture_pro(
                tex,
                pr.Rectangle(0, 0, tex.width, tex.height),
                pr.Rectangle(tree.position.x, tree.position.y, 8, 8),
                pr.Vector2(4, 4),
                np.rad2deg(tree.angle),
                pr.WHITE,  # type: ignore
            )
        for house in _world.houses:
            tex = res.load_texture(house.type)
            sx, sy = HOUSE_SIZES[house.type]
            pr.draw_texture_pro(
                tex,
                pr.Rectangle(0, 0, tex.width, tex.height),
                pr.Rectangle(house.position.x, house.position.y, sx, sy),
                pr.Vector2(sx * 0.5, sy * 0.5),
                np.rad2deg(house.angle),
                pr.WHITE,  # type: ignore
            )

    [draw_bg, draw_fg][layer]()


def get_location(position: Point) -> Optional[tuple[Point, Segment]]:
    nearest = lambda x: (nearest_point_segment(position, x), x)
    closest = lambda x: distance(position, x[0]) if x[0] is not None else np.Infinity
    location = min(map(nearest, _world.borders.skeleton), key=closest)
    return location if location[0] is not None else None  # type: ignore


def get_nearest_segments(position: Point, length: float = 50) -> Iterable[Segment]:
    nearest = lambda x: distance_point_segment(position, x) < length
    return filter(nearest, _world.borders.segments)


def cast_ray(
    position: Point,
    direction: np.ndarray,
    length: float = 50,
) -> Segment:
    nearest_segments = get_nearest_segments(position, length)
    return cast_ray_segments(position, direction, length, nearest_segments)


def cast_rays(
    position: Point,
    direction: np.ndarray,
    length: float = 50,
    sampling: int = 10,
):
    rays = []
    alpha = np.arctan2(direction[1], direction[0])
    nearest_segments = list(get_nearest_segments(position, length))
    for i in range(sampling):
        beta = np.interp(
            i / sampling, [0, 1], [alpha - math.pi / 4, alpha + math.pi / 4]
        )
        direction = np.array([math.cos(beta), math.sin(beta)])
        rays.append(cast_ray_segments(position, direction, length, nearest_segments))
    return rays


def collision(position: Point, radius: float) -> Optional[np.ndarray]:
    nearest_segments = list(get_nearest_segments(position, radius))
    collide = curry(collision_circle_segment)(position, radius)
    cols = np.array([x for x in map(collide, nearest_segments) if x is not None])
    return np.average(cols, axis=0) if len(cols) > 0 else None


_world = init()
