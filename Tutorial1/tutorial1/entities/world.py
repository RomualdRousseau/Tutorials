import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pyray as pr

import tutorial1.resources as res
from tutorial1.math import envelope, graph
from tutorial1.math.geom import (
    Point,
    Segment,
    cast_ray_segments,
    collision_circle_segment,
    distance,
    distance_point_segment,
    nearest_point_segment,
)
from tutorial1.math.linalg import lst_2_vec

GRASS_COLOR = pr.Color(86, 180, 57, 255)
ROAD_COLOR = pr.Color(192, 192, 192, 255)
BORDER1_COLOR = pr.Color(255, 255, 255, 255)
BORDER2_COLOR = pr.Color(255, 0, 0, 255)
BORDER3_COLOR = pr.Color(255, 255, 0, 255)

ROAD_WIDTH = 10  # m
START_OFFSET = 2  # m

HOUSE_DENSITY = 0.9
HOUSE_DISTANCE = 10  # m
HOUSE_SIZES = {"house1": (10, 10), "house2": (16, 16), "house3": (16, 16)}

TREE_DENSITY = 0.5
TREE_DISTANCE = 25  # m
TREE_OFFSET = 5  # m

Location = tuple[Segment, Point]


@dataclass
class House:
    position: Point
    segment: Segment
    type: str

    def __post_init__(self):
        self.angle = self.segment.angle
        path_end = nearest_point_segment(self.position, self.segment)
        if path_end:
            self.path = Segment(self.position, path_end)


@dataclass
class Tree:
    position: Point
    angle: float


@dataclass
class World:
    roads: graph.SpatialGraph
    borders: envelope.Envelope
    houses: list[House]
    trees: list[Tree]
    corridor: envelope.Envelope


@lru_cache(1)
def get_singleton(name: str = "default"):
    pr.trace_log(pr.TraceLogLevel.LOG_INFO, "WORLD: Initialize singleton")

    roads = graph.generate_random()

    borders, anchors = envelope.generare_from_spatial_graph(roads, ROAD_WIDTH)

    houses = [
        House(
            a,
            min(borders.segments, key=lambda x: distance_point_segment(a, x)),
            random.choice(["house1", "house2", "house3"]),
        )
        for a in anchors
        if HOUSE_DISTANCE <= min((distance_point_segment(a, x) for x in borders.segments)) <= TREE_DISTANCE
        and random.random() < HOUSE_DENSITY
    ]

    trees = [
        Tree(
            Point(
                a.xy
                + [
                    random.randint(-TREE_OFFSET, TREE_OFFSET),
                    random.randint(-TREE_OFFSET, TREE_OFFSET),
                ]
            ),
            random.random() * np.pi / 2,
        )
        for a in anchors
        if min((distance_point_segment(a, x) for x in borders.segments)) > TREE_DISTANCE
        and random.random() < TREE_DENSITY
    ]

    start = random.choice(roads.vertice)
    stop = max(roads.vertice, key=lambda x: distance(start.point, x.point))
    corridor, _ = envelope.generare_from_spatial_graph(roads.get_shortest_path(start, stop), ROAD_WIDTH)

    return World(roads, borders, houses, trees, corridor)


def get_corridor() -> envelope.Envelope:
    _world = get_singleton()
    return _world.corridor


def is_alive() -> bool:
    return True


def reset() -> None:
    get_singleton()


def update(dt: float) -> None:
    pass


def draw(layer: int) -> None:
    _world = get_singleton()

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
        for s in _world.corridor.segments:
            s.draw(1, BORDER3_COLOR, None, True)

    def draw_fg():
        for tree in _world.trees:
            tex = res.load_texture("tree")
            pr.draw_texture_pro(
                tex,
                pr.Rectangle(0, 0, tex.width, tex.height),
                pr.Rectangle(tree.position.xy[0], tree.position.xy[1], 8, 8),
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
                pr.Rectangle(house.position.xy[0], house.position.xy[1], sx, sy),
                pr.Vector2(sx * 0.5, sy * 0.5),
                np.rad2deg(house.angle),
                pr.WHITE,  # type: ignore
            )

    [draw_bg, draw_fg][layer]()


@lru_cache(256)
def get_nearest_location(position: Point) -> Location:
    _world = get_singleton()
    nearest_location = lambda x: (x, nearest_point_segment(position, x, True))
    closest_distance = lambda x: distance(position, x[1])
    location = min(map(nearest_location, _world.corridor.skeleton), key=closest_distance)
    return location  # type: ignore


@lru_cache(256)
def get_nearest_corridors(position: Point, radius: float, closest: bool = True) -> list[Segment]:
    _world = get_singleton()
    nearest_distance = lambda x: distance_point_segment(position, x, closest) < radius
    return [x for x in _world.corridor.segments if nearest_distance(x)]


def cast_rays(
    position: Point,
    direction: np.ndarray,
    length: float,
    fov: float,
    sampling: int = 16,
) -> list[Segment]:
    nearest_segments = get_nearest_corridors(position, length)
    rays = []
    alpha = np.arctan2(direction[1], direction[0])
    for i in range(sampling):
        beta = np.interp(i / sampling, [0, 1], [alpha - fov, alpha + fov])
        direction = lst_2_vec([np.cos(beta), np.sin(beta)])
        rays.append(cast_ray_segments(position, direction, length, nearest_segments))
    return rays


def collision(position: Point, radius: float) -> Optional[np.ndarray]:
    nearest_segments = get_nearest_corridors(position, radius)
    collide = lambda x: collision_circle_segment(position, radius, x)
    cols = [x for x in map(collide, nearest_segments) if x is not None]
    return np.average(lst_2_vec(cols), axis=0) if len(cols) > 0 else None
