from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

import random
import numpy as np
import pyray as pr

from tutorial1.constants import GAME_SEED
from tutorial1.util.funcs import curry
from tutorial1.math.geom import (
    Point,
    Segment,
    angle,
    cast_ray_segments,
    collision_circle_segment,
    distance,
    distance_point_segment,
    nearest_point_segment,
)

import tutorial1.math.envelope as envelope
import tutorial1.math.graph as graph
import tutorial1.util.resources as res

GRASS_COLOR = pr.Color(0, 192, 0, 255)
ROAD_COLOR = pr.Color(192, 192, 192, 255)
BORDER1_COLOR = pr.Color(255, 255, 255, 255)
BORDER2_COLOR = pr.Color(255, 0, 0, 255)
BORDER3_COLOR = pr.Color(255, 255, 0, 255)

ROAD_WIDTH = 10  # m
START_OFFSET = 2  # m

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
    roads: graph.SpatialGraph
    borders: envelope.Envelope
    houses: list[House]
    trees: list[Tree]
    corridor: envelope.Envelope


def _init():
    roads = graph.generate_random(GAME_SEED)

    borders, anchors = envelope.generare_from_spatial_graph(roads, ROAD_WIDTH)
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
            Point(a.xy + [random.randint(-5, 5), random.randint(-5, 5)]),
            random.random() * np.pi,
        )
        for a in anchors
        if min(map(curry(distance_point_segment)(a), borders.segments)) > 20
        and random.random() < TREE_DENSITY
    ]
    
    corridor = envelope.Envelope([], [], ROAD_WIDTH)

    return World(roads, borders, houses, trees, corridor)


def is_alive() -> bool:
    return True


def reset() -> None:
    pr.trace_log(pr.TraceLogLevel.LOG_DEBUG, "WORLD: reset")
    start = random.choice(_world.roads.vertice)
    stop = max(_world.roads.vertice, key=lambda x: distance(start.point, x.point))
    corridor, _ = envelope.generare_from_spatial_graph(
        _world.roads.get_shortest_path(start, stop), ROAD_WIDTH
    )
    _world.corridor = corridor


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
        for s in _world.corridor.segments:
            s.draw(1, BORDER3_COLOR, None, True)
        # for s in _world.corridor.skeleton:
        #     s.draw(_world.corridor.width, pr.Color(255, 255, 0, 128), None, True)
        # _world.corridor.skeleton[0].start.draw(1, pr.GREEN)  # type: ignore
        # _world.corridor.skeleton[-1].end.draw(1, pr.RED)  # type: ignore

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


def get_location(position: Point) -> Optional[tuple[Point, Segment]]:
    nearest = lambda x: (nearest_point_segment(position, x), x)
    closest = lambda x: distance(position, x[0]) if x[0] is not None else np.Infinity
    location = min(map(nearest, _world.corridor.skeleton), key=closest)
    return location if location[0] is not None else None  # type: ignore


@lru_cache
def get_nearest_segments(position: Point, length: float = 50) -> list[Segment]:
    nearest = (
        lambda x: distance_point_segment(position, x) < length
        or distance(position, x.start) < length
        or distance(position, x.end) < length
    )
    return list(filter(nearest, _world.corridor.segments))


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
    nearest_segments = get_nearest_segments(position, length)
    for i in range(sampling):
        beta = np.interp(i / sampling, [0, 1], [alpha - np.pi / 4, alpha + np.pi / 4])
        direction = np.array([np.cos(beta), np.sin(beta)])
        rays.append(cast_ray_segments(position, direction, length, nearest_segments))
    return rays


def collision(position: Point, radius: float) -> Optional[np.ndarray]:
    nearest_segments = get_nearest_segments(position, radius)
    collide = curry(collision_circle_segment)(position, radius)
    cols = np.array([x for x in map(collide, nearest_segments) if x is not None])
    return np.average(cols, axis=0) if len(cols) > 0 else None


_world = _init()
