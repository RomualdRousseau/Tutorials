from __future__ import annotations

from typing import Protocol
import numpy as np
import pyray as pr

import tutorial1.resources as res
from tutorial1.math.envelope import Location
from tutorial1.math.geom import Point, point_in_polygon
from tutorial1.math.linalg import normalize

COLOR = pr.Color(255, 255, 255, 128)
RADIUS = 6
MAX_LIFE = 30


class MarkerListener(Protocol):
    def get_previous_pos(self) -> Point: ...

    def get_current_pos(self) -> Point: ...

    def on_enter(self, marker: Marker) -> None: ...

    def on_leave(self, marker: Marker) -> None: ...


class Marker:
    def __init__(self, location: Location, direct: bool = True, width: float = 0.5, height: float = 0.5) -> None:
        self.location = location
        self.width = width
        self.height = height

        seg_dir = normalize(self.location[0].end.xy - self.location[0].start.xy)
        self.front = seg_dir if direct else -seg_dir
        self.right = np.array([-self.front[1], self.front[0]])

        self.polygon = self._get_polygon()
        self.listeners: list[MarkerListener] = []

    def add_listener(self, listener: MarkerListener) -> None:
        self.listeners.append(listener)

    def is_alive(self) -> bool:
        return True

    def reset(self) -> None:
        pass

    def hit(self, damage: int) -> None:
        pass

    def update(self, dt: float) -> None:
        for listener in self.listeners:
            prev = listener.get_previous_pos()
            curr = listener.get_current_pos()
            prev_in = point_in_polygon(prev, self.polygon, False)
            curr_in = point_in_polygon(curr, self.polygon, False)
            if not prev_in and curr_in:
                listener.on_enter(self)
            if prev_in and not curr_in:
                listener.on_leave(self)

    def draw(self, layer: int) -> None:
        if layer != 0:
            return

        pos = self.location[1].xy + self.right * self.width * 0.5
        tex = res.load_texture("start")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(pos[0], pos[1], self.width, self.height),
            pr.Vector2(self.width * 0.5, self.height * 0.5),
            np.rad2deg(np.arctan2(self.right[1], self.right[0])),
            pr.WHITE,  # type: ignore
        )

    def _get_polygon(self) -> list[Point]:
        p = self.location[1].xy
        u = self.right * self.width
        v = self.front * self.height * 0.5
        return [Point(p + v + u), Point(p - v + u), Point(p - v), Point(p + v)]
