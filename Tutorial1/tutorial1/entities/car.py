from typing import Optional

import numpy as np
import pyray as pr

import tutorial1.resources as res
from tutorial1.constants import GAMEPAD_AXIS_X, GAMEPAD_AXIS_Y, GAMEPAD_ID
from tutorial1.entities import world
from tutorial1.math import envelope
from tutorial1.math.geom import (
    Point,
    Segment,
    cast_ray_segments,
    collision_circle_segment,
    distance,
    nearest_point_segment,
)
from tutorial1.math.linalg import EPS, lst_2_vec, norm, normalize
from tutorial1.util.types import bit_set, bit_set_if, bit_unset, is_bit_set

MAX_LIFE = 100
MASS = 650  # kg
LENGTH = 5  # m
WIDTH = 2  # m
WHEEL_ANGLE_RATE = np.pi / 100  # rad
MAX_ENGINE_POWER = 200  # kN
MAX_SPEED = 125  # km.h-1
DRAG_ROAD = 0.9  # Concrete/Rubber
DRAG_ROLLING = 0.01  # Concrete/Rubber
C_G = 9.81  # m.s-2

FLAG_DAMAGED = 0
FLAG_OUT_OF_TRACK = 1

RAY_MAX_LEN = 25  # m
RAY_FOV = np.pi * 0.3

START_OFFSET = world.ROAD_WIDTH / 4  # m
MAX_VISITED_LOCATION = 10


class Car:
    def __init__(self, color: pr.Color, input_mode: str = "human") -> None:
        assert input_mode in ("human", "ai")
        self.color = color
        self.input_mode = input_mode
        self.debug_mode = False
        self.spawn_mode = False
        self.corridor: Optional[envelope.Envelope] = None

    def set_debug_mode(self, debug_mode: bool) -> None:
        self.debug_mode = debug_mode

    def set_corridor(self, corridor: envelope.Envelope) -> None:
        self.corridor = corridor

    def set_spawn_location(self, spawn_location: Optional[envelope.Location]) -> None:
        self.spawn_mode = spawn_location is not None
        if spawn_location is not None:
            self.spawn_seg, self.spawn_pos = spawn_location

    def get_total_distance_in_km(self) -> float:
        _, last_start = self.visited_location[-1]
        return (self.total_distance + distance(last_start, self.current_location_pos)) * 0.001

    def get_average_speed_in_kmh(self) -> float:
        return (self.total_velocity / (self.total_tick + EPS)) * 3.6

    def get_speed_in_kmh(self) -> float:
        return norm(self.vel) * 3.6

    def get_spawn_location(self) -> envelope.Location:
        return self.visited_location[-1 if len(self.visited_location) <= 1 else -2]

    def turn_wheel(self, torque: float) -> None:
        self.wheel = float(np.interp(torque, [-1, 1], [-WHEEL_ANGLE_RATE, WHEEL_ANGLE_RATE]))

    def push_throttle(self, power: float) -> None:
        self.throttle = MAX_ENGINE_POWER * 1000 * power

    def is_alive(self) -> bool:
        return self.life > 0

    def reset(self) -> None:
        assert self.corridor is not None

        if self.spawn_mode:
            start_seg = self.spawn_seg
            start_pos, end_pos = (
                self.spawn_seg.closest_ep(self.spawn_pos).xy,
                self.spawn_seg.farest_ep(self.spawn_pos).xy,
            )
        else:
            start_seg = self.corridor.skeleton[0]
            start_pos, end_pos = start_seg.start.xy, start_seg.end.xy
        start_pos = start_pos * 0.99 + end_pos * 0.01
        start_dir = normalize(end_pos - start_pos)
        start_off = lst_2_vec([-start_dir[1], start_dir[0]]) * START_OFFSET

        self.pos = start_pos + start_off
        self.vel = np.zeros(2, dtype=np.float64)
        self.head = start_dir

        self.life = MAX_LIFE
        self.wheel = 0.0
        self.throttle = 0.0
        self.flags = 0

        self.current_location_pos = Point(start_pos)
        self.visited_location = [(start_seg, self.current_location_pos)]

        self.camera: list[Segment] = self._cast_rays()
        self.proximity: Optional[Segment] = None

        self.total_distance = 0.0
        self.total_velocity = 0.0
        self.total_tick = 0

    def hit(self, damage: int) -> None:
        self.life = max(0, self.life - damage)

    def update(self, dt: float) -> None:
        if self.input_mode == "human":
            self._input_human()
        self._update_physic(dt)

    def draw(self, layer: int) -> None:
        if layer != 1:
            return

        if self.debug_mode:
            color: pr.Color = pr.YELLOW if not is_bit_set(self.flags, FLAG_DAMAGED | FLAG_OUT_OF_TRACK) else pr.RED  # type: ignore
            color = pr.color_alpha(color, 0.25)

            pr.draw_line_v(self.visited_location[-1][1].to_vec(), self.current_location_pos.to_vec(), color)

            for ray in self.camera:
                pr.draw_line_v(ray.start.to_vec(), ray.end.to_vec(), color)

            if self.proximity is not None:
                pr.draw_line_v(self.proximity.start.to_vec(), self.proximity.end.to_vec(), color)

        tex = res.load_texture("car")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(self.pos[0], self.pos[1], LENGTH, LENGTH),
            pr.Vector2(LENGTH * 0.5, LENGTH * 0.5),
            np.rad2deg(np.arctan2(self.head[1], self.head[0]) + np.pi / 2),
            pr.color_alpha(self.color, 1.0) if self.debug_mode else self.color,
        )

    def _input_human(self) -> None:
        # Gamepad

        self.turn_wheel(pr.get_gamepad_axis_movement(GAMEPAD_ID, GAMEPAD_AXIS_X) ** 3)
        self.push_throttle(-1.0 * pr.get_gamepad_axis_movement(GAMEPAD_ID, GAMEPAD_AXIS_Y) ** 3)

        # Keyboard

        if pr.is_key_down(pr.KeyboardKey.KEY_RIGHT):
            self.turn_wheel(0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_LEFT):
            self.turn_wheel(-0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_UP):
            self.push_throttle(0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_DOWN):
            self.push_throttle(-0.25)

    def _update_physic(self, dt: float) -> None:
        assert self.corridor is not None

        # Simple car modelisation (traction, drag road, drag rolling)

        forces = np.zeros(2)

        if self.wheel != 0:
            circ_radius = LENGTH / (np.sin(self.wheel))
            ang_vel = np.linalg.norm(self.vel) / circ_radius
            c, s = np.cos(ang_vel), np.sin(ang_vel)
            self.head = [[c, -s], [s, c]] @ self.head
        tract = self.head * self.throttle
        forces += tract

        drag_rd = self.vel * -DRAG_ROAD * MASS * C_G
        forces += drag_rd

        drag_rr = self.vel * -DRAG_ROLLING * MASS * C_G
        forces += drag_rr

        # Second Newton law

        acc = forces / MASS

        # Simple Euler integration

        self.vel += acc * dt
        self.pos += self.vel * dt

        # Collisions

        pos = Point(self.pos)

        match self._collision():
            case None:
                self.flags = bit_unset(self.flags, FLAG_DAMAGED)
            case reaction:
                self.vel = self.vel * 0.5 + reaction
                self.pos += reaction
                self.head = normalize(self.vel)
                self.flags = bit_set(self.flags, FLAG_DAMAGED)

        # Sensors

        self.camera = self._cast_rays()

        match nearest_point_segment(pos, self.visited_location[-1][0], True):
            case None:
                self.proximity = None
                self.flags = bit_unset(self.flags, FLAG_OUT_OF_TRACK)
            case Point() as nearest:
                self.proximity = Segment(pos, nearest)
                self.flags = bit_set_if(self.flags, FLAG_OUT_OF_TRACK, self.proximity.length < WIDTH * 0.75)

        # Localisation

        is_new_location_added = False
        seg, self.current_location_pos = envelope.get_nearest_location(self.corridor, pos)
        if self.visited_location[-1][0] != seg:
            self.visited_location.append((seg, seg.closest_ep(self.current_location_pos)))
            if len(self.visited_location) > MAX_VISITED_LOCATION:
                self.visited_location.pop(0)
            is_new_location_added = True

        # Statistics

        self.total_distance += self.visited_location[-2][0].length if is_new_location_added else 0
        self.total_velocity += norm(self.vel)
        self.total_tick += 1

    def _cast_rays(
        self,
        length: float = RAY_MAX_LEN,
        fov: float = RAY_FOV,
        sampling: int = 16,
    ) -> list[Segment]:
        assert self.corridor is not None

        position = Point(self.pos)
        nearest_segments = envelope.get_nearest_segments(self.corridor, position, length)
        alpha = np.arctan2(self.head[1], self.head[0])
        rays = []

        for i in range(sampling):
            beta = np.interp(i / sampling, [0, 1], [alpha - fov, alpha + fov])
            direction = lst_2_vec([np.cos(beta), np.sin(beta)])
            rays.append(cast_ray_segments(position, direction, length, nearest_segments))
        return rays

    def _collision(self, radius: float = WIDTH * 0.5) -> Optional[np.ndarray]:
        assert self.corridor is not None

        position = Point(self.pos)
        nearest_segments = envelope.get_nearest_segments(self.corridor, position, radius)
        collide = lambda x: collision_circle_segment(position, radius, x)

        return next((x for x in map(collide, nearest_segments) if x is not None), None)
