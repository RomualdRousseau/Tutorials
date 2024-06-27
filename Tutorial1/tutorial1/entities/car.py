from typing import Optional

import numpy as np
import pyray as pr

import tutorial1.resources as res
from tutorial1.constants import GAMEPAD_AXIS_X, GAMEPAD_AXIS_Y, GAMEPAD_ID
from tutorial1.entities import world
from tutorial1.math.geom import Point, Segment, distance
from tutorial1.math.linalg import EPS, almost, normalize

MASS = 650  # kg
LENGTH = 5  # m
WIDTH = 2  # m
WHEEL_ANGLE_RATE = np.pi / 100  # rad
MAX_ENGINE_POWER = 200  # kN
MAX_SPEED = 125  # km.h-1
DRAG_ROAD = 0.9  # Concrete/Rubber
DRAG_ROLLING = 0.01  # Concrete/Rubber
C_G = 9.81  # m.s-2
START_OFFSET = world.ROAD_WIDTH / 4  # m

SpawnLocation = tuple[Segment, Point]


class Car:
    def __init__(self, color: pr.Color, input_mode: str = "human") -> None:
        self.color = color
        assert input_mode in ("human", "ai")
        self.input_mode = input_mode
        self.debug_mode = input_mode == "human"
        self.spawn_mode = False

    def get_travel_distance_in_km(self) -> float:
        return (self.total_distance + distance(self.current_start, self.current_pos)) * 0.001

    def get_speed_in_kmh(self) -> float:
        return float(np.linalg.norm(self.vel)) * 3.6

    def get_avg_velocity(self) -> float:
        return self.total_velocity / (self.total_time + EPS)

    def set_debug_mode(self, debug_mode: bool) -> None:
        self.debug_mode = debug_mode

    def get_spawn_location(self) -> SpawnLocation:
        spawn_seg = self.current_seg
        spawn_pos = self.current_start
        return spawn_seg, spawn_pos

    def set_spawn_location(self, spawn_location: Optional[SpawnLocation]) -> None:
        self.spawn_mode = spawn_location is not None
        if spawn_location is not None:
            self.spawn_seg, self.spawn_pos = spawn_location

    def turn_wheel(self, torque: float) -> None:
        self.wheel = float(np.interp(torque, [-1, 1], [-WHEEL_ANGLE_RATE, WHEEL_ANGLE_RATE]))

    def push_throttle(self, power: float) -> None:
        self.throttle = MAX_ENGINE_POWER * 1000 * power

    def is_alive(self) -> bool:
        return True

    def reset(self) -> None:
        if self.spawn_mode:
            start_seg = self.spawn_seg
            start_pos, end_pos = (
                self.spawn_seg.closest_ep(self.spawn_pos).xy,
                self.spawn_seg.farest_ep(self.spawn_pos).xy,
            )
        else:
            start_seg = world.get_corridor().skeleton[0]
            start_pos, end_pos = start_seg.start.xy, start_seg.end.xy
        assert not almost(start_pos, end_pos)
        start_pos = start_pos * 0.99 + end_pos * 0.01
        start_dir = normalize(end_pos - start_pos)
        start_off = np.array([-start_dir[1], start_dir[0]]) * START_OFFSET

        self.pos = start_pos + start_off
        self.vel = np.array([0.0, 0.0])
        self.head = start_dir

        self.wheel = 0.0
        self.throttle = 0.0
        self.damaged = False
        self.out_of_track = False

        self.total_distance = 0.0
        self.total_velocity = 0.0
        self.total_time = 0.0

        self.current_seg = start_seg
        self.current_start = Point(start_pos)
        self.current_pos = self.current_start

        self._update_sensors()

    def update(self, dt: float) -> None:
        prev_damaged = self.damaged

        if self.input_mode == "human":
            self._input_human()

        self._update_physic(dt)
        self._update_sensors()

        if self.input_mode == "human":
            if self.damaged and not prev_damaged and not pr.is_sound_playing(res.load_sound("crash")):
                pr.play_sound(res.load_sound("crash"))

            if self.out_of_track and not pr.is_sound_playing(res.load_sound("klaxon")):
                pr.play_sound(res.load_sound("klaxon"))

    def draw(self, layer: int) -> None:
        if layer != 0:
            return

        if self.debug_mode:
            color = pr.YELLOW if not self.out_of_track else pr.RED
            for ray in self.camera:
                pr.draw_line_v(ray.start.to_vec(), ray.end.to_vec(), color)  # type: ignore
            pr.draw_line_v(self.proximity.start.to_vec(), self.proximity.end.to_vec(), color)  # type: ignore
            pr.draw_line_v(self.current_start.to_vec(), self.current_pos.to_vec(), color)  # type: ignore
            pr.draw_circle_v(self.current_pos.to_vec(), 1, color)  # type: ignore

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

    def _update_physic(self, dt: float) -> None:
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

        match world.collision(Point(self.pos), WIDTH * 0.5):
            case v if v is not None:
                self.vel = self.vel * 0.5 + v
                self.pos += v
                self.head = normalize(self.vel)
                self.damaged = True
            case _:
                self.damaged = False

        # Localisation

        match world.get_location(Point(self.pos)):
            case loc if loc is not None:
                pos, seg = loc
                if self.current_seg != seg:
                    self.current_seg = seg
                    self.total_distance += self.current_seg.length
                    self.current_start = self.current_seg.closest_ep(pos)
                self.current_pos = pos

        self.total_time += 1
        self.total_velocity += float(np.linalg.norm(self.vel))

    def _update_sensors(self) -> None:
        pos = Point(self.pos)
        right = np.array([-self.head[1], self.head[0]])
        self.camera = world.cast_rays(pos, self.head)
        self.proximity = world.cast_ray(pos, right, world.ROAD_WIDTH)
        self.out_of_track = (world.ROAD_WIDTH / 2 - 1) < self.proximity.length < (world.ROAD_WIDTH - 0.1)
