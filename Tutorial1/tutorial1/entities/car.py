import numpy as np
import pyray as pr

import tutorial1.resources as res
from tutorial1.constants import GAMEPAD_AXIS_X, GAMEPAD_AXIS_Y, GAMEPAD_ID
from tutorial1.entities import world
from tutorial1.math.geom import Point, distance

MASS = 650  # kg
LENGTH = 5  # m
WIDTH = 2  # m
WHEEL_ANGLE_RATE = np.pi / 100  # rad
MAX_ENGINE_POWER = 200  # kN
DRAG_ROAD = 0.9  # Concrete/Rubber
DRAG_ROLLING = 0.01  # Concrete/Rubber
C_G = 9.81  # m.s-2
START_OFFSET = 2  # m


class Car:
    def __init__(self, color: pr.Color) -> None:
        self.color = color

    def get_travel_distance_in_km(self) -> float:
        return (
            self.total_distance + distance(self.current_start, self.current_pos)
        ) * 0.001

    def get_speed_in_kmh(self) -> float:
        return float(np.linalg.norm(self.vel) * 3.6)

    def turn_wheel(self, torque: float) -> None:
        self.wheel = float(
            np.interp(torque, [-1, 1], [-WHEEL_ANGLE_RATE, WHEEL_ANGLE_RATE])
        )

    def push_throttle(self, power: float) -> None:
        self.throttle = MAX_ENGINE_POWER * 1000 * power

    def is_alive(self) -> bool:
        return True

    def reset(self) -> None:
        pr.trace_log(pr.TraceLogLevel.LOG_DEBUG, "CAR: reset")

        start_seg = world._world.corridor.skeleton[0]
        start_pos, end_pos = (
            start_seg.start.xy + start_seg.end.xy
        ) * 0.5, start_seg.end.xy
        start_dir = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
        start_off = np.array([-start_dir[1], start_dir[0]]) * START_OFFSET

        self.pos = start_pos + start_off
        self.vel = np.array([0.0, 0.0])
        self.head = start_dir

        self.wheel = 0
        self.throttle = 0
        self.damaged = False

        self.total_distance = 0
        self.current_seg = start_seg
        self.current_start = (
            self.current_seg.start
            if distance(Point(self.pos), self.current_seg.start)
            < distance(Point(self.pos), self.current_seg.end)
            else self.current_seg.end
        )
        self.current_pos = self.current_start

        self._update_sensors()

    def update(self, dt: float) -> None:
        prev_damaged = self.damaged

        self._think_gp()
        self._update_physic(dt)
        self._update_sensors()

        match world.get_location(Point(self.pos)):
            case r if r is not None:
                if self.current_seg != r[1]:
                    self.total_distance += r[1].length
                    self.current_seg = r[1]
                    self.current_start = (
                        self.current_seg.start
                        if distance(Point(self.pos), self.current_seg.start)
                        < distance(Point(self.pos), self.current_seg.end)
                        else self.current_seg.end
                    )
                self.current_pos = r[0]

        if (
            self.damaged
            and not prev_damaged
            and not pr.is_sound_playing(res.load_sound("crash"))
        ):
            pr.play_sound(res.load_sound("crash"))

        if self.out_of_track and not pr.is_sound_playing(res.load_sound("klaxon")):
            pr.play_sound(res.load_sound("klaxon"))

    def draw(self, layer: int) -> None:
        if layer != 0:
            return

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
            self.color,
        )

    def _think_gp(self) -> None:
        self.turn_wheel(pr.get_gamepad_axis_movement(GAMEPAD_ID, GAMEPAD_AXIS_X) ** 3)
        self.push_throttle(
            -1.0 * pr.get_gamepad_axis_movement(GAMEPAD_ID, GAMEPAD_AXIS_Y) ** 3
        )

    def _think_kb(self) -> None:
        self.wheel = 0
        self.throttle = 0
        if pr.is_key_down(pr.KeyboardKey.KEY_RIGHT):
            self.turn_wheel(0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_LEFT):
            self.turn_wheel(-0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_UP):
            self.push_throttle(0.5)

    def _update_physic(self, dt: float) -> None:
        # Second Newton law

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

        acc = forces / MASS

        # Simple Euler integration

        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        # Collisions

        match world.collision(Point(self.pos), WIDTH * 0.5):
            case v if v is not None:
                self.vel = self.vel * 0.5 + v
                self.pos += v
                self.head = self.vel / np.linalg.norm(self.vel)
                self.damaged = True
            case _:
                self.damaged = False

    def _update_sensors(self) -> None:
        pos = Point(self.pos)
        right = np.array([-self.head[1], self.head[0]])
        self.camera = world.cast_rays(pos, self.head)
        self.proximity = world.cast_ray(pos, right, world.ROAD_WIDTH)
        self.out_of_track = self.proximity.length > (world.ROAD_WIDTH / 2 - 1)
