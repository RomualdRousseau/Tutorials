import math
import random
import numpy as np
import pyray as pr

from tutorial1.math.geom import Point, distance
import tutorial1.util.resources as res
import tutorial1.entities.world as world

MASS = 650  # kg
LENGTH = 5  # m
WIDTH = 2  # m
WHEEL_ANGLE_RATE = math.pi / 50  # rad
MAX_ENGINE_POWER = 200  # kN
DRAG_ROAD = 0.9  # Concrete/Rubber
DRAG_ROLLING = 0.01  # Concrete/Rubber
C_G = 9.81  # m.s-2
START_OFFSET = 2  # m


class Car:
    def __init__(self, color: pr.Color) -> None:
        self.color = color

        start_seg = random.choice(world._world.borders.skeleton)
        start_pos, end_pos = start_seg.start.xy, start_seg.end.xy
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

    def get_travel_distance_in_km(self) -> float:
        return (self.total_distance + distance(self.current_start, self.current_pos)) * 0.001

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

    def update(self, dt: float) -> None:
        self._think()
        self._update_physic(dt)
        self._update_sensors()  # very slow

        match world.get_location(Point(self.pos)):
            case r if r is not None:
                if self.current_seg != r[1]:
                    self.total_distance +=  r[1].length
                    self.current_seg = r[1]
                    self.current_start = (
                        self.current_seg.start
                        if distance(Point(self.pos), self.current_seg.start)
                        < distance(Point(self.pos), self.current_seg.end)
                        else self.current_seg.end
                    )
                self.current_pos = r[0]

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
            np.rad2deg(np.arctan2(self.head[1], self.head[0]) + math.pi / 2),
            self.color,
        )

    def _think(self) -> None:
        self.wheel = 0
        self.throttle = 0
        if pr.is_key_down(pr.KeyboardKey.KEY_RIGHT):
            self.turn_wheel(0.25)
        if pr.is_key_down(pr.KeyboardKey.KEY_LEFT):
            self.turn_wheel(-0.25)
        if pr.is_key_down(pr.KeyboardKey.KEY_UP):
            self.push_throttle(0.5)

    def _update_physic(self, dt: float) -> None:
        # Second Newton law

        forces = np.array([0.0, 0.0])

        if self.wheel != 0:
            circ_radius = LENGTH / (math.sin(self.wheel))
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
