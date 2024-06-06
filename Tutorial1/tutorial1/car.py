from logging.config import valid_ident
import math
import random
import numpy as np
import pyray as pr

from tutorial1.util.geom import Point, collision_circle_segment
import tutorial1.util.resources as res
import tutorial1.world as world

C_MASS = 650  # kg
C_LENGTH = 5  # m
C_WIDTH = 2  # m
C_WHEEL_ANGLE_RATE = math.pi / 50  # rad
C_POWER = 200  # kN
C_DRAG_ROAD = 0.9  # Concrete/Rubber
C_DRAG_ROLLING = 0.01  # Concrete/Rubber
C_G = 9.81  # m.s-2
C_START_OFFSET = 2 # m


class Car:
    def __init__(self, color: pr.Color) -> None:
        self.vertice = [
            pr.Vector2(-C_LENGTH * 0.5, -C_WIDTH * 0.5),
            pr.Vector2(C_LENGTH * 0.5, 0),
            pr.Vector2(-C_LENGTH * 0.5, C_WIDTH * 0.5),
        ]
        self.color = color

        start_seg = random.choice(world._world.borders.skeleton)
        start_pos, end_pos = start_seg.start.to_np(), start_seg.end.to_np()
        start_dir = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
        start_off = np.array([-start_dir[1], start_dir[0]]) * C_START_OFFSET

        self.pos = start_pos + start_off
        self.vel = np.array([0.0, 0.0])
        self.head = start_dir

        self.wheel = 0
        self.throttle = 0

        self.rays = []
        self.damaged = False

    def get_speed_in_kmh(self) -> int:
        return int(np.linalg.norm(self.vel) * 3.6)

    def turn_wheel(self, t: float) -> None:
        self.wheel = float(
            np.interp(t, [-1, 1], [-C_WHEEL_ANGLE_RATE, C_WHEEL_ANGLE_RATE])
        )

    def push_throttle(self, t: float) -> None:
        self.throttle = C_POWER * 1000 * t

    def is_alive(self) -> bool:
        return True

    def update(self, dt: float) -> None:
        self.wheel = 0
        self.throttle = 0
        if pr.is_key_down(pr.KeyboardKey.KEY_RIGHT):
            self.turn_wheel(0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_LEFT):
            self.turn_wheel(-0.5)
        if pr.is_key_down(pr.KeyboardKey.KEY_UP):
            self.push_throttle(0.5)

        # Second Newton law

        forces = np.array([0.0, 0.0])

        if self.wheel != 0:
            circ_radius = C_LENGTH / (math.sin(self.wheel))
            ang_vel = np.linalg.norm(self.vel) / circ_radius
            c, s = np.cos(ang_vel), np.sin(ang_vel)
            self.head = [[c, -s], [s, c]] @ self.head
        tract = self.head * self.throttle
        forces += tract

        drag_rd = self.vel * -C_DRAG_ROAD * C_MASS * C_G
        forces += drag_rd

        drag_rr = self.vel * -C_DRAG_ROLLING * C_MASS * C_G
        forces += drag_rr

        acc = forces / C_MASS

        # Simple Euler integration

        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        # Collisions
        
        pos = Point(self.pos[0], self.pos[1])

        match world.collision(pos, C_WIDTH * 0.5):
            case v if v is not None:
                self.vel = self.vel * 0.5 + v
                self.pos += v
                self.head = self.vel / np.linalg.norm(self.vel)
                self.damaged = True
            case _:
                self.damaged = False
                
        # Sensors

        self.rays = list(world.cast_rays(pos, pr.Vector2(*self.head)))

    def draw(self, layer: int) -> None:
        if layer != 1:
            return

        for ray in self.rays:
            pr.draw_line_v(ray.start.to_vec(), ray.end.to_vec(), pr.YELLOW)  # type: ignore

        tex = res.load_texture("car")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(self.pos[0], self.pos[1], C_LENGTH, C_LENGTH),
            pr.Vector2(C_LENGTH * 0.5, C_LENGTH * 0.5),
            np.rad2deg(np.arctan2(self.head[1], self.head[0]) + math.pi / 2),
            self.color,
        )
