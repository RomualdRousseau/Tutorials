import math
import numpy as np
import pyray as pr

from tutorial1.util.geom import Point
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


class Car:
    def __init__(self, color: pr.Color) -> None:
        self.vertice = [
            pr.Vector2(-C_LENGTH * 0.5, -C_WIDTH * 0.5),
            pr.Vector2(C_LENGTH * 0.5, 0),
            pr.Vector2(-C_LENGTH * 0.5, C_WIDTH * 0.5),
        ]
        self.color = color
        
        self.pos = pr.Vector2(300, 300)
        self.vel = pr.Vector2(0, 0)
        self.head = pr.Vector2(0, -1)

        self.wheel = 0
        self.throttle = 0
        
        self.rays = []

    def get_speed_in_kmh(self) -> int:
        return int(pr.vector2_length(self.vel) * 3.6)

    def turn_wheel(self, t: float):
        self.wheel = float(
            np.interp(t, [-1, 1], [-C_WHEEL_ANGLE_RATE, C_WHEEL_ANGLE_RATE])
        )

    def push_throttle(self, t: float):
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

        forces = pr.vector2_zero()

        if self.wheel != 0:
            circ_radius = C_LENGTH / (math.sin(self.wheel))
            ang_vel = pr.vector2_length(self.vel) / circ_radius
            self.head = pr.vector2_rotate(self.head, ang_vel)

        tract = pr.vector2_scale(self.head, self.throttle)
        forces = pr.vector2_add(forces, tract)

        drag_rd = pr.vector2_scale(self.vel, -C_DRAG_ROAD * C_MASS * C_G)
        forces = pr.vector2_add(forces, drag_rd)

        drag_rr = pr.vector2_scale(self.vel, -C_DRAG_ROLLING * C_MASS * C_G)
        forces = pr.vector2_add(forces, drag_rr)

        acc = pr.vector2_scale(forces, 1 / C_MASS)

        # Simple Euler integration

        self.vel = pr.vector2_add(self.vel, pr.vector2_scale(acc, dt))
        self.pos = pr.vector2_add(self.pos, pr.vector2_scale(self.vel, dt))

        # Sensors

        self.rays = [
            ray for ray in world.cast_rays(Point(self.pos.x, self.pos.y), self.head)
        ]

    def draw(self, layer: int) -> None:
        if layer != 1:
            return
        
        for ray in self.rays:
            pr.draw_line_v(ray.start.to_vec(), ray.end.to_vec(), pr.YELLOW)  # type: ignore

        sprite = res.load_texture("car")
        pr.draw_texture_pro(
            sprite,
            pr.Rectangle(0, 0, sprite.width, sprite.height),
            pr.Rectangle(self.pos.x, self.pos.y, C_LENGTH, C_LENGTH),
            pr.Vector2(C_LENGTH / 2, C_LENGTH / 2),
            math.atan2(self.head.y, self.head.x) * 180 / math.pi + 90,
            self.color
        )
