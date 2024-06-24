from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyray as pr

import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car, world
from tutorial1.util.types import Entity

BEST_CAR_COLOR = pr.Color(255, 255, 255, 255)
CAR_COLOR = pr.Color(255, 255, 255, 64)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1
MIN_SPEED = 5  # km.h-1


@dataclass
class Context:
    cars: list[car.Car]
    entities: list[Entity]
    camera: pr.Camera2D
    best_car: Optional[car.Car]
    update_camera: Callable[[], None]


def _init() -> Context:
    camera = pr.Camera2D(
        pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        pr.Vector2(0, 0),
        0,
        ZOOM_DEFAULT,
    )
    return Context([], [world], camera, None, _update_camera_game_mode)


def add_agents(agent_count: int) -> None:
    _context.cars = [car.Car(CAR_COLOR, input_mode="ai") for _ in range(agent_count)]


def get_agents() -> list[car.Car]:
    return _context.cars


def get_agent_scores() -> list[float]:
    return [_get_agent_score(x) for x in _context.cars]


def reset() -> None:
    _context.entities = [world, *_context.cars]
    for x in _context.entities:
        x.reset()


def update(dt: float) -> str:
    for x in _context.entities:
        x.update(dt)

    alive_agents = [x for x in _context.cars if _is_agent_alive(x)]
    _context.entities = [world, *alive_agents]
    _context.best_car = max(alive_agents, key=_get_agent_score, default=None)
    _context.update_camera()

    return "trainer"


def draw() -> None:
    pr.begin_mode_2d(_context.camera)
    for layer in range(2):
        for x in _context.entities:
            if isinstance(x, car.Car):
                x.color = BEST_CAR_COLOR if x is _context.best_car else CAR_COLOR
            x.draw(layer)
    pr.end_mode_2d()

    match _context.best_car:
        case None:
            prx.draw_text_shadow("Distance: ---", pr.Vector2(2, 2), 22, pr.WHITE)  # type: ignore
            prx.draw_text_shadow("Speed: ---", pr.Vector2(2, 23), 22, pr.WHITE)  # type: ignore
        case best_car:
            prx.draw_text_shadow(f"Distance: {best_car.get_travel_distance_in_km():.3f}km", pr.Vector2(2, 2), 22, pr.WHITE)  # type: ignore
            prx.draw_text_shadow(f"Speed: {best_car.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 23), 22, pr.WHITE)  # type: ignore

    prx.draw_text_shadow(f"{pr.get_fps()}fps", pr.Vector2(2, 44), 22, pr.WHITE)  # type: ignore


def _get_agent_score(agent: Optional[car.Car]) -> float:
    return agent.get_travel_distance_in_km() if agent else 0.0


def _is_agent_alive(agent: car.Car) -> bool:
    return agent is not None and (
        not agent.damaged
        and not agent.out_of_track
        and agent.get_speed_in_kmh() >= MIN_SPEED
        and np.dot(agent.vel, agent.head) >= 0
    )


def _update_camera_free_mode() -> None:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_ARROW)
        pr.hide_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_camera = _update_camera_game_mode
    else:
        if pr.is_mouse_button_down(pr.MouseButton.MOUSE_BUTTON_LEFT):
            _context.camera.target = pr.vector2_subtract(_context.camera.target, pr.get_mouse_delta())
        _context.camera.zoom = max(1, _context.camera.zoom + pr.get_mouse_wheel_move() * 0.5)


def _update_camera_game_mode() -> None:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_RESIZE_ALL)
        pr.show_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_camera = _update_camera_free_mode
    elif _context.best_car is not None:
        _context.camera.target = pr.Vector2(*_context.best_car.pos)
        _context.camera.zoom = 0.8 * _context.camera.zoom + 0.2 * (
            max(
                1,
                ZOOM_DEFAULT - _context.best_car.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF,
            )
        )


_context = _init()
