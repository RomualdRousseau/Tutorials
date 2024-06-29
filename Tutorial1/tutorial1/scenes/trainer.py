import datetime
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyray as pr

import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car, world
from tutorial1.math.linalg import lst_2_vec
from tutorial1.util.types import Entity

BEST_CAR_COLOR = pr.Color(255, 255, 255, 255)
CAR_COLOR = pr.Color(255, 255, 255, 64)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


@dataclass
class Context:
    agents: list[car.Car]
    entities: list[Entity]
    camera: pr.Camera2D
    best_agent: Optional[car.Car]
    update_camera: Callable[[], None]
    last_spawn: Optional[world.Location] = None
    timestep: int = 0


def _init() -> Context:
    camera = pr.Camera2D(
        pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        pr.Vector2(0, 0),
        0,
        ZOOM_DEFAULT,
    )
    return Context([], [world], camera, None, _update_camera_game_mode)


def get_agents() -> list[car.Car]:
    return _context.agents


def spawn_agents(agent_count: int) -> None:
    _context.agents = [car.Car(CAR_COLOR, input_mode="ai") for _ in range(agent_count)]


def reset_agents() -> None:
    for agent in _context.agents:
        if _context.last_spawn is not None:
            agent.set_spawn_location(_context.last_spawn)


def get_agent_obs(agent: car.Car) -> dict[str, np.ndarray]:
    return {
        "agent_vel": lst_2_vec([agent.get_speed_in_kmh() / car.MAX_SPEED]),
        "agent_cam": lst_2_vec([1.0 - x.length / world.RAY_MAX_LEN for x in agent.camera]),
    }


def get_agent_score(agent: car.Car) -> float:
    score = int(agent.get_total_distance_in_km() * 1000)  # farest in meter
    score += int(agent.get_average_speed_in_kmh() / 3.6)  # fatest in meter per second
    score += -100 if agent.out_of_track else -1000  # penalties
    return score


def is_agent_alive(agent: car.Car) -> bool:
    return not agent.damaged and not agent.out_of_track and np.dot(agent.vel, agent.head) >= 0


def is_terminated() -> bool:
    return _context.best_agent is None


def reset() -> None:
    _context.entities = [world, *_context.agents]
    _context.best_agent = None
    _context.timestep += 1
    for entity in _context.entities:
        entity.reset()


def update(dt: float) -> str:
    for entity in _context.entities:
        entity.update(dt)

    alive_agents = [agent for agent in _context.agents if is_agent_alive(agent)]

    _context.entities = [world, *alive_agents]
    _context.best_agent = max(alive_agents, key=lambda x: get_agent_score(x) if x is not None else 0.0, default=None)
    _context.update_camera()

    if _context.best_agent is not None:
        _context.last_spawn = _context.best_agent.get_spawn_location()

    return "trainer"


def draw() -> None:
    for agent in _context.agents:
        agent.set_debug_mode(agent is _context.best_agent)

    pr.begin_mode_2d(_context.camera)
    for layer in range(2):
        for entity in _context.entities:
            entity.draw(layer)
    if _context.last_spawn is not None:
        _context.last_spawn[1].draw(1, pr.BLUE)  # type: ignore
    pr.end_mode_2d()

    match _context.best_agent:
        case None:
            prx.draw_text("Distance: ---", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
            prx.draw_text("Speed: ---", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore
        case best_car:
            prx.draw_text(f"Distance: {best_car.get_total_distance_in_km():.3f}km", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
            prx.draw_text(f"Speed: {best_car.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore

    prx.draw_text(f"Time Elapsed: {datetime.timedelta(seconds=pr.get_time())}", pr.Vector2(2, 46), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Time Step: {_context.timestep}", pr.Vector2(2, 68), 20, pr.WHITE, shadow=True)  # type: ignore

    prx.draw_text(f"{pr.get_fps()}fps", pr.Vector2(2, 2), 20, pr.WHITE, align="right", shadow=True)  # type: ignore


def _update_camera_free_mode() -> None:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_ARROW)
        pr.hide_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_camera = _update_camera_game_mode
    else:
        if pr.is_mouse_button_down(pr.MouseButton.MOUSE_BUTTON_LEFT):
            _context.camera.target = pr.vector2_lerp(
                _context.camera.target, pr.vector2_subtract(_context.camera.target, pr.get_mouse_delta()), 0.8
            )
        _context.camera.zoom = max(1, _context.camera.zoom + pr.get_mouse_wheel_move() * 0.5)


def _update_camera_game_mode() -> None:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_RESIZE_ALL)
        pr.show_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_camera = _update_camera_free_mode
    elif _context.best_agent is not None:
        _context.camera.target = pr.vector2_lerp(_context.camera.target, pr.Vector2(*_context.best_agent.pos), 0.8)
        _context.camera.zoom = 0.8 * _context.camera.zoom + 0.2 * (
            max(
                1,
                ZOOM_DEFAULT - _context.best_agent.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF,
            )
        )


_context = _init()
