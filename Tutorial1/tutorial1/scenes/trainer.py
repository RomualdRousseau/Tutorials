from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pyray as pr

import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car, world
from tutorial1.math.linalg import lst_2_vec
from tutorial1.util.types import Entity

CAR_BEST_COLOR = pr.Color(255, 255, 255, 255)
CAR_COLOR = pr.Color(255, 255, 255, 64)
CAR_MIN_SPEED = 5

ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


@dataclass
class Context:
    agents: list[car.Car]
    entities: list[Entity]
    camera: pr.Camera2D
    best_agent: Optional[car.Car]
    update_camera: Callable[[Context], None]
    last_spawn: Optional[world.Location] = None
    timestep: int = 0


@lru_cache
def get_singleton(name: str = "default"):
    camera = pr.Camera2D(
        pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        pr.Vector2(0, 0),
        0,
        ZOOM_DEFAULT,
    )
    return Context([], [world], camera, None, _update_camera_game_mode)


def get_agents() -> list[car.Car]:
    return get_singleton().agents


def spawn_agents(agent_count: int) -> None:
    get_singleton().agents = [car.Car(CAR_COLOR, input_mode="ai") for _ in range(agent_count)]


def reset_agents() -> None:
    context = get_singleton()
    for agent in context.agents:
        if context.last_spawn is not None:
            agent.set_spawn_location(context.last_spawn)


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
    return (
        not agent.damaged
        and not agent.out_of_track
        and np.dot(agent.vel, agent.head) >= 0
        and agent.get_speed_in_kmh() >= CAR_MIN_SPEED
    )


def is_terminated() -> bool:
    return get_singleton().best_agent is None


def reset() -> None:
    pr.trace_log(pr.TraceLogLevel.LOG_DEBUG, "TRAINER: reset")
    context = get_singleton()
    context.entities = [world, *context.agents]
    context.best_agent = None
    context.timestep += 1
    for entity in context.entities:
        entity.reset()


def update(dt: float) -> str:
    context = get_singleton()
    for entity in context.entities:
        entity.update(dt)

    if context.best_agent is not None and not is_agent_alive(context.best_agent):
        pr.trace_log(
            pr.TraceLogLevel.LOG_INFO,
            f"{not context.best_agent.damaged} \
            {not context.best_agent.out_of_track} \
            {np.dot(context.best_agent.vel, context.best_agent.head) >= 0} \
            {context.best_agent.get_speed_in_kmh() >= CAR_MIN_SPEED} \
            {get_agent_score(context.best_agent)}",
        )

    alive_agents = [agent for agent in context.agents if is_agent_alive(agent)]

    context.entities = [world, *alive_agents]
    context.best_agent = max(alive_agents, key=lambda x: get_agent_score(x) if x is not None else 0.0, default=None)
    context.update_camera(context)

    if context.best_agent is not None:
        context.last_spawn = context.best_agent.get_spawn_location()

    return "trainer"


def draw() -> None:
    context = get_singleton()

    for agent in context.agents:
        agent.set_debug_mode(agent is context.best_agent)

    pr.begin_mode_2d(context.camera)
    for layer in range(2):
        for entity in context.entities:
            entity.draw(layer)
    if context.last_spawn is not None:
        context.last_spawn[1].draw(1, pr.BLUE)  # type: ignore
    pr.end_mode_2d()

    match context.best_agent:
        case None:
            prx.draw_text("Distance: ---", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
            prx.draw_text("Speed: ---", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore
        case best_car:
            prx.draw_text(f"Distance: {best_car.get_total_distance_in_km():.3f}km", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
            prx.draw_text(f"Speed: {best_car.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore

    prx.draw_text(f"Time Elapsed: {datetime.timedelta(seconds=pr.get_time())}", pr.Vector2(2, 46), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Time Step: {context.timestep}", pr.Vector2(2, 68), 20, pr.WHITE, shadow=True)  # type: ignore

    prx.draw_text(f"{pr.get_fps()}fps", pr.Vector2(2, 2), 20, pr.WHITE, align="right", shadow=True)  # type: ignore


def _update_camera_free_mode(_context: Context) -> None:
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


def _update_camera_game_mode(_context: Context) -> None:
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
