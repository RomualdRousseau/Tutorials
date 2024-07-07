from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pyray as pr

import tutorial1.utils.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car, world
from tutorial1.entities.explosion import Explosion
from tutorial1.entities.marker import Marker
from tutorial1.math import envelope
from tutorial1.math.geom import Point, distance
from tutorial1.math.linalg import lst_2_vec
from tutorial1.utils.bitbang import is_bit_set
from tutorial1.utils.types import Entity

CAR_BEST_COLOR = pr.Color(255, 255, 255, 255)
CAR_COLOR = pr.Color(255, 255, 255, 64)
CAR_MIN_SPEED = 5
CORRIDOR_COLOR = pr.Color(255, 255, 0, 64)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


@dataclass
class Context:
    agents: list[car.Car]
    entities: list[Entity]
    camera: pr.Camera2D
    update_camera: Callable[[Context], None]
    corridor: Optional[envelope.Envelope] = None
    best_agent: Optional[car.Car] = None
    last_spawn_location: Optional[envelope.Location] = None
    spawn_location_changed: bool = False
    timestep: int = 0
    lap: int = 0

    def get_previous_pos(self) -> Point:
        return self.best_agent.prev_pos if self.best_agent is not None else Point(np.zeros(2))

    def get_current_pos(self) -> Point:
        return self.best_agent.curr_pos if self.best_agent is not None else Point(np.zeros(2))

    def on_enter(self, marker: Marker) -> None:
        self.lap += 1

    def on_leave(self, marker: Marker) -> None:
        pass


@lru_cache(1)
def get_singleton(name: str = "default"):
    camera = pr.Camera2D(
        pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        pr.Vector2(0, 0),
        0,
        ZOOM_DEFAULT,
    )
    return Context([], [world], camera, _update_camera_game_mode)


def reset_corridor():
    context = get_singleton()
    roads = world.get_singleton().roads
    start = random.choice(roads.vertice)
    stop = max(roads.vertice, key=lambda x: distance(start.point, x.point))
    context.corridor, _ = envelope.generare_borders_from_spatial_graph(
        roads.get_shortest_path(start, stop), world.ROAD_WIDTH, []
    )


def get_agents() -> list[car.Car]:
    return get_singleton().agents


def spawn_agents(agent_count: int) -> None:
    get_singleton().agents = [car.Car(CAR_COLOR, input_mode="ai", vin=i) for i in range(agent_count)]


def reset_agents() -> None:
    context = get_singleton()
    assert context.corridor is not None
    for agent in context.agents:
        agent.set_corridor(context.corridor)
        if context.last_spawn_location is not None:
            agent.set_spawn_location(context.last_spawn_location)


def get_best_agent():
    context = get_singleton()
    return context.best_agent


def get_agent_obs(agent: car.Car) -> dict[str, np.ndarray]:
    return {
        "agent_vel": lst_2_vec([agent.get_speed_in_kmh() / car.MAX_SPEED]),
        "agent_cam": lst_2_vec([1.0 - x.length / car.RAY_MAX_LEN for x in agent.camera]),
    }


def get_agent_score(agent: car.Car) -> float:
    score = int(agent.get_total_distance_in_km() * 1000)  # farest in meter
    score += int(agent.get_average_speed_in_kmh() * 10 / car.MAX_SPEED)  # fatest in meter per second
    score += -10 if is_bit_set(agent.flags, car.FLAG_OUT_OF_TRACK) else -100  # penalties
    return score


def is_agent_alive(agent: car.Car) -> bool:
    return (
        not is_bit_set(agent.flags, car.FLAG_DAMAGED)
        and not is_bit_set(agent.flags, car.FLAG_OUT_OF_TRACK)
        and np.dot(agent.vel, agent.head) >= 0
        and agent.get_speed_in_kmh() >= CAR_MIN_SPEED
    )


def has_spawn_location_changed() -> bool:
    return get_singleton().spawn_location_changed


def is_terminated() -> bool:
    return get_singleton().best_agent is None


def reset() -> None:
    context = get_singleton()

    if context.corridor is None:
        reset_corridor()

    reset_agents()

    context.entities = [world, *context.agents]
    context.best_agent = None
    context.timestep += 1

    for entity in context.entities:
        entity.reset()

    if context.corridor is not None:
        marker = Marker(context.agents[0].get_spawn_location(), world.ROAD_WIDTH * 0.5, 2, context.agents[0].head)
        marker.add_listener(context)
        context.entities.append(marker)


def update(dt: float) -> str:
    context = get_singleton()

    for entity in context.entities:
        entity.update(dt)

    for agent in context.agents:
        if agent.is_alive() and not is_agent_alive(agent):
            agent.hit(car.MAX_LIFE)
            context.entities.append(Explosion(Point(agent.pos)))

    context.best_agent = max((x for x in context.agents if x.is_alive()), key=get_agent_score, default=None)
    if context.best_agent is not None:
        last_spawn_location = context.best_agent.get_spawn_location()
        context.spawn_location_changed = context.last_spawn_location != last_spawn_location
        context.last_spawn_location = last_spawn_location

    context.entities = [entity for entity in context.entities if entity.is_alive()]
    context.update_camera(context)

    return "trainer"


def draw() -> None:
    context = get_singleton()
    assert context.corridor is not None

    for agent in context.agents:
        agent.set_debug_mode(agent is context.best_agent)

    pr.begin_mode_2d(context.camera)

    for entity in context.entities:
        entity.draw(0)

    for segment in context.corridor.segments:
        segment.draw(1, CORRIDOR_COLOR, None, True)

    if context.last_spawn_location is not None:
        context.last_spawn_location[1].draw(1, CORRIDOR_COLOR)  # type: ignore

    for entity in context.entities:
        entity.draw(1)

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
    prx.draw_text(f"Lap: {context.lap}", pr.Vector2(2, 90), 20, pr.WHITE, shadow=True)  # type: ignore

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
                _context.camera.target, pr.vector2_subtract(_context.camera.target, pr.get_mouse_delta()), 0.2
            )
        _context.camera.zoom = max(1, _context.camera.zoom + pr.get_mouse_wheel_move() * 0.5)


def _update_camera_game_mode(_context: Context) -> None:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_RESIZE_ALL)
        pr.show_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_camera = _update_camera_free_mode
    elif _context.best_agent is not None:
        _context.camera.target = pr.vector2_lerp(_context.camera.target, pr.Vector2(*_context.best_agent.pos), 0.2)
        _context.camera.zoom = 0.8 * _context.camera.zoom + 0.2 * (
            max(
                1,
                ZOOM_DEFAULT - _context.best_agent.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF,
            )
        )
