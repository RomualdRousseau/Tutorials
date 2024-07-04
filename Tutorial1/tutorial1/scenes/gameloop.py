from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import pyray as pr

import tutorial1.resources as res
import tutorial1.util.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car, world
from tutorial1.entities.explosion import Explosion
from tutorial1.entities.marker import Marker
from tutorial1.math import envelope
from tutorial1.math.geom import Point, distance
from tutorial1.util.types import Entity, is_bit_set

CAR_COLOR = pr.Color(255, 255, 255, 255)
CORRIDOR_COLOR = pr.Color(255, 255, 0, 64)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


@dataclass
class Context:
    player: car.Car
    corridor: envelope.Envelope
    entities: list[Entity]
    camera: pr.Camera2D
    update_state: Callable[[Context, float], str]
    lap: int = 0

    def get_previous_pos(self) -> Point:
        return self.player.prev_pos

    def get_current_pos(self) -> Point:
        return self.player.curr_pos

    def on_enter(self, marker: Marker) -> None:
        self.lap += 1

    def on_leave(self, marker: Marker) -> None:
        pass


@lru_cache(1)
def get_singleton(name: str = "default"):
    roads = world.get_singleton().roads
    start = random.choice(roads.vertice)
    stop = max(roads.vertice, key=lambda x: distance(start.point, x.point))
    corridor, _ = envelope.generare_from_spatial_graph(roads.get_shortest_path(start, stop), world.ROAD_WIDTH)

    player = car.Car(CAR_COLOR)
    player.set_corridor(corridor)
    player.set_debug_mode(True)

    entities: list[Entity] = [world, player]

    camera = pr.Camera2D(
        pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        pr.Vector2(0, 0),
        0,
        ZOOM_DEFAULT,
    )

    return Context(player, corridor, entities, camera, _update_game_mode)


def reset() -> None:
    context = get_singleton()

    for x in context.entities:
        x.reset()

    context.camera.target = pr.Vector2(*context.player.pos)

    marker = Marker(context.player.get_spawn_location(), True, world.ROAD_WIDTH * 0.5, 2)
    marker.add_listener(context)
    context.entities.append(marker)

    context.lap = 0


def update(dt: float) -> str:
    context = get_singleton()
    return get_singleton().update_state(context, dt)


def draw() -> None:
    context = get_singleton()

    pr.begin_mode_2d(context.camera)

    for x in context.entities:
        x.draw(0)

    for segment in context.corridor.segments:
        segment.draw(1, CORRIDOR_COLOR, None, True)

    for x in context.entities:
        x.draw(1)

    pr.end_mode_2d()

    prx.draw_text(f"Distance: {context.player.get_total_distance_in_km():.3f}km", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Speed: {context.player.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Time Elapsed: {datetime.timedelta(seconds=pr.get_time())}", pr.Vector2(2, 46), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Lap: {context.lap}", pr.Vector2(2, 68), 20, pr.WHITE, shadow=True)  # type: ignore

    prx.draw_text(f"{pr.get_fps()}fps", pr.Vector2(2, 2), 20, pr.WHITE, align="right", shadow=True)  # type: ignore


def _update_free_mode(context: Context, dt: float) -> str:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_ARROW)
        pr.hide_cursor()
        context.camera.zoom = ZOOM_DEFAULT
        context.update_state = _update_game_mode
    else:
        if pr.is_mouse_button_down(pr.MouseButton.MOUSE_BUTTON_LEFT):
            context.camera.target = pr.vector2_lerp(
                context.camera.target, pr.vector2_subtract(context.camera.target, pr.get_mouse_delta()), 0.2
            )
        context.camera.zoom = max(1, context.camera.zoom + pr.get_mouse_wheel_move() * 0.5)

    return "gameloop"


def _update_game_mode(context: Context, dt: float) -> str:
    prev_flags = context.player.flags

    for entity in context.entities:
        entity.update(dt)

    if (
        is_bit_set(context.player.flags, car.FLAG_DAMAGED)
        and not is_bit_set(prev_flags, car.FLAG_DAMAGED)
        and not pr.is_sound_playing(res.load_sound("crash"))
    ):
        context.entities.append(Explosion(Point(context.player.pos)))
        pr.play_sound(res.load_sound("crash"))

    if is_bit_set(context.player.flags, car.FLAG_OUT_OF_TRACK) and not pr.is_sound_playing(res.load_sound("klaxon")):
        pr.play_sound(res.load_sound("klaxon"))

    context.entities = [entity for entity in context.entities if entity.is_alive()]

    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_RESIZE_ALL)
        pr.show_cursor()
        context.camera.zoom = ZOOM_DEFAULT
        context.update_state = _update_free_mode
    else:
        context.camera.target = pr.vector2_lerp(context.camera.target, pr.Vector2(*context.player.pos), 0.2)
        context.camera.zoom = 0.8 * context.camera.zoom + 0.2 * (
            max(
                1,
                ZOOM_DEFAULT - context.player.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF,
            )
        )

    return "gameloop"
