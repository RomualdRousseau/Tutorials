from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import pyray as pr

import tutorial1.resources as res
import tutorial1.utils.pyray_ex as prx
from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car, world
from tutorial1.entities.explosion import Explosion
from tutorial1.entities.minimap import Minimap
from tutorial1.entities.taxi_driver import TaxiDriver
from tutorial1.utils.bitbang import is_bit_set
from tutorial1.utils.types import Entity

CAR_COLOR = pr.Color(255, 255, 255, 255)
CORRIDOR_COLOR = pr.Color(255, 255, 0, 64)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


@dataclass
class Context:
    player: TaxiDriver
    minimap: Minimap
    entities: list[Entity]
    camera: pr.Camera2D
    update_state: Callable[[Context, float], str]


@lru_cache(1)
def get_singleton(name: str = "default"):
    player = TaxiDriver("human")
    minimap = Minimap(player.car)
    entities: list[Entity] = [world, player, minimap]
    camera = pr.Camera2D(
        pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        pr.Vector2(0, 0),
        0,
        ZOOM_DEFAULT,
    )
    return Context(player, minimap, entities, camera, _update_game_mode)


def reset() -> None:
    context = get_singleton()
    for x in context.entities:
        x.reset()
    context.camera.target = context.player.car.curr_pos.to_vec()


def update(dt: float) -> str:
    context = get_singleton()
    return get_singleton().update_state(context, dt)


def draw() -> None:
    context = get_singleton()

    pr.begin_mode_2d(context.camera)
    for layer in range(2):
        for entity in context.entities:
            entity.draw(layer)
    pr.end_mode_2d()

    context.minimap.draw(3)

    if context.player.state == TaxiDriver.STATE_PICKUP_WAIT:
        tex = res.load_texture("pickup")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(100, 100, 200, 200),
            pr.Vector2(0, 0),
            0,
            pr.WHITE,  # type: ignore
        )

    if context.player.state == TaxiDriver.STATE_DROPOFF_WAIT:
        tex = res.load_texture("dropoff")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(100, 100, 200, 200),
            pr.Vector2(0, 0),
            0,
            pr.WHITE,  # type: ignore
        )

    prx.draw_text(f"Distance: {context.player.car.get_total_distance_in_km():.3f}km", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Speed: {context.player.car.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Time Elapsed: {datetime.timedelta(seconds=pr.get_time())}", pr.Vector2(2, 46), 20, pr.WHITE, shadow=True)  # type: ignore

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
    prev_flags = context.player.car.flags

    if pr.is_key_pressed(pr.KeyboardKey.KEY_A):
        context.player.accept_call(
            world.get_singleton().borders.get_random_location(),
            world.get_singleton().borders.get_random_location(),
        )

    for entity in context.entities:
        entity.update(dt)

    if (
        is_bit_set(context.player.car.flags, car.FLAG_DAMAGED)
        and not is_bit_set(prev_flags, car.FLAG_DAMAGED)
        and not pr.is_sound_playing(res.load_sound("crash"))
    ):
        context.entities.append(Explosion(context.player.car.curr_pos))
        pr.play_sound(res.load_sound("crash"))

    if is_bit_set(context.player.car.flags, car.FLAG_OUT_OF_TRACK) and not pr.is_sound_playing(
        res.load_sound("klaxon")
    ):
        pr.play_sound(res.load_sound("klaxon"))

    context.entities = [entity for entity in context.entities if entity.is_alive()]

    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_RESIZE_ALL)
        pr.show_cursor()
        context.camera.zoom = ZOOM_DEFAULT
        context.update_state = _update_free_mode
    else:
        context.camera.target = pr.vector2_lerp(context.camera.target, context.player.car.curr_pos.to_vec(), 0.2)
        context.camera.zoom = 0.8 * context.camera.zoom + 0.2 * (
            max(
                1,
                ZOOM_DEFAULT - context.player.car.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF,
            )
        )

    return "gameloop"
