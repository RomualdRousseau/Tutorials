from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pyray as pr

import taxi_driver_env.resources as res
import taxi_driver_env.utils.pyrayex as prx
from taxi_driver_env.cameras.camera_follower import CameraFollower
from taxi_driver_env.effects.fade_scr import FadeScr
from taxi_driver_env.effects.open_vertical import OpenVertical
from taxi_driver_env.entities import car, world
from taxi_driver_env.entities.badge import Badge
from taxi_driver_env.entities.explosion import Explosion
from taxi_driver_env.entities.minimap import Minimap
from taxi_driver_env.entities.taxi_driver import WAIT_TIMER, TaxiDriver
from taxi_driver_env.types import Entity, Widget
from taxi_driver_env.utils.bitbang import is_bit_set
from taxi_driver_env.widgets.message_box import MessageBox

CAR_COLOR = pr.Color(255, 255, 255, 255)
CORRIDOR_COLOR = pr.Color(255, 255, 0, 64)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


@dataclass
class Context:
    state: int
    player: TaxiDriver
    camera: CameraFollower
    minimap: Minimap
    entities: list[Entity]
    fade_in: FadeScr
    message_box: Optional[OpenVertical]
    minimap_visible: bool = True


@lru_cache(1)
def get_singleton(name: str = "default"):
    player = TaxiDriver("human")
    camera = CameraFollower(player.car)
    minimap = Minimap(player)
    fade_in = FadeScr(1)
    entities: list[Entity] = [world, player]
    return Context(0, player, camera, minimap, entities, fade_in, None)


def reset() -> None:
    res.clear_caches()
    ctx = get_singleton()
    ctx.state = 0
    for x in ctx.entities:
        x.reset()
    ctx.camera.reset()
    ctx.minimap.reset()
    ctx.fade_in.reset()


def update(dt: float) -> str:
    ctx = get_singleton()
    prev_flags = ctx.player.car.flags

    if pr.is_key_pressed(pr.KeyboardKey.KEY_F1):
        ctx.player.car.set_debug_mode(not ctx.player.car.debug_mode)
    if pr.is_key_pressed(pr.KeyboardKey.KEY_F2):
        ctx.minimap_visible = not ctx.minimap_visible

    match ctx.state:
        case 0:
            ctx.fade_in.update(dt)
            if not ctx.fade_in.is_playing():
                ctx.state = 1

        case 1:
            if pr.is_key_pressed(pr.KeyboardKey.KEY_A):
                ctx.player.accept_call(
                    world.get_singleton().borders.get_random_location(),
                    world.get_singleton().borders.get_random_location(),
                )

            if ctx.player.state == TaxiDriver.STATE_ACCEPTING_CALL and ctx.message_box is None:
                ctx.message_box = OpenVertical(
                    MessageBox(
                        pr.Vector2(800, 256),
                        "\nYou receive a call from a customer!\n\nPlease go to the pickup ...",
                        title="GRAB:",
                        icon="accept",
                        callback=message_box_cb,
                    ),
                    0.5,
                    MessageBox.BG_COLOR,
                )
                ctx.message_box.reset()
                pr.show_cursor()
                ctx.state = 2

            if ctx.player.state == TaxiDriver.STATE_PICKUP_WAIT and ctx.message_box is None:
                ctx.message_box = OpenVertical(
                    MessageBox(
                        pr.Vector2(800, 256),
                        "\nWelcome to my taxi!\n\nAttach your seat belt and let's go ...",
                        title="YOU:",
                        icon="pickup",
                        callback=message_box_cb,
                    ),
                    0.5,
                    MessageBox.BG_COLOR,
                )
                ctx.message_box.reset()
                pr.show_cursor()
                ctx.state = 2

            if ctx.player.state == TaxiDriver.STATE_DROPOFF_WAIT and ctx.message_box is None:
                ctx.message_box = OpenVertical(
                    MessageBox(
                        pr.Vector2(800, 256),
                        "\nGoodbye!\n\nMake sure to check for your belongings.\n\nHave a nice nice day!",
                        title="YOU:",
                        icon="dropoff",
                        callback=message_box_cb,
                    ),
                    0.5,
                    MessageBox.BG_COLOR,
                )
                ctx.message_box.reset()
                pr.show_cursor()
                ctx.state = 2

        case 3:
            if ctx.player.state == TaxiDriver.STATE_DROPOFF_WAIT:
                ctx.entities.append(Badge(ctx.player.car.curr_pos, "+$5"))
                ctx.player.money += 5
            ctx.message_box = None
            ctx.player.timer = WAIT_TIMER
            pr.hide_cursor()
            ctx.state = 1

    if ctx.message_box is None:
        for entity in ctx.entities:
            entity.update(dt)
        ctx.entities = [entity for entity in ctx.entities if entity.is_alive()]
        ctx.minimap.update(dt)
        ctx.camera.update(dt)
    else:
        ctx.message_box.update(dt)

    if (
        is_bit_set(ctx.player.car.flags, car.FLAG_DAMAGED)
        and not is_bit_set(prev_flags, car.FLAG_DAMAGED)
        and not pr.is_sound_playing(res.load_sound("crash"))
    ):
        ctx.entities.append(Explosion(ctx.player.car.curr_pos))
        ctx.entities.append(Badge(ctx.player.car.curr_pos, "-$100"))
        ctx.player.money -= 100
        pr.play_sound(res.load_sound("crash"))

    if is_bit_set(ctx.player.car.flags, car.FLAG_OUT_OF_TRACK) and not pr.is_sound_playing(res.load_sound("klaxon")):
        ctx.entities.append(Badge(ctx.player.car.curr_pos, "-$10"))
        ctx.player.money -= 10
        pr.play_sound(res.load_sound("klaxon"))

    return "gameplay"


def draw() -> None:
    ctx = get_singleton()

    pr.begin_mode_2d(ctx.camera.camera)
    for layer in range(2):
        for entity in ctx.entities:
            entity.draw(layer)
    pr.end_mode_2d()

    if ctx.minimap_visible:
        ctx.minimap.draw()

    if ctx.message_box is not None:
        ctx.message_box.draw()

    if ctx.player.car.debug_mode:
        prx.draw_text(f"Distance: {ctx.player.car.get_total_distance_in_km():.3f}km", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
        prx.draw_text(f"Speed: {ctx.player.car.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore
        prx.draw_text(f"Time Elapsed: {datetime.timedelta(seconds=pr.get_time())}", pr.Vector2(2, 46), 20, pr.WHITE, shadow=True)  # type: ignore
        prx.draw_text(f"{pr.get_fps()}fps", pr.Vector2(2, 2), 20, pr.WHITE, align="right", shadow=True)  # type: ignore

    prx.draw_text(f"${ctx.player.money}", pr.Vector2(2, 2), 20, pr.WHITE, align="left", shadow=True)  # type: ignore

    if ctx.state == 0:
        ctx.fade_in.draw()


def message_box_cb(_: Widget) -> None:
    ctx = get_singleton()
    ctx.state = 3
