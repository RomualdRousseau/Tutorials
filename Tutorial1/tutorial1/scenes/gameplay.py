from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import lru_cache

import pyray as pr

import tutorial1.resources as res
import tutorial1.utils.pyray_ex as prx
from tutorial1.cameras.camera_follower import CameraFollower
from tutorial1.effects.fade_scr import FadeScr
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
    state: int
    entities: list[Entity]
    player: TaxiDriver
    camera: CameraFollower
    minimap: Minimap
    fade_in: FadeScr


@lru_cache(1)
def get_singleton(name: str = "default"):
    player = TaxiDriver("human")
    entities: list[Entity] = [player]
    camera = CameraFollower(player.car)
    minimap = Minimap(player.car)
    fade_in = FadeScr(1)
    return Context(0, entities, player, camera, minimap, fade_in)


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

    world.update(dt)
    for entity in ctx.entities:
        entity.update(dt)
    ctx.entities = [entity for entity in ctx.entities if entity.is_alive()]
    ctx.minimap.update(dt)
    ctx.camera.update(dt)

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

            if (
                is_bit_set(ctx.player.car.flags, car.FLAG_DAMAGED)
                and not is_bit_set(prev_flags, car.FLAG_DAMAGED)
                and not pr.is_sound_playing(res.load_sound("crash"))
            ):
                ctx.entities.append(Explosion(ctx.player.car.curr_pos))
                pr.play_sound(res.load_sound("crash"))

            if is_bit_set(ctx.player.car.flags, car.FLAG_OUT_OF_TRACK) and not pr.is_sound_playing(
                res.load_sound("klaxon")
            ):
                pr.play_sound(res.load_sound("klaxon"))

    return "gameplay"


def draw() -> None:
    ctx = get_singleton()

    pr.begin_mode_2d(ctx.camera.camera)
    for layer in range(2):
        world.draw(layer)
        for entity in ctx.entities:
            entity.draw(layer)
    pr.end_mode_2d()

    ctx.minimap.draw(3)

    if ctx.player.state == TaxiDriver.STATE_PICKUP_WAIT:
        tex = res.load_texture("pickup")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(100, 100, 200, 200),
            pr.Vector2(0, 0),
            0,
            pr.WHITE,  # type: ignore
        )

    if ctx.player.state == TaxiDriver.STATE_DROPOFF_WAIT:
        tex = res.load_texture("dropoff")
        pr.draw_texture_pro(
            tex,
            pr.Rectangle(0, 0, tex.width, tex.height),
            pr.Rectangle(100, 100, 200, 200),
            pr.Vector2(0, 0),
            0,
            pr.WHITE,  # type: ignore
        )

    prx.draw_text(f"Distance: {ctx.player.car.get_total_distance_in_km():.3f}km", pr.Vector2(2, 2), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Speed: {ctx.player.car.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 24), 20, pr.WHITE, shadow=True)  # type: ignore
    prx.draw_text(f"Time Elapsed: {datetime.timedelta(seconds=pr.get_time())}", pr.Vector2(2, 46), 20, pr.WHITE, shadow=True)  # type: ignore

    prx.draw_text(f"{pr.get_fps()}fps", pr.Vector2(2, 2), 20, pr.WHITE, align="right", shadow=True)  # type: ignore

    if ctx.state == 0:
        ctx.fade_in.draw()
