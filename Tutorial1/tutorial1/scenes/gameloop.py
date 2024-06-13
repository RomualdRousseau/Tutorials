from dataclasses import dataclass
from typing import Callable

import pyray as pr

from tutorial1.util.types import Entity
from tutorial1.util.funcs import compose, constant, apply

import tutorial1.util.pyray_ex as prx
import tutorial1.world as world
import tutorial1.car as car


CAR_COLOR = pr.Color(255, 255, 255, 255)
ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.2


@dataclass
class Context:
    player: car.Car
    entities: list[Entity]
    camera: pr.Camera2D
    update_state: Callable[[float], str]


def init() -> Context:
    _player = car.Car(CAR_COLOR)
    _entities: list[Entity] = [world, _player]
    _camera = pr.Camera2D(
        pr.Vector2(300, 300),
        pr.Vector2(*_player.pos),
        0,
        ZOOM_DEFAULT,
    )
    return Context(_player, _entities, _camera, update_game_mode)


def update_free_mode(dt: float) -> str:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_ARROW)
        pr.hide_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_state = update_game_mode
    else:
        if pr.is_mouse_button_down(pr.MouseButton.MOUSE_BUTTON_LEFT):
            _context.camera.target = pr.vector2_subtract(
                _context.camera.target, pr.get_mouse_delta()
            )
        _context.camera.zoom = max(
            1, _context.camera.zoom + pr.get_mouse_wheel_move() * 0.5
        )
    return "gameloop"


def update_game_mode(dt: float) -> str:
    if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_RIGHT):
        pr.set_mouse_cursor(pr.MouseCursor.MOUSE_CURSOR_RESIZE_ALL)
        pr.show_cursor()
        _context.camera.zoom = ZOOM_DEFAULT
        _context.update_state = update_free_mode
    else:
        _context.camera.target = pr.Vector2(*_context.player.pos)
        _context.camera.zoom = 0.8 * _context.camera.zoom + 0.2 * (
            ZOOM_DEFAULT - _context.player.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF
        )
        _context.entities = [
            compose(constant(x))(apply(dt))(x.update)
            for x in _context.entities
            if x.is_alive()
        ]
    return "gameloop"


update = lambda dt: _context.update_state(dt)


def draw() -> None:
    pr.begin_mode_2d(_context.camera)
    for l in range(2):
        [x.draw(l) for x in _context.entities]
    pr.end_mode_2d()

    if _context.update_state == update_game_mode:
        prx.draw_text_shadow(f"Distance: {_context.player.get_travel_distance_in_km():.3f}km", pr.Vector2(2, 2), 22, pr.WHITE)  # type: ignore
        prx.draw_text_shadow(f"Speed: {_context.player.get_speed_in_kmh():.1f}km/h", pr.Vector2(2, 23), 22, pr.WHITE)  # type: ignore
        prx.draw_text_shadow(f"{pr.get_fps()}fps", pr.Vector2(2, 44), 22, pr.WHITE)  # type: ignore
    elif _context.update_state == update_free_mode:
        prx.draw_text_shadow(f"Zoom: x{_context.camera.zoom}", pr.Vector2(2, 2), 22, pr.WHITE)  # type: ignore
        prx.draw_text_shadow(f"{pr.get_fps()}fps", pr.Vector2(2, 23), 22, pr.WHITE)  # type: ignore


_context = init()
