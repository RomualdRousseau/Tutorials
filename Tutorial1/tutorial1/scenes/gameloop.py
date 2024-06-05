from functools import partial
import pyray as pr

import tutorial1.world as world
import tutorial1.car as car

from tutorial1.util.types import Entity
from tutorial1.util.funcs import apply

CAR_COLOR = pr.Color(255, 255, 255, 255)


car = car.Car(CAR_COLOR)
entities: list[Entity] = [world, car]
camera = pr.Camera2D(
    car.pos, pr.Vector2(pr.get_screen_width() * 0.5, pr.get_screen_height() * 0.5), 0, 4
)


def update(dt: float) -> str:
    f_update = partial(apply, lambda y: y.update(dt))  # type: ignore
    global entities
    entities = [f_update(x) for x in entities if x.is_alive()]
    camera.target = car.pos
    camera.zoom = max(1, camera.zoom + pr.get_mouse_wheel_move() * 0.5)
    return "gameloop"


def draw() -> None:
    pr.begin_mode_2d(camera)
    for l in range(3):
        [x.draw(l) for x in entities]
    pr.end_mode_2d()
    pr.draw_text(f"{car.get_speed_in_kmh()}km/h", 2, 2, 22, pr.BLACK)  # type: ignore
    pr.draw_text(f"{car.get_speed_in_kmh()}km/h", 1, 1, 22, pr.WHITE)  # type: ignore
