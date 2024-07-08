import pyray as pr

from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.entities import car

ZOOM_DEFAULT = 20
ZOOM_ACCELERATION_COEF = 0.1


class CameraFollower:
    def __init__(self, acar: car.Car):
        self.camera = pr.Camera2D(
            pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
            pr.Vector2(0, 0),
            0,
            ZOOM_DEFAULT,
        )
        self.car = acar

    def get_bound(self) -> pr.Rectangle:
        return pr.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    def reset(self) -> None:
        self.camera.target = self.car.curr_pos.to_vec()

    def update(self, dt: float) -> None:
        self.camera.target = pr.vector2_lerp(self.camera.target, self.car.curr_pos.to_vec(), 0.2)
        self.camera.zoom = 0.8 * self.camera.zoom + 0.2 * (
            max(
                1,
                ZOOM_DEFAULT - self.car.get_speed_in_kmh() * ZOOM_ACCELERATION_COEF,
            )
        )

    def draw(self) -> None:
        pass
