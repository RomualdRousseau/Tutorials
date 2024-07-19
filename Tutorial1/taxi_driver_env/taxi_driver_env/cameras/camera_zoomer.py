import pyray as pr
from taxi_driver_env.math.linalg import absmin

BORDER = 10
ZOOM_RATE = 0.001


class CameraZoomer:
    def __init__(
        self, map_width: float, map_height: float, tex_width: float, tex_height: float
    ) -> None:
        self.width = map_width
        self.height = map_height
        self.rx = map_width / tex_width
        self.ry = map_height / tex_height
        self.camera = pr.Camera2D(
            pr.Vector2(tex_width // 2, tex_height // 2),
            pr.Vector2(0, 0),
            0,
            1,
        )

    def set_bound(self, bound: pr.Rectangle) -> None:
        self.bound = bound

    def reset(self) -> None:
        self.camera.zoom = 1

    def update(self, dt: float) -> None:
        target = pr.Vector2(
            self.bound.x + self.bound.width // 2, self.bound.y + self.bound.height // 2
        )
        self.camera.target = pr.vector2_lerp(self.camera.target, target, 0.2)

        xy1 = pr.Vector2(self.bound.x, self.bound.y)
        xy1 = pr.get_world_to_screen_2d(xy1, self.camera)
        xy2 = pr.Vector2(
            self.bound.x + self.bound.width, self.bound.y + self.bound.height
        )
        xy2 = pr.get_world_to_screen_2d(xy2, self.camera)
        dx = absmin(xy1.x * self.rx - BORDER, self.width - xy2.x * self.rx + BORDER)
        dy = absmin(xy1.y * self.ry - BORDER, self.height - xy2.y * self.ry + BORDER)
        self.camera.zoom += absmin(dx, dy) * ZOOM_RATE

    def draw(self) -> None:
        pass
