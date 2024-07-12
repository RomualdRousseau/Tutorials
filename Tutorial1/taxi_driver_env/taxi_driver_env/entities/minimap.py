import pyray as pr
from taxi_driver_env.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from taxi_driver_env.entities import car, world


class Minimap:
    def __init__(self, acar: car.Car) -> None:
        self.car = acar

    def get_bound(self) -> pr.Rectangle:
        return pr.Rectangle(WINDOW_WIDTH - 310, WINDOW_HEIGHT - 410, 300, 400)

    def reset(self) -> None:
        self.frame_buffer = pr.load_render_texture(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.camera = pr.Camera2D(
            pr.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
            pr.Vector2(0, 0),
            0,
            1,
        )

    def update(self, dt: float) -> None:
        # self.camera.target = pr.vector2_lerp(self.camera.target, self.car.curr_pos.to_vec(), 0.2)
        # self.camera.rotation = np.rad2deg(np.pi / 2 - np.arctan2(self.car.head[1], self.car.head[0]))
        pass

    def draw(self) -> None:
        pr.begin_texture_mode(self.frame_buffer)
        pr.clear_background(pr.WHITE)  # type: ignore
        pr.begin_mode_2d(self.camera)

        for bone in world.get_singleton().borders.skeleton:
            bone.draw(5, pr.GRAY)  # type: ignore

        for bone in self.car.corridor.skeleton:
            bone.draw(10, pr.BLUE)  # type: ignore

        self.car.curr_pos.draw(15, pr.YELLOW)  # type: ignore

        pr.end_mode_2d()
        pr.end_texture_mode()

        pr.draw_texture_pro(
            self.frame_buffer.texture,
            pr.Rectangle(0, 0, WINDOW_WIDTH, -WINDOW_HEIGHT),
            self.get_bound(),
            pr.Vector2(0, 0),
            0,
            pr.WHITE,  # type: ignore
        )
