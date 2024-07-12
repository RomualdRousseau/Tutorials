from typing import Callable, Optional

import pyray as pr
import taxi_driver_env.resources as res
from taxi_driver_env.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from taxi_driver_env.utils.types import Widget
from taxi_driver_env.widgets.button import Button


class MessageBox:
    FONT_SIZE = 20  # pixel
    FONT_COLOR = pr.Color(216, 216, 216, 255)
    BORDER_SIZE = 10  # pixel
    BUTTON_WIDTH = 100  # pixel
    BUTTON_HEIGHT = 38  # pixel
    BUTTON_COLOR = pr.Color(255, 125, 109, 192)
    FG_COLOR = pr.Color(38, 51, 59, 255)
    BG_COLOR = pr.Color(82, 96, 105, 216)
    TITLE_COLOR = pr.Color(192, 192, 192, 255)
    ICON_BORDER = 10  # pixel
    TYPEWRITER_SPEED = 20  # char.s-1

    def __init__(
        self,
        size: pr.Vector2,
        message: str,
        title: Optional[str] = None,
        icon: Optional[str] = None,
        callback: Optional[Callable[[Widget], None]] = None,
    ):
        self.size = size
        self.title = title
        self.icon = icon
        self.message = message
        self.callback = callback
        self.button_ok = Button(
            self._compute_button_position(self.get_bound()),
            pr.Vector2(MessageBox.BUTTON_WIDTH, MessageBox.BUTTON_HEIGHT),
            "OK",
            MessageBox.BUTTON_COLOR,
            self._button_is_clicked,
        )
        self.timer: float = 0

    def get_bound(self) -> pr.Rectangle:
        return pr.Rectangle(
            (WINDOW_WIDTH - self.size.x) / 2,
            (WINDOW_HEIGHT - self.size.y) / 2,
            self.size.x,
            self.size.y,
        )

    def reset(self) -> None:
        self.button_ok.reset()
        self.timer = 0

    def update(self, dt: float):
        self.button_ok.update(dt)
        self.timer += MessageBox.TYPEWRITER_SPEED * dt

    def draw(self):
        bound = self.get_bound()

        pr.draw_rectangle_rec(bound, MessageBox.BG_COLOR)
        pr.draw_rectangle_lines_ex(bound, 2, MessageBox.FG_COLOR)

        if self.icon is not None:
            tex = res.load_texture(self.icon)
            pr.draw_texture_pro(
                tex,
                pr.Rectangle(0, 0, tex.width, tex.height),
                self._compute_icon_position(bound),
                pr.Vector2(0, 0),
                0,
                pr.WHITE,  # type: ignore
            )

        if self.title is not None:
            pos = self._compute_title_position(bound)
            pr.draw_text(
                self.title,
                int(pos.x),
                int(pos.y),
                MessageBox.FONT_SIZE,
                MessageBox.TITLE_COLOR,
            )

        pos = self._compute_text_position(bound)
        mark = min(int(self.timer) + 1, len(self.message))
        pr.draw_text(
            self.message[:mark],
            int(pos.x),
            int(pos.y),
            MessageBox.FONT_SIZE,
            MessageBox.FONT_COLOR,
        )

        self.button_ok.draw()

    def _compute_button_position(self, bound: pr.Rectangle) -> pr.Vector2:
        pos_x = bound.x + bound.width - MessageBox.BORDER_SIZE - MessageBox.BUTTON_WIDTH
        pos_y = bound.y + bound.height - MessageBox.BORDER_SIZE - MessageBox.BUTTON_HEIGHT
        return pr.Vector2(pos_x, pos_y)

    def _compute_title_position(self, bound: pr.Rectangle) -> pr.Vector2:
        if self.icon is not None:
            pos_x = int(bound.x + bound.height)
            pos_y = int(bound.y + MessageBox.BORDER_SIZE)
        else:
            pos_x = int(bound.x + MessageBox.BORDER_SIZE)
            pos_y = int(bound.y + MessageBox.BORDER_SIZE)
        return pr.Vector2(pos_x, pos_y)

    def _compute_text_position(self, bound: pr.Rectangle) -> pr.Vector2:
        if self.icon is not None:
            pos_x = int(bound.x + bound.height)
            pos_y = int(bound.y + MessageBox.BORDER_SIZE)
        else:
            hw = pr.measure_text(self.message, MessageBox.FONT_SIZE) / 2
            pos_x = int(bound.x + bound.width / 2 - hw - MessageBox.BORDER_SIZE)
            pos_y = int(
                bound.y
                + bound.height / 2
                - MessageBox.FONT_SIZE / 2
                - MessageBox.BUTTON_HEIGHT / 2
                - MessageBox.BORDER_SIZE
            )
        if self.title is not None:
            pos_y += MessageBox.FONT_SIZE
        return pr.Vector2(pos_x, pos_y)

    def _compute_icon_position(self, bound: pr.Rectangle) -> pr.Rectangle:
        pos_x = int(bound.x + MessageBox.ICON_BORDER)
        pos_y = int(bound.y + MessageBox.ICON_BORDER)
        size = int(bound.height - MessageBox.ICON_BORDER * 2)
        return pr.Rectangle(pos_x, pos_y, size, size)

    def _button_is_clicked(self, button: Button):
        if button is self.button_ok and self.callback is not None:
            self.callback(self)
