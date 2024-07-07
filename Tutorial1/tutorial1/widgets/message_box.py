import pyray as pr

from tutorial1.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.widgets.button import Button


class MessageBox:

    FONT_SIZE = 20
    FONT_COLOR = pr.Color(216, 216, 216, 255)
    BORDER_SIZE = 10
    BUTTON_WIDTH = 100
    BUTTON_HEIGHT = 38
    BUTTON_COLOR = pr.Color(255, 125, 109, 192)
    FG_COLOR = pr.Color(38, 51, 59, 255)
    BG_COLOR = pr.Color(82, 96, 105, 216)

    def __init__(self, size: pr.Vector2, message: str, callback=None):
        self.size = size
        self.message = message
        self.callback = callback

    def get_bound(self) -> pr.Rectangle:
        return pr.Rectangle(
            (WINDOW_WIDTH - self.size.x) / 2,
            (WINDOW_HEIGHT - self.size.y) / 2,
            self.size.x,
            self.size.y,
        )

    def reset(self) -> None:
        self.button_ok = Button(
            self._compute_button_position(self.get_bound()),
            pr.Vector2(MessageBox.BUTTON_WIDTH, MessageBox.BUTTON_HEIGHT),
            "OK",
            MessageBox.BUTTON_COLOR,
            self._button_is_clicked,
        )

    def update(self, dt: float):
        self.button_ok.update(dt)

    def draw(self):
        bound = self.get_bound()
        pr.draw_rectangle_rec(bound, MessageBox.BG_COLOR)
        pr.draw_rectangle_lines_ex(bound, 2, MessageBox.FG_COLOR)
        pos = self._compute_text_position(bound)
        pr.draw_text(
            self.message,
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

    def _compute_text_position(self, bound: pr.Rectangle) -> pr.Vector2:
        hw = pr.measure_text(self.message, MessageBox.FONT_SIZE) / 2
        pos_x = int(bound.x + bound.width / 2 - hw - MessageBox.BORDER_SIZE)
        pos_y = int(
            bound.y
            + bound.height / 2
            - MessageBox.FONT_SIZE / 2
            - MessageBox.BUTTON_HEIGHT / 2
            - MessageBox.BORDER_SIZE
        )
        return pr.Vector2(pos_x, pos_y)

    def _button_is_clicked(self, button: Button):
        if button is self.button_ok and self.callback is not None:
            self.callback(self)
