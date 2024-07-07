import pyray as pr

# import tutorial1.resources as res


class Button:

    FONT_SIZE = 20
    FG_COLOR = pr.Color(38, 51, 59, 255)

    RED = pr.Color(252, 127, 104, 192)
    YELLOW = pr.Color(254, 201, 123, 192)
    BLUE = pr.Color(64, 191, 182, 192)

    def __init__(
        self,
        position: pr.Vector2,
        size: pr.Vector2,
        text: str,
        color: pr.Color,
        callback=None,
    ):
        self.position = position
        self.size = size
        self.text = text
        self.color_up = color
        self.color_dn = pr.Color(color.r // 2, color.g // 2, color.b // 2, 255)
        self.callback = callback

    def get_bound(self) -> pr.Rectangle:
        return pr.Rectangle(self.position.x, self.position.y, self.size.x, self.size.y)

    def is_playing(self, _: float = 0.1) -> bool:
        return True

    def reset(self) -> None:
        self.clicked = False
        self.action = False
        # self.sound = res.load_sound("button_click")
        # while pr.is_sound_ready(self.sound):
        #     pass

    def update(self, _: float):
        pos = pr.get_mouse_position()
        if pr.check_collision_point_rec(pos, self.get_bound()):
            if not self.clicked and pr.is_mouse_button_down(pr.MouseButton.MOUSE_BUTTON_LEFT):
                self.clicked = True
                self.action = False
            if self.clicked and pr.is_mouse_button_released(pr.MouseButton.MOUSE_BUTTON_LEFT):
                # pr.play_sound(self.sound)
                self.clicked = False
                self.action = True
            if self.action:
                if self.callback is not None:
                    self.callback(self)
                self.action = False
        else:
            self.clicked = False
            self.action = False

    def draw(self):
        if self.clicked:
            pr.draw_rectangle_rounded(self.get_bound(), 0.5, 4, self.color_dn)
        else:
            pr.draw_rectangle_rounded(self.get_bound(), 0.5, 4, self.color_up)
        pr.draw_rectangle_rounded_lines(self.get_bound(), 0.5, 4, 4, Button.FG_COLOR)

        hw = pr.measure_text(self.text, Button.FONT_SIZE) / 2
        px = int(self.get_bound().x + self.get_bound().width / 2 - hw)
        py = int(self.get_bound().y + self.get_bound().height / 2 - Button.FONT_SIZE / 2)
        pr.draw_text(self.text, px, py, Button.FONT_SIZE, Button.FG_COLOR)
