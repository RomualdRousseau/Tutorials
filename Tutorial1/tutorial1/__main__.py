import pyray as pr

import tutorial1.util.pyray_ex as prx
from tutorial1.constants import APP_NAME, FRAME_RATE, WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.scenes import first_scene, next_scene


def main():
    pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
    pr.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME)
    pr.set_target_fps(FRAME_RATE)
    pr.hide_cursor()
    pr.init_audio_device()
    prx.init_gamepad()

    scene = first_scene("title")

    while not pr.window_should_close():
        next = scene.update(pr.get_frame_time())

        pr.begin_drawing()
        scene.draw()
        pr.end_drawing()

        scene = next_scene(scene, next)

    pr.close_window()


if __name__ == "__main__":
    main()
