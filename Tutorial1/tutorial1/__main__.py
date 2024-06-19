import pyray as pr

from tutorial1.constants import APP_NAME, FRAME_RATE, WINDOW_HEIGHT, WINDOW_WIDTH
from tutorial1.scenes import SCENES


def main():
    pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
    pr.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME)
    pr.set_target_fps(FRAME_RATE)
    pr.hide_cursor()
    pr.init_audio_device()

    for i in range(5):
        pr.trace_log(
            pr.TraceLogLevel.LOG_INFO, f"GAMEPAD: id: {i} - {pr.get_gamepad_name(i)}"
        )

    scene = SCENES["title"]
    scene.reset()

    while not pr.window_should_close():
        next = scene.update(pr.get_frame_time())

        pr.begin_drawing()
        scene.draw()
        pr.end_drawing()

        if scene != SCENES[next]:
            scene = SCENES[next]
            scene.reset()

    pr.close_window()


if __name__ == "__main__":
    main()
