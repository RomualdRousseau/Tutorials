import pyray as pr

from tutorial1.scenes import SCENES
from tutorial1.util.types import Scene


def run_once(scene: Scene) -> Scene:
    next = scene.update(pr.get_frame_time())
    pr.begin_drawing()
    scene.draw()
    pr.end_drawing()
    return SCENES[next]


def main():
    pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
    pr.init_window(600, 600, "main")
    pr.set_target_fps(60)
    pr.hide_cursor()
    scene = SCENES["title"]
    while not pr.window_should_close():
        scene = run_once(scene)
    pr.close_window()


if __name__ == "__main__":
    main()
