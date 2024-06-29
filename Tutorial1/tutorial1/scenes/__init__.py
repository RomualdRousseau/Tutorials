from tutorial1.scenes import gameloop, loading, title, trainer
from tutorial1.util.types import Scene

SCENES = {"title": title, "loading": loading, "gameloop": gameloop, "trainer": trainer}


def first_scene(next: str) -> Scene:
    scene = SCENES[next]
    scene.reset()
    return scene


def next_scene(scene: Scene, next: str) -> Scene:
    if scene != SCENES[next]:
        scene = SCENES[next]
        scene.reset()
    return scene
