import pyray as pr
from functools import cache

RESOURCES = {
    "car": "resources/car.png",
    "tree": "resources/tree.png",
    "house1": "resources/house1.png",
    "house2": "resources/house2.png",
    "house3": "resources/house3.png",
}


@cache
def load_texture(name: str) -> pr.Texture:
    return pr.load_texture(RESOURCES[name])
