from functools import cache
from importlib import resources as impresources

import pyray as pr

import tutorial1.resources as res

RESOURCES = {
    "car": "sprites/car.png",
    "tree": "sprites/tree.png",
    "house1": "sprites/house1.png",
    "house2": "sprites/house2.png",
    "house3": "sprites/house3.png",
    "klaxon": "sounds/klaxon.ogg",
    "crash": "sounds/crash.ogg",
}


@cache
def load_texture(name: str) -> pr.Texture:
    return pr.load_texture(str(impresources.files(res) / RESOURCES[name]))


@cache
def load_sound(name: str) -> pr.Sound:
    return pr.load_sound(str(impresources.files(res) / RESOURCES[name]))
