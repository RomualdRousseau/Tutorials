from functools import cache
from importlib import resources as impresources
from typing import Callable

import pyray as pr

import taxi_driver_env.resources as res

RESOURCES = {
    "title": "screens/title.png",
    "loading": "screens/loading.png",
    "pickup": "screens/pickup.png",
    "dropoff": "screens/dropoff.png",
    "car": "sprites/car.png",
    "start": "sprites/start.png",
    "tree1": "sprites/tree1.png",
    "tree2": "sprites/tree2.png",
    "tree3": "sprites/tree3.png",
    "house1": "sprites/house1.png",
    "house2": "sprites/house2.png",
    "house3": "sprites/house3.png",
    "house4": "sprites/house4.png",
    "klaxon": "sounds/klaxon.ogg",
    "crash": "sounds/crash.ogg",
}

_gc: list[Callable[[], None]] = []


def clear_caches():
    [x() or True for x in _gc]
    load_texture.cache_clear()
    load_sound.cache_clear()


@cache
def load_texture(name: str) -> pr.Texture:
    texture = pr.load_texture(str(impresources.files(res) / RESOURCES[name]))
    _gc.append(lambda: pr.unload_texture(texture))
    return texture


@cache
def load_sound(name: str) -> pr.Sound:
    sound = pr.load_sound(str(impresources.files(res) / RESOURCES[name]))
    _gc.append(lambda: pr.unload_sound(sound))
    return sound
