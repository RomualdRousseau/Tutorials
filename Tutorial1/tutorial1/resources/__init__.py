from functools import cache
from importlib import resources as impresources

import pyray as pr

import tutorial1.resources as res

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


@cache
def load_texture(name: str) -> pr.Texture:
    return pr.load_texture(str(impresources.files(res) / RESOURCES[name]))


@cache
def load_sound(name: str) -> pr.Sound:
    return pr.load_sound(str(impresources.files(res) / RESOURCES[name]))
