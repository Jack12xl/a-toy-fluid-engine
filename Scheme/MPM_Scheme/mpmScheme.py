import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod


@ti.data_oriented
class mpmScheme(metaclass=ABCMeta):
    def __init__(self, cfg):
        pass