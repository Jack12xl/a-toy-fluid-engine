from abc import ABCMeta , abstractmethod
from utils import Vector, Float, EuclideanDistance
import taichi as ti

@ti.data_oriented
class renderer(ABCMeta):
    pass