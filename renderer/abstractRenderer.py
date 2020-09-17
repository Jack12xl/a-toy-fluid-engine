from abc import ABCMeta , abstractmethod
from utils import Vector, Float, EuclideanDistance
import taichi as ti

@ti.data_oriented
class renderer(metaclass=ABCMeta):
    def __init__(self, cfg, ):
        self.cfg = cfg

    @abstractmethod
    def step(self, ):
        pass