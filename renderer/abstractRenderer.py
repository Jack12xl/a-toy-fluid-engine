from abc import ABCMeta, abstractmethod
from utils import Vector, Float, EuclideanDistance
import taichi as ti


@ti.data_oriented
class renderer(metaclass=ABCMeta):
    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid

        self.clr_bffr = ti.Vector.field(3, dtype=ti.float32, shape=cfg.screen_res)

    @abstractmethod
    def renderStep(self, ):
        pass
