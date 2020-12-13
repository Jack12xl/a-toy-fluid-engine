from abc import ABCMeta, abstractmethod
import taichi as ti
from config import PixelType, VisualizeEnum

@ti.data_oriented
class renderer(metaclass=ABCMeta):
    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid

        self.clr_bffr = ti.Vector.field(3, dtype=ti.float32, shape=cfg.screen_res)

    @abstractmethod
    def renderStep(self, bdrySolver):
        pass

    @abstractmethod
    def render_frame(self, render_what: VisualizeEnum = None):
        pass
