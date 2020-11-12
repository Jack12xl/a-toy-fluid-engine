import taichi as ti
import taichi_glsl as ts
from .abstractRenderer import renderer
from config import PixelType, VisualizeEnum
from utils import cmapper

@ti.data_oriented
class renderer25D(renderer):
    def __init__(self, cfg, grid):
        """
        Simply visualize a scarf
        :param cfg:
        :param grid:
        """
        super(renderer25D, self).__init__(cfg, grid)
        self.mapper = cmapper()

    @ti.kernel
    def render_collider(self, bdrySolver: ti.template()):
        pass
