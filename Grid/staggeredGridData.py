import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix
from .DataGrid import DataGrid
from config import SimulateType
from .FluidGridData import FluidGridData


@ti.data_oriented
class MacGridData(FluidGridData):
    """
    staggered grid
    vector on the grid face
    """

    def __init__(self, cfg):
        super(MacGridData, self).__init__(cfg)

        self.inv_d = 1.0 / self.cfg.dx