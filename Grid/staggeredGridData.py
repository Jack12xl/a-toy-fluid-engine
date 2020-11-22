import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix
from .CellGrid import CellGrid
from .FaceGrid import FaceGrid
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
        # the distance between two neighbour when calculating divergence, vorticity
        self.inv_d = 1.0 / self.cfg.dx

        self.v = FaceGrid(ti.f32,
                          shape=cfg.res,
                          dim=cfg.dim,
                          dx=ts.vecND(self.dim, self.dx),
                          o=ts.vecND(self.dim, 0.5)
                          )
        self.new_v = FaceGrid(ti.f32,
                              shape=cfg.res,
                              dim=cfg.dim,
                              dx=ts.vecND(self.dim, self.dx),
                              o=ts.vecND(self.dim, 0.5)
                              )
        # buffer for advection-reflection
        self.tmp_v = FaceGrid(ti.f32,
                              shape=cfg.res,
                              dim=cfg.dim,
                              dx=ts.vecND(self.dim, self.dx),
                              o=ts.vecND(self.dim, 0.5)
                              )

        self.v_pairs = []
        for d in range(self.dim):
            self.v_pairs.append()

