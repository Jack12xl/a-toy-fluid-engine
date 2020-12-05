import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix, MultiBufferPair, Wrapper
from .FaceGrid import FaceGrid, CellGrid
from config import SimulateType
from .staggeredGridData import MacGridData


@ti.data_oriented
class BimMocqGridData(MacGridData):
    def __init__(self, cfg):
        super(BimMocqGridData, self).__init__(cfg)

        self.forward_map, self.backward_map, self.backward_map_bffr, \
        self.forward_scalar_map, self.backward_scalar_map, self.backward_scalar_map_bffr, \
        self.tmp_map = \
            [CellGrid(
                ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res), cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0)
            )
                for _ in range(7)]

        self.d_T, self.d_T_tmp, self.drho, self.drho_tmp = \
            [CellGrid(
                ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res), cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0)
            )
                for _ in range(4)]

        self.dv, self.dv_tmp =



    def materialize(self):
        super(BimMocqGridData, self).materialize()
        # init the forward and backward, t == 0 should map to itself
        self.init_map(self.forward_map)
        self.init_map(self.forward_scalar_map)

        self.init_map(self.backward_map)
        self.init_map(self.backward_map_bffr)
        self.init_map(self.backward_scalar_map)
        self.init_map(self.backward_scalar_map_bffr)

    @ti.kernel
    def init_map(self, m: Matrix):
        for I in ti.static(m):
            m[I] = m.getW(I)
