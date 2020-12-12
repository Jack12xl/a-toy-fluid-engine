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
                dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.5)
            )
                for _ in range(7)]

        # Temperature
        self.T_tmp, self.d_T, self.d_T_tmp, self.d_T_prev, self.T_init, self.T_origin = \
            [CellGrid(
                ti.Vector.field(1, dtype=ti.f32, shape=cfg.res), cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.5)
            )
                for _ in range(6)]

        # Density
        self.rho_tmp, self.d_rho, self.d_rho_tmp, self.d_rho_prev, self.rho_init, self.rho_origin = \
            [CellGrid(
                ti.Vector.field(3, dtype=ti.f32, shape=cfg.res), cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.5)
            )
                for _ in range(6)]

        self.d_v, self.d_v_prev, self.d_v_tmp, self.d_v_proj, self.v_init, self.v_origin, self.v_presave, self.v_tmp = \
            [FaceGrid(ti.f32,
                      shape=cfg.res,
                      dim=cfg.dim,
                      dx=ts.vecND(self.dim, self.cfg.dx),
                      o=ts.vecND(self.dim, 0.5)
                      )
             for _ in range(8)]

        # Debug
        self.distortion, self.BM, self.FM= \
            [CellGrid(
                ti.Vector.field(3, dtype=ti.f32, shape=cfg.res),
                cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx),
                o=ts.vecND(self.dim, 0.5)
            )
                for _ in range(3)]


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
    def init_map(self, m: Wrapper):
        for I in ti.static(m):
            m[I] = m.getW(I)

    def reset(self):
        super(BimMocqGridData, self).reset()
        self.materialize()
        # init the map
        self.d_T.fill(ts.vecND(1, 0.0))
        self.d_T_tmp.fill(ts.vecND(1, 0.0))
        self.d_T_prev.fill(ts.vecND(1, 0.0))
        # self.T_init.fill(ts.vecND(1, 0.0))
        # self.T_origin.fill(ts.vecND(1, 0.0))

        self.d_rho.fill(ts.vecND(3, 0.0))
        self.d_rho_tmp.fill(ts.vecND(3, 0.0))
        self.d_rho_prev.fill(ts.vecND(3, 0.0))
        # self.rho_init.fill(ts.vecND(3, 0.0))
        # self.rho_origin.fill(ts.vecND(3, 0.0))

        self.d_v.fill(ts.vecND(self.dim, 0.0))
        self.d_v_prev.fill(ts.vecND(self.dim, 0.0))
        self.d_v_tmp.fill(ts.vecND(self.dim, 0.0))
        self.d_v_proj.fill(ts.vecND(self.dim, 0.0))
        # self.v_init.fill(ts.vecND(self.dim, 0.0))
        # self.v_origin.fill(ts.vecND(self.dim, 0.0))

        self.v_presave.fill(ts.vecND(self.dim, 0.0))
        self.v_tmp.fill(ts.vecND(self.dim, 0.0))

