# implementation ref: Taichi sand & water
# Paper ref: Multi-species simulation of porous sand and water mixtures
import numpy as np
import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import TwoGridmpmCFG, DLYmethod, MaType, BC
from utils import Int, Float, Matrix, Vector
from Grid import CellGrid
from .dataLayout import mpmLayout


@ti.data_oriented
class DGridLayout(mpmLayout):
    """
    Featured double grid
    Especially designed for coupling of Sand 
    and water
    """

    def __init__(self, cfg: TwoGridmpmCFG):
        # We assign super class property belong to sand
        super(DGridLayout, self).__init__(cfg)
        # except Jp(belongs to water)

        # sand part
        self.g_f = CellGrid(
            ti.field(dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )

        # water part
        self.max_n_w_particle = cfg.max_n_w_particle
        self.n_w_particle = ti.field(dtype=Int, shape=())
        # particle
        self.p_w_x = ti.Vector.field(self.dim, dtype=Float)
        self.p_w_v = ti.Vector.field(self.dim, dtype=Float)
        self.p_w_C = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        self.p_w_Jp = ti.field(dtype=Float)

        # Grid
        # velocity
        self.g_w_v = CellGrid(
            ti.Vector.field(self.dim, dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )
        # mass
        self.g_w_m = CellGrid(
            ti.field(dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )

        self.g_w_f = CellGrid(
            ti.field(dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )

    def materialize(self):
        super(DGridLayout, self).materialize()
        self._grid.place(self.g_f.field)

        self._w_particle = ti.root.dense(ti.i, self.max_n_w_particle)
        self._w_particle.place(self.p_w_x,
                               self.p_w_v,
                               self.p_w_C,
                               self.p_w_Jp)

        self._w_grid = ti.root.dense(self._indices, self.cfg.res)
        self._w_grid.place(self.g_w_v.field)
        self._w_grid.place(self.g_w_m.field)
        self._w_grid.place(self.g_w_f.field)

    def G2zero(self):
        super(DGridLayout, self).G2zero()
        self.g_f.fill(ts.vecND(self.dim, 0.0))

        self.g_w_m.fill(0.0)
        self.g_v.fill(ts.vecND(self.dim, 0.0))
        self.g_w_f.fill(ts.vecND(self.dim, 0.0))

    @ti.kernel
    def P2G(self, dt: Float):
        """
        For Sand and water
        :param dt:
        :return:
        """
        pass
