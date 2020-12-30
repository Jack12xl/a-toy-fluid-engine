# implementation ref: Taichi sand & water
# Paper ref @17: Multi-species simulation of porous sand and water mixtures
# Paper ref @16:
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
        self.cfg = cfg
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
        # self.p_w_Jp = ti.field(dtype=Float)

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

    @ti.func
    def h_s(self, z):
        """
        @17 (9)
        :param z:
        :return:
        """
        ret = 0.0
        if z < 0:
            ret = 1.0
        elif z > 1:
            ret = 0.0
        else:
            ret = 1 - 10 * z ** 3 + 15.0 * z ** 4 - 6.0 * z ** 5
        return ret

    @ti.func
    def h(self, epsilon):
        """

        :param epsilon:
        :return:
        """
        pass

    @ti.kernel
    def P2G(self, dt: Float):
        """
        For Sand and water
        :param dt:
        :return:
        """
        # sand
        p_C = ti.static(self.p_C)
        p_v = ti.static(self.p_v)
        p_x = ti.static(self.p_x)
        g_m = ti.static(self.g_m)
        g_v = ti.static(self.g_v)
        g_f = ti.static(self.g_f)
        p_F = ti.static(self.p_F)

        for P in range(self.n_particle[None]):
            base = ti.floor(g_m.getG(p_x[P] - 0.5 * g_m.dx)).cast(Int)
            # TODO boundary condiiton
            fx = g_m.getG(p_x[P]) - base.cast(Float)

            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            U, sig, V = ti.svd(p_F[P])
            inv_sig = sig.inverse()

            epsilon = ti.Matrix(Float, self.dim, self.dim)
            for d in ti.static(range(self.dim)):
                epsilon[d, d] = ti.log(sig[d, d])

            # @16 (26)
            # Venant-kirchoff
            FCR = 2.0 * self.cfg.mu_0 * inv_sig @ epsilon + self.cfg.lambda_0 * epsilon.trace() @ inv_sig
            stress = U @ FCR @ V.transpose()
            # TODO where is stress
            affine = self.cfg.p_mass * p_C[P]

            for offset in ti.static(ti.grouped(self.stencil_range3())):
                dpos = g_m.getW(offset.cast(Float) - fx)

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                # @17 (15)
                # TODO ? why mass is in
                g_v[base + offset] += weight * (self.cfg.p_mass * p_v[P] + affine * dpos)
                g_m[base + offset] += weight * self.cfg.p_mass
                # TODO where did this come from
                g_f[base + offset] += weight * stress @ dpos

        # water
        p_w_x = ti.static(self.p_w_x)
        p_w_v = ti.static(self.p_w_v)
        p_w_C = ti.static(self.p_w_C)
        p_w_Jp = ti.static(self.p_Jp)
        g_w_v = ti.static(self.g_w_v)
        g_w_m = ti.static(self.g_w_m)
        g_w_f = ti.static(self.g_w_f)

        for P in range(self.n_w_particle[None]):
            base = ti.floor(g_w_m(p_w_x[P] - 0.5 * g_w_m.dx)).cast(Int)
            fx = g_w_m.getG(p_w_x[P]) - base.cast(Float)

            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            stress = self.cfg.w_k * (1.0 - 1.0 / p_w_Jp[P] ** self.cfg.w_gamma)
            # TODO ? also why not dt
            stress *= (- self.cfg.p_vol * 4 * self.cfg.inv_dx ** 2) * p_w_Jp[P]

            affine = self.cfg.p_mass * p_w_C[P]

            for offset in ti.static(ti.grouped(self.stencil_range3())):
                dpos = g_w_m.getW(offset.cast(Float) - fx)

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                g_w_v[base + offset] += weight * (self.cfg.p_mass * p_w_v[P] + affine @ dpos)
                g_w_m[base + offset] += weight * self.cfg.p_mass
                g_w_f[base + offset] += weight * stress * dpos