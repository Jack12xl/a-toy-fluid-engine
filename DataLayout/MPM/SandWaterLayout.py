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
            # Prof.Jiang said he agreed with this...
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

    @ti.kernel
    def G_Normalize_plus_Gravity(self, dt: Float):
        g_s_m = ti.static(self.g_m)
        g_s_v = ti.static(self.g_v)

        g_w_m = ti.static(self.g_w_m)
        g_w_v = ti.static(self.g_w_v)

        for I in ti.grouped(g_s_m):
            # assume the double grid size is the same
            if g_s_m[I] > 0:
                g_s_v[I] /= g_s_m[I]
            if g_w_m[I] > 0:
                g_w_v[I] /= g_w_m[I]

    @ti.kernel
    def G_momentum_exchange(self, dt: Float):
        """
        @17 (13), 4.1,
        Different from @17
        we do not solve newton iteration here
        for simplicity
        :return:
        """
        g_s_m = ti.static(self.g_m)
        g_s_v = ti.static(self.g_v)
        g_s_f = ti.static(self.g_f)

        g_w_m = ti.static(self.g_w_m)
        g_w_v = ti.static(self.g_w_v)
        g_w_f = ti.static(self.g_w_f)

        cE = self.cfg.n ** 2 * self.cfg.p_rho * 9.8 / self.cfg.k_hat

        for I in ti.grouped(g_s_m):
            if g_s_m[I] > 0 and g_w_m[I] > 0:
                sm, wm = g_s_m[I], g_w_m[I]
                # TODO different from @17 (20) (21)
                d = cE * sm * wm
                # TODO @17 (21.5)
                M = ti.Matrix([[sm, 0.0], [0.0, wm]])
                # TODO still different from (20) (21)
                D = ti.Matrix([-d, d], [d, -d])
                V = ti.Matrix.rows([g_s_v[I], g_w_v[I]])
                # @17 (22
                G = ti.Matrix.rows([self.gravity, self.gravity])
                F = ti.Matrix.rows([g_s_f[I], g_w_f[I]])
                # directly solve Ax = B by inverse hahaha
                # Niubi.jpg
                A = M + dt * D
                B = M @ V + dt * (M @ G + F)
                # Get X_n_1
                X = A.inverse() @ B

                new_v = ts.vecND(self.dim, 0.0)
                for d in range(self.dim):
                    new_v[d] = X[0, d]
                g_s_v[I] = new_v

                for d in range(self.dim):
                    new_v[d] = X[1, d]
                g_w_v[I] = new_v

            elif g_s_m[I] > 0:
                g_s_v[I] += dt * (self.gravity + g_s_f[I] / g_s_m[I])
            elif g_w_m[I] > 0:
                g_w_v[I] += dt * (self.gravity + g_w_f[I] / g_w_m[I])

    @ti.kernel
    def G_boundary_condition(self):
        """
        @16 8.1 Friction
        boundary condition for both water and sand
        Assume we only have the surrounding cube boundary, with v == 0
        :return:
        """
        # water boundary
        g_w_m = ti.static(self.g_w_m)
        g_w_v = ti.static(self.g_w_v)
        g_s_m = ti.static(self.g_m)
        g_s_v = ti.static(self.g_v)

        for I in ti.static(g_w_m):
            # TODO Unbound
            # TODO vectorize
            for d in ti.static(range(self.dim)):
                if I[d] < self.cfg.g_padding[d] and g_w_v[I][d] < 0.0:
                    if ti.static(self.cfg.bdryCdtn) == BC.sticky:
                        g_w_v[I][d] = 0.0
                        g_s_v[I][d] = 0.0
                    elif ti.static(self.cfg.bdryCdtn) == BC.slip:
                        n = ts.vecND(self.dim, 0.0)
                        n[d] = 1.0
                        g_w_v[I] -= n * n.dot(g_w_v[I])
                        g_s_v[I] = self.calFriction(n, g_s_v[I])
                    else:
                        n = ts.vecND(self.dim, 0.0)
                        n[d] = 1.0
                        g_w_v[I] -= n * min(n.dot(g_w_v[I]), 0.0)
                        g_s_v[I] = self.calFriction(n, g_s_v[I])

                if I[d] > self.cfg.res[d] - self.cfg.g_padding[d] and g_w_v[I][d] > 0.0:
                    if self.cfg.bdryCdtn == BC.sticky:
                        g_w_v[I][d] = 0.0
                        g_s_v[I][d] = 0.0
                    elif ti.static(self.cfg.bdryCdtn) == BC.slip:
                        n = ts.vecND(self.dim, 0.0)
                        n[d] = -1.0
                        g_w_v[I] -= n * n.dot(g_w_v[I])
                        g_s_v[I] = self.calFriction(n, g_s_v[I])
                    else:
                        n = ts.vecND(self.dim, 0.0)
                        n[d] = -1.0
                        g_w_v[I] -= n * min(n.dot(g_w_v[I]), 0.0)
                        g_s_v[I] = self.calFriction(n, g_s_v[I])

    @ti.func
    def calFriction(self, normal, s_v):
        """
        Coulomb friction law
        @16 8.1
        :param normal:
        :param s_v:
        :param s:
        :return:
        """
        s = s_v.dot(normal)
        ret = s_v
        if s <= 0:
            v_normal = s * normal
            # slip velocity
            v_tangent = s_v - v_normal
            vt = v_tangent.norm()
            if vt > 1e-12:
                threshold = -self.cfg.mu_b * s
                if vt < threshold:
                    ret = v_tangent - vt * v_tangent / vt
                else:
                    ret = v_tangent - threshold * v_tangent / vt

        return ret

