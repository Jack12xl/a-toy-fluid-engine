# implementation ref: Taichi sand & water
# Paper ref @17: Multi-species simulation of porous sand and water mixtures
# Paper ref @16:
import numpy as np
from math import radians
import taichi as ti
import taichi_glsl as ts
from config.CFG_wrapper import TwinGridmpmCFG, DLYmethod, MaType, BC
from utils import Int, Float, Matrix, Vector
from Grid import CellGrid
from .dataLayout import mpmLayout


@ti.data_oriented
class TwinGridLayout(mpmLayout):
    """
    Featured double grid
    Especially designed for coupling of Sand 
    and water
    """

    def __init__(self, cfg: TwinGridmpmCFG):
        # We assign super class property belong to sand
        super(TwinGridLayout, self).__init__(cfg)
        self.cfg = cfg
        # except Jp(belongs to water)

        # sand part
        # particle
        self.p_phi = ti.field(dtype=Float)  # water saturation
        self.c_C0 = ti.field(dtype=Float)  # sand cohesion
        # volume correction scalar, which tracks changes of volume gained during extension
        self.vc_s = ti.field(dtype=Float)
        self.alpha_s = ti.field(dtype=Float)
        self.q_s = ti.field(dtype=Float)

        self.g_f = CellGrid(
            ti.Vector.field(self.dim, dtype=Float),
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
        self.p_w_color = ti.field(dtype=Int)
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
            ti.Vector.field(self.dim, dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )

    def materialize(self):
        super(TwinGridLayout, self).materialize()
        self._particle.place(self.p_phi,
                             self.c_C0,
                             self.vc_s,
                             self.alpha_s,
                             self.q_s)

        self._grid.place(self.g_f.field)

        self._w_particle = ti.root.dense(ti.i, self.max_n_w_particle)
        self._w_particle.place(self.p_w_x,
                               self.p_w_v,
                               self.p_w_C,
                               self.p_w_color)

        self._w_grid = ti.root.dense(self._indices, self.cfg.res)
        self._w_grid.place(self.g_w_v.field)
        self._w_grid.place(self.g_w_m.field)
        self._w_grid.place(self.g_w_f.field)

    def G2zero(self):
        super(TwinGridLayout, self).G2zero()
        self.g_f.fill(0.0)
        self.g_w_f.fill(0.0)

        self.g_w_m.fill(0.0)
        # self.g_m.fill(0.0)

        self.g_w_v.fill(ts.vecND(self.dim, 0.0))
        # self.g_v.fill(ts.vecND(self.dim, 0.0))

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

    def P2G(self, dt: Float):
        self.P2G_sand(dt)
        self.P2G_liquid(dt)

    @ti.kernel
    def P2G_sand(self, dt: Float):
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

            epsilon = ti.Matrix.zero(Float, self.dim, self.dim)
            for d in ti.static(range(self.dim)):
                epsilon[d, d] = ti.log(sig[d, d])

            # @16 (26)
            # Venant-kirchoff
            FCR = 2.0 * self.cfg.mu_0 * inv_sig @ epsilon + self.cfg.lambda_0 * epsilon.trace() * inv_sig
            # energy density
            stress = U @ FCR @ V.transpose()
            stress = (-self.cfg.p_vol * 4 * self.cfg.inv_dx ** 2) * stress @ p_F[P].transpose()
            # TODO where is stress
            affine = self.cfg.p_mass * p_C[P]
            for offset in ti.static(ti.grouped(self.stencil_range3())):
                dpos = g_m.getW(offset.cast(Float) - fx)

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                # @17 (15)
                # TODO ? why mass is in
                g_v[base + offset] += weight * (self.cfg.p_mass * p_v[P] + affine @ dpos)
                g_m[base + offset] += weight * self.cfg.p_mass
                # TODO where did this come from
                g_f[base + offset] += weight * stress @ dpos

    @ti.kernel
    def P2G_liquid(self, dt: Float):
        """

        :param dt:
        :return:
        """
        # water
        p_w_x = ti.static(self.p_w_x)
        p_w_v = ti.static(self.p_w_v)
        p_w_C = ti.static(self.p_w_C)
        p_w_Jp = ti.static(self.p_Jp)
        g_w_v = ti.static(self.g_w_v)
        g_w_m = ti.static(self.g_w_m)
        g_w_f = ti.static(self.g_w_f)

        for P in range(self.n_w_particle[None]):
            base = ti.floor(g_w_m.getG(p_w_x[P] - 0.5 * g_w_m.dx)).cast(Int)
            fx = g_w_m.getG(p_w_x[P]) - base.cast(Float)
            # print("P", P, "base:", base, "fx: ", fx)
            # ti.sync()

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

        cE = self.cfg.n ** 2 * self.cfg.p_rho * self.gravity[1] / self.cfg.k_hat

        for I in ti.grouped(g_s_m):
            if g_s_m[I] > 0 and g_w_m[I] > 0:
                sm, wm = g_s_m[I], g_w_m[I]
                sv, wv = g_s_v[I], g_w_v[I]
                # TODO different from @17 (20) (21)
                d_ij = cE * sm * wm
                # TODO @17 (21.5)
                M = ti.Matrix([[sm, 0.0], [0.0, wm]])
                # TODO still different from (20) (21)
                D = ti.Matrix([[-d_ij, d_ij], [d_ij, -d_ij]])
                V = ti.Matrix.rows([sv, wv])
                # @17 (22
                G = ti.Matrix.rows([self.gravity, self.gravity])
                F = ti.Matrix.rows([g_s_f[I], g_w_f[I]])
                # directly solve Ax = B by inverse hahaha
                # Bu kui shi ni.jpg
                A = M + dt * D
                B = M @ V + dt * (M @ G + F)
                # Get X_n_1
                X = A.inverse() @ B

                new_v = ts.vecND(self.dim, 0.0)
                for d in ti.static(range(self.dim)):
                    new_v[d] = X[0, d]
                g_s_v[I] = new_v

                for d in ti.static(range(self.dim)):
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
        TODO still different from origin paper...
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

    @ti.func
    def sand_projection(self, epsilon, P):
        """
        Drucker-Prager projection
        :param epsilon:
        :param P:
        :return:
        """
        # TODO where did this volume correction come from
        e = epsilon + self.vc_s[P] / self.dim * ti.Matrix.identity(Float, self.dim)
        # Cohesion
        e += self.c_C0[P] * (1.0 - self.p_phi[P]) / (self.dim * self.alpha_s[P]) * ti.Matrix.identity(Float, self.dim)
        ehat = e - e.trace() / self.dim * ti.Matrix.identity(Float, self.dim)

        Fnorm = 0.0
        for d in ti.static(range(self.dim)):
            Fnorm += ehat[d, d] ** 2
        Fnorm = ti.sqrt(Fnorm)
        # @16 (27) amout of plastic deformation
        d_gamma = Fnorm + (self.dim * self.cfg.lambda_0 / (2.0 * self.cfg.mu_0) + 1.0) * e.trace() * self.alpha_s[P]

        ret_e = ti.Matrix.zero(Float, self.dim, self.dim)
        # TODO what's q...
        ret_q = 0.0
        # TODO different from @16 (27)
        if Fnorm <= 0 or e.trace() > 0.0:  # Case II
            ret_e = ti.Matrix.zero(Float, self.dim, self.dim)
            # @16 7.3 Hardening
            for d in ti.static(range(self.dim)):
                ret_q += e[d, d] ** 2
            ret_q = ti.sqrt(ret_q)
        elif d_gamma <= 0:  # Case I
            ret_e = epsilon
            ret_q = 0.0
        else:  # Case III
            # @16 (28)
            ret_e = e - d_gamma * ehat / Fnorm
            ret_q = d_gamma

        ret_sig = ti.Matrix.zero(Float, self.dim, self.dim)
        for d in ti.static(range(self.dim)):
            ret_sig[d, d] = ti.exp(ret_e[d, d])

        return ret_sig, ret_q

    @ti.func
    def hardening(self, dq, P):
        """
        @16 7.3
        We adopt the hardening model of Mast et al[2014]
        plastic deformation can increase the friction between sand particles
        :param P:
        :param dq: delta hardening state
        :return:
        """
        # (29)
        q_n_1 = self.q_s[P]
        q_n_1 += dq
        # (30) friction angle
        phi = radians(self.cfg.h0 + (self.cfg.h1 * q_n_1 - self.cfg.h3) * ti.exp(-self.cfg.h2 * q_n_1))
        sin_phi = ti.sin(phi)

        self.q_s[P] = q_n_1
        self.alpha_s[P] = ti.sqrt(2.0 / 3.0) * (2 * sin_phi) / (3 - sin_phi)

    @ti.kernel
    def G2P(self, dt: Float):
        """

        :param dt:
        :return:
        """
        # water
        p_w_C = ti.static(self.p_w_C)
        p_w_x = ti.static(self.p_w_x)
        p_w_v = ti.static(self.p_w_v)
        p_w_Jp = ti.static(self.p_Jp)
        g_w_m = ti.static(self.g_w_m)
        g_w_v = ti.static(self.g_w_v)

        for P in range(self.n_w_particle[None]):
            base = ti.floor(g_w_m.getG(P - 0.5 * g_w_m.dx)).cast(Int)
            # TODO boundary
            fx = g_w_m.getG(p_w_x[P]) - base.cast(Float)

            w = [
                0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
            ]

            new_v = ti.Vector.zero(Float, self.dim)
            new_C = ti.Matrix.zero(Float, self.dim, self.dim)

            for offset in ti.static(ti.grouped(self.stencil_range3())):
                dpos = offset.cast(Float) - fx
                v = g_w_v[base + offset]

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                new_v += weight * v
                new_C += 4 * self.cfg.inv_dx * weight * v.outer_product(dpos)
            # @17 4.3.1 update F^{S}
            p_w_Jp[P] = (1.0 + dt * new_C.trace()) * p_w_Jp[P]
            p_w_v[P], p_w_C[P] = new_v, new_C

        # sand
        g_s_m = ti.static(self.g_m)
        g_s_v = ti.static(self.g_v)
        p_s_v = ti.static(self.p_v)
        p_s_x = ti.static(self.p_x)
        p_s_phi = ti.static(self.p_phi)
        p_s_F = ti.static(self.p_F)
        p_s_C = ti.static(self.p_C)

        for P in range(self.n_particle[None]):
            base = ti.floor(g_s_m.getG(p_s_x[P] - 0.5 * g_s_m.dx)).cast(Int)
            fx = g_s_m.getG(p_s_x[P]) - base.cast(Float)

            new_v = ti.Vector.zero(Float, self.dim)
            new_C = ti.Matrix.zero(Float, self.dim, self.dim)

            w = [
                0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
            ]
            # @17 4.3.3
            p_s_phi[P] = 0.0  # clear it for weight sum
            for offset in ti.static(ti.grouped(self.stencil_range3())):
                dpos = offset.cast(Float) - fx
                v = g_s_v[base + offset]

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                new_v += weight * v
                new_C += 4 * self.cfg.inv_dx * weight * v.outer_product(dpos)
                if g_s_m[base + offset] > 0 and g_w_m[base + offset] > 0:
                    # @17 4.3.3 (24) Saturation based cohesion
                    p_s_phi[P] += weight * 1.0

            p_s_F[P] = (ti.Matrix.identity(Float, self.dim) + dt * new_C) @ p_s_F[P]
            p_s_v[P], p_s_C[P] = new_v, new_C
            p_s_x[P] += dt * p_s_v[P]
            # (27.5)
            U, sig, V = ti.svd(p_s_F[P])
            epsilon_hat = ti.Matrix.identity(Float, self.dim, self.dim)
            for d in range(self.dim):
                epsilon_hat[d, d] = sig[d, d]

            new_sig, dq = self.sand_projection(epsilon_hat, P)
            self.hardening(dq, P)

            # @16 (27 ~ 28)
            new_F = U @ new_sig @ V.transpose()
            # @17 (26)
            self.vc_s[P] += -ti.log(new_F.determinant()) + ti.log(p_s_F[P].determinant())
            p_s_F[P] = new_F

    def add_sand_cube(self,
                      l_b: Vector,
                      cube_size: Vector,
                      n_p: Int,
                      velocity: Vector,
                      color=0xFFFFFF,
                      ):
        assert self.n_particle[None] + n_p <= self.max_n_particle
        self.source_bound[0] = l_b
        self.source_bound[1] = cube_size
        self.source_velocity[None] = velocity
        self.seed_sand(n_p, int(color))

        self.n_particle[None] += n_p

    def add_liquid_cube(self,
                        l_b: Vector,
                        cube_size: Vector,
                        n_p: Int,
                        velocity: Vector,
                        color=0xFFFFFF,
                        ):
        assert self.n_w_particle[None] + n_p <= self.max_n_w_particle
        self.source_bound[0] = l_b
        self.source_bound[1] = cube_size
        self.source_velocity[None] = velocity
        self.seed_liquid(n_p, int(color))

        self.n_w_particle[None] += n_p
        print("now we have {} water ps".format(self.n_w_particle[None]))

    # @ti.kernel
    # def seed(self,
    #          n_p: Int,
    #          mat: Int,
    #          color: Int):
    #     cur_n_p = 0
    #     if mat == MaType.sand:
    #         cur_n_p = self.n_particle[None]
    #     else:  # water
    #         cur_n_p = self.n_w_particle[None]
    #
    #     for P in range(cur_n_p,
    #                    cur_n_p + n_p):
    #         x = self.source_bound[0] + ts.randND(self.dim) * self.source_bound[1]
    #         if mat == MaType.sand:
    #             self.seed_sand_particle(P, x, color, self.source_velocity[None])
    #         else:
    #             self.seed_liquid_particle(P, x, color, self.source_velocity[None])

    @ti.kernel
    def seed_sand(self,
                  n_p: Int,
                  color: Int):
        for P in range(self.n_particle[None],
                       self.n_particle[None] + n_p):
            x = self.source_bound[0] + ts.randND(self.dim) * self.source_bound[1]
            self.seed_sand_particle(P, x, color, self.source_velocity[None])

    @ti.kernel
    def seed_liquid(self,
                    n_p: Int,
                    color: Int):
        for P in range(self.n_w_particle[None],
                       self.n_w_particle[None] + n_p):
            x = self.source_bound[0] + ts.randND(self.dim) * self.source_bound[1]
            self.seed_liquid_particle(P, x, color, self.source_velocity[None])

    @ti.func
    def seed_sand_particle(self, P, x, color, velocity):
        self.p_x[P] = x
        self.p_v[P] = velocity
        self.p_F[P] = ti.Matrix.identity(Float, self.dim)

        # what the hell...
        self.c_C0[P] = -0.01
        self.alpha_s[P] = 0.267765

        self.p_color[P] = color
        self.p_C[P] = ti.Matrix.zero(Float, self.dim, self.dim)

    @ti.func
    def seed_liquid_particle(self, P, x, color, velocity):
        self.p_w_x[P] = x
        self.p_w_v[P] = velocity
        self.p_Jp[P] = 1.0

        r = ts.rand()
        if r <= 0.85:
            self.p_w_color[P] = self.cfg.sand_yellow
        elif r <= 0.95:
            self.p_w_color[P] = self.cfg.sand_brown
        else:
            self.p_w_color[P] = self.cfg.sand_white

        self.p_w_C[P] = ti.Matrix.zero(Float, self.dim, self.dim)

    # @ti.func
    # def seed_particle(self, P, x, mat, color, velocity):
    #     # self.p_material_id[P] = mat
    #     if mat == MaType.sand:
    #         self.p_x[P] = x
    #         self.p_v[P] = velocity
    #         self.p_F[P] = ti.Matrix.identity(Float, self.dim)
    #
    #         # what the hell...
    #         self.c_C0[P] = -0.01
    #         self.alpha_s[P] = 0.267765
    #
    #         self.p_color[P] = color
    #         self.p_C[P] = ti.Matrix.zero(Float, self.dim, self.dim)
    #     else:
    #         self.p_w_x[P] = x
    #         self.p_w_v[P] = velocity
    #         self.p_Jp[P] = 1.0
    #
    #         r = ts.rand()
    #         if r <= 0.85:
    #             self.p_w_color[P] = self.cfg.sand_yellow
    #         elif r <= 0.95:
    #             self.p_w_color[P] = self.cfg.sand_brown
    #         else:
    #             self.p_w_color[P] = self.cfg.sand_white
    #
    #         self.p_w_C[P] = ti.Matrix.zero(Float, self.dim, self.dim)

    @ti.kernel
    def update_liquid_color(self):
        for P in range(self.n_w_particle[None]):
            self.p_w_color[P] = self.color_lerp(0.2, 0.231, 0.792, 0.867, 0.886, 0.886, self.p_w_v[P].norm() / 5.0)
