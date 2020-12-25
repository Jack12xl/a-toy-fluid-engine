import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import mpmCFG, DLYmethod, MaType
from utils import Int, Float, Matrix, Vector
from Grid import CellGrid


@ti.data_oriented
class mpmLayout(metaclass=ABCMeta):
    """
    the default data for MPM,
    Support fix size particle only
    """

    def __init__(self, cfg: mpmCFG):
        self.cfg = cfg

        self.dim = cfg.dim
        self.n_particles = ti.field(dtype=Int, shape=())

        self.gravity = ts.vecND(self.dim, 0.0)
        # TODO
        self.gravity[1] = -9.8
        # Particle

        # position
        self.p_x = ti.Vector.field(self.dim, dtype=Float)
        # velocity
        self.p_v = ti.Vector.field(self.dim, dtype=Float)
        # affine velocity field
        self.p_C = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        # deformation gradient
        self.p_F = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        # material
        self.p_material_id = ti.field(dtype=Int)
        self.p_color = ti.field(dtype=Int)
        # plastic deformation volume ratio
        self.p_Jp = ti.field(dtype=Float)

        # Grid
        # velocity
        self.g_v = CellGrid(
            ti.Vector.field(self.dim, dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )
        # mass
        self.g_m = CellGrid(
            ti.field(dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )

        self._particle = None
        self._grid = None

    def materialize(self):
        self._particle = ti.root.dense(ti.i, self.cfg.n_particle)
        _indices = ti.ij if self.dim == 2 else ti.ijk
        if ti.static(self.cfg.layout_method) == DLYmethod.SoA:
            self._particle.place(self.p_x)
            self._particle.place(self.p_v)
            self._particle.place(self.p_C)
            self._particle.place(self.p_F)
            self._particle.place(self.p_material_id)
            self._particle.place(self.p_color)
            self._particle.place(self.p_Jp)

            self._grid = ti.root.dense(_indices, self.cfg.res)

            self._grid.place(self.g_v.field)
            self._grid.place(self.g_m.field)
        elif ti.static(self.cfg.layout_method) == DLYmethod.AoS:

            self._particle.place(self.p_x,
                                 self.p_v,
                                 self.p_C,
                                 self.p_F,
                                 self.p_material_id,
                                 self.p_color,
                                 self.p_Jp)
            self._grid = ti.root.dense(_indices, self.cfg.res)
            self._grid.place(self.g_v.field)
            self._grid.place(self.g_m.field)
            # TODO finish the setup
        pass

    # @ti.kernel
    def G2zero(self):
        # ti.block_dim(128)
        # for I in ti.static(self.g_m):
        #     self.g_m[I] = 0.0
        # for I in ti.static(self.g_v):
        #     self.g_v[I] = ts.vecND(self.dim, 0.0)
        self.g_m.fill(0.0)
        self.g_v.fill(ts.vecND(self.dim, 0.0))

    @ti.func
    def kirchoff_FCR(self, F, R, J, mu, la):
        return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(Float, self.dim) * la * J * (
                J - 1)  # compute kirchoff stress for FCR model (remember tau = P F^T)

    @ti.func
    def stencil_range(self, l_b, r_u):
        return [[l_b[d], r_u[d]] for d in range(self.dim)]

    @ti.func
    def stencil_range3(self):
        return ti.ndrange(*((3,) * self.dim))

    @ti.func
    def elasticP2Gpp(self, P, dt):
        """
        P2G pre-process for elastic
        :param P:
        :param dt:
        :return:
        """
        p_Jp = ti.static(self.p_Jp)
        p_F = ti.static(self.p_F)

        h = self.cfg.elastic_h
        mu, la = h * self.cfg.mu_0, h * self.cfg.lambda_0

        U, sig, V = ti.svd(p_F[P])
        J = 1.0

        for d in ti.static(range(self.dim)):
            new_sig = sig[d, d]
            p_Jp[P] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        force = self.kirchoff_FCR(p_F[P], U @ V.transpose(), J, mu, la)
        force *= (-dt * self.cfg.p_vol * 4 * self.cfg.inv_dx ** 2)

        return force

    @ti.func
    def liquidP2Gpp(self, P, dt):
        """

        :param P:
        :param dt:
        :return:
        """
        p_Jp = ti.static(self.p_Jp)
        p_F = ti.static(self.p_F)
        # hardening coefficient
        h = self.cfg.elastic_h
        mu, la = 0.0, h * self.cfg.lambda_0

        U, sig, V = ti.svd(p_F[P])
        J = 1.0

        for d in ti.static(range(self.dim)):
            new_sig = sig[d, d]
            p_Jp[P] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        new_F = ti.Matrix.identity(Float, self.dim)
        new_F[0, 0] = J
        p_F[P] = new_F

        force = self.kirchoff_FCR(p_F[P], U @ V.transpose(), J, mu, la)
        force *= (-dt * self.cfg.p_vol * 4 * self.cfg.inv_dx ** 2)

        return force

    @ti.func
    def snowP2Gpp(self, P, dt):
        """

        :param P:
        :param dt:
        :return:
        """
        p_Jp = ti.static(self.p_Jp)
        p_F = ti.static(self.p_F)

        h = ti.exp(10 * (1.0 - p_Jp[P]))
        mu, la = h * self.cfg.mu_0, h * self.cfg.lambda_0

        U, sig, V = ti.svd(p_F[P])
        J = 1.0

        for d in ti.static(range(self.dim)):
            new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                          1 + 4.5e-3)  # Plasticity
            p_Jp[P] *= sig[d, d] / new_sig

            sig[d, d] = new_sig
            J *= new_sig

        p_F[P] = U @ sig @ V.transpose()

        force = self.kirchoff_FCR(p_F[P], U @ V.transpose(), J, mu, la)
        force *= (-dt * self.cfg.p_vol * 4 * self.cfg.inv_dx ** 2)

        return force

    @ti.func
    def sandP2Gpp(self, P, dt):
        """

        :param P:
        :param dt:
        :return:
        """
        p_Jp = ti.static(self.p_Jp)
        p_F = ti.static(self.p_F)

        h = ti.exp(10 * (1.0 - p_Jp[P]))
        mu, la = h * self.cfg.mu_0, h * self.cfg.lambda_0
        # TODO ?
        U, sig, V = ti.svd(p_F[P])
        # TODO ?
        J = 1.0
        # TODO what the hell is this
        sig = self.sand_projection(sig, P)
        p_F[P] = U @ sig @ V.transpose()
        log_sig_sum = 0.0
        center = ti.Matrix.zero(Float, self.dim, self.dim)

        for d in ti.static(range(self.dim)):
            log_sig_sum += ti.log(sig[d, d])
            center[d, d] = 2.0 * self.cfg.mu_0 * ti.log(sig[d, d]) * (1 / sig[d, d])

        for d in ti.static(range(self.dim)):
            center[d, d] += self.cfg.lambda_0 * log_sig_sum * (1 / sig[d, d])

        force = U @ center @ V.transpose() @ p_F[P].transpose()
        force *= (-dt * self.cfg.p_vol * 4 * self.cfg.inv_dx ** 2)

        return force

    @ti.func
    def sand_projection(self, sigma, P):
        sigma_out = ti.Matrix.zero(Float, self.dim, self.dim)
        epsilon = ts.vecND(self.dim, 0.0)

        for d in ti.static(range(self.dim)):
            epsilon[d] = ti.log(max(abs(sigma[d, d]), 1e-4))
            sigma_out[d, d] = 1

        tr = epsilon.sum() + self.p_Jp[P]
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = epsilon_hat.norm() + 1e-20

        if tr >= 0.0:
            self.p_Jp[P] = tr
        else:
            self.p_Jp[P] = 0.0
            delta_gamma = epsilon_hat_norm + (
                    self.dim * self.cfg.lambda_0 +
                    2 * self.cfg.mu_0) / (2 * self.cfg.mu_0) * tr * self.cfg.alpha
            for d in ti.static(range(self.dim)):
                sigma_out[d, d] = ti.exp(epsilon[d] - max(0, delta_gamma) /
                                         epsilon_hat_norm * epsilon_hat[d])

        return sigma_out

    @ti.kernel
    def P2G(self, dt: Float):
        """
        Particle to Grid
        :param dt:
        :return:
        """
        p_C = ti.static(self.p_C)
        p_v = ti.static(self.p_v)
        p_x = ti.static(self.p_x)
        g_m = ti.static(self.g_m)
        g_v = ti.static(self.g_v)
        p_F = ti.static(self.p_F)
        p_Jp = ti.static(self.p_Jp)

        for P in p_x:
            base = ti.floor(g_m.getG(p_x[P] - 0.5 * g_m.dx)).cast(Int)
            fx = g_m.getG(p_x[P]) - base.cast(Float)
            # print("P2G base: {}, fx: {}".format(base, fx))
            # print("base:", base)
            # print("fx", fx)
            # Here we adopt quadratic kernels
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

            # affine would do this in P2G.. why
            p_F[P] = (ti.Matrix.identity(Int, self.dim) + dt * p_C[P]) @ p_F[P]

            force = ti.Matrix.zero(Float, self.dim, self.dim)
            # want to decrease branching
            if self.p_material_id[P] == MaType.elastic:
                force = self.elasticP2Gpp(P, dt)
            elif self.p_material_id[P] == MaType.liquid:
                force = self.liquidP2Gpp(P, dt)
            elif self.p_material_id[P] == MaType.snow:
                force = self.snowP2Gpp(P, dt)
            elif self.p_material_id[P] == MaType.sand:
                force = self.sandP2Gpp(P, dt)

            affine = force + self.cfg.p_mass * p_C[P]
            for offset in ti.static(ti.grouped(self.stencil_range3())):
                # print("P2G: ", offset)
                dpos = g_m.getW(offset.cast(Float) - fx)

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                # dweight = ts.vecND(self.dim, self.cfg.inv_dx)
                # for d1 in ti.static(range(self.dim)):
                #     for d2 in ti.static(range(self.dim)):
                #         if d1 == d2:
                #             dweight[d1] *= dw[offset[d2]][d2]
                #         else:
                #             dweight[d1] *= w[offset[d2]][d2]

                # force = - self.cfg.p_vol * kirchoff @ dweight
                # TODO ? AFFINE
                # g_v[base + offset] += self.cfg.p_mass * weight * (p_v[P] + p_C[P] @ dpos)  # momentum transfer
                # TODO Got lots of simultaneous atomic here
                g_v[base + offset] += weight * (self.cfg.p_mass * self.p_v[P] + affine @ dpos)
                g_m[base + offset] += weight * self.cfg.p_mass

                # g_v[base + offset] += dt * force

    @ti.kernel
    def G_Normalize_plus_Gravity(self, dt: Float):
        g_m = ti.static(self.g_m)
        g_v = ti.static(self.g_v)
        for I in ti.static(g_m):
            # TODO why no need for epsilon here
            if g_m[I] > 0:
                # TODO
                g_v[I] = 1 / g_m[I] * g_v[I]  # Momentum to velocity
                g_v[I] += dt * self.gravity

    @ti.kernel
    def G_boundary_condition(self):
        g_m = ti.static(self.g_m)
        g_v = ti.static(self.g_v)
        for I in ti.static(g_m):
            # TODO Unbound
            # TODO vectorize
            for d in ti.static(range(self.dim)):
                if I[d] < self.cfg.g_padding[d] and g_v[I][d] < 0.0:
                    g_v[I][d] = 0.0
                if I[d] > self.cfg.res[d] - self.cfg.g_padding[d] and g_v[I][d] > 0.0:
                    g_v[I][d] = 0.0

    @ti.kernel
    def G2P(self, dt: Float):
        p_C = ti.static(self.p_C)
        p_v = ti.static(self.p_v)
        p_x = ti.static(self.p_x)
        g_m = ti.static(self.g_m)
        g_v = ti.static(self.g_v)
        p_F = ti.static(self.p_F)

        for P in p_x:
            base = ti.floor(g_m.getG(p_x[P] - 0.5 * g_m.dx)).cast(Int)
            fx = g_m.getG(p_x[P]) - base.cast(Float)

            w = [
                0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
            ]
            # dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

            new_v = ti.Vector.zero(Float, self.dim)
            new_C = ti.Matrix.zero(Float, self.dim, self.dim)
            # new_F = ti.Matrix.zero(Float, self.dim, self.dim)

            # for offset in ti.static(ti.grouped(ti.ndrange(*self.stencil_range(l_b, r_u)))):
            for offset in ti.static(ti.grouped(self.stencil_range3())):
                dpos = offset.cast(Float) - fx
                v = g_v[base + offset]

                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                # dweight = ts.vecND(self.dim, self.cfg.inv_dx)
                # for d1 in ti.static(range(self.dim)):
                #     for d2 in ti.static(range(self.dim)):
                #         if d1 == d2:
                #             dweight[d1] *= dw[offset[d2]][d2]
                #         else:
                #             dweight[d1] *= w[offset[d2]][d2]
                new_v += weight * v
                # TODO ? what the hell
                new_C += 4 * self.cfg.inv_dx * weight * v.outer_product(dpos)
                # new_F += v.outer_product(dweight)
            # Semi-Implicit
            p_v[P], p_C[P] = new_v, new_C
            p_x[P] += dt * p_v[P]  # advection
            # p_F[P] = (ti.Matrix.identity(Float, self.dim) + (dt * new_F)) @ p_F[P]  # updateF (explicitMPM way)

    @ti.kernel
    def init_cube(self):
        # TODO evolve this
        self.n_particles[None] = self.cfg.n_particle
        group_size = self.n_particles[None] // 4
        for P in self.p_x:
            self.p_x[P] = ts.randND(self.dim) * 0.2 + 0.05 + 0.32 * (P // group_size)
            self.p_x[P][0] = ti.random() * 0.2 + 0.3 + 0.10 * (P // group_size)
            # self.p_x[P] = [ti.random() * 0.2 + 0.3 + 0.10 * (P // group_size),
            #                ti.random() * 0.2 + 0.05 + 0.32 * (P // group_size)]
            self.p_material_id[P] = P // group_size # 1: fluid 0: jelly 2: snow
            self.p_v[P] = ts.vecND(self.dim, 0.0)
            self.p_F[P] = ti.Matrix.identity(Float, self.dim)
            self.p_Jp[P] = 1
            self.p_C[P] = ti.Matrix.zero(Float, self.dim, self.dim)
