import taichi as ti
import taichi_glsl as ts
import numpy as np
from .Euler_Scheme import EulerScheme
from utils import Vector, Matrix, Wrapper, Float

# ref Bimocq 2019, By qi ziyin
err = 0.0001


class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.blend_coefficient = 0.5

        self.doubleAdvect_kernel = None
        self.w_self = None
        # weight S
        self.ws = None
        # direction S
        self.dirs = None
        self.dA_d = 0.25  # neighbour of double Advect
        if self.dim == 2:
            # self.w_self = 0.5
            # self.w_nghbr = (1.0 - self.w_self) / 4.0
            self.doubleAdvect_kernel = self.doubleAdvectKern2D
            self.ws = [0.125, 0.125, 0.125, 0.125, 0.5]
            self.dirs = [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25], [0.0, 0.0]]
        elif self.dim == 3:
            # self.w_self = 0.0
            # self.ws = (1.0 - self.w_self) / 8.0
            self.doubleAdvect_kernel = self.doubleAdvectKern2D

    @ti.pyfunc
    def clampPos(self, pos):
        """

        :param pos: world pos
        :return:
        """
        return ts.clamp(pos, 0.0, ti.Vector(self.cfg.res))

    def advect(self, dt):
        self.updateForward(self.grid.forward_map)
        self.updateForward(self.grid.forward_scalar_map)
        self.updateBackward(self.grid.backward_map)
        self.updateBackward(self.grid.backward_scalar_map)
        # advect velocity, temperature, density
        super(Bimocq_Scheme, self).advect(dt)

        # IntegrateMultiLevel {3.5}
        self.grid.v_pair.nxt.fill(ts.vecND(self.dim, 0.0))
        self.grid.p_pair.nxt.fill(ts.vecND(self.dim, 0.0))
        self.grid.density_pair.nxt.fill(ts.vecND(self.dim, 0.0))
        #
        self.IntegrateMultiLevel(dt)

    def IntegrateMultiLevel(self, dt):
        # TODO need correct
        self.advectBimocq_velocity(self.grid.advect_v_pairs)
        self.doubleAdvect_kernel(self.grid.t_pair.cur,
                                 self.grid.t_pair.nxt,
                                 self.grid.T_origin,
                                 self.grid.T_init,
                                 self.grid.d_T,
                                 self.grid.d_T_prev,
                                 self.grid.backward_scalar_map,
                                 self.grid.backward_scalar_map_bffr
                                 )
        self.doubleAdvect_kernel(self.grid.density_pair.cur,
                                 self.grid.density_pair.nxt,
                                 self.grid.rho_origin,
                                 self.grid.rho_init,
                                 self.grid.d_rho,
                                 self.grid.d_rho_prev,
                                 self.grid.backward_scalar_map,
                                 self.grid.backward_scalar_map_bffr
                                 )
        self.grid.v_pair.swap()
        self.grid.t_pair.swap()
        self.grid.swap_v()

    def advectBimocq_velocity(self, v_pairs: list):
        for v_pair in v_pairs:
            self.doubleAdvect_kernel(v_pair.cur,
                                     v_pair.nxt,
                                     self.grid.v_origin,
                                     self.grid.v_init,
                                     self.grid.d_v,
                                     self.grid.d_v_prev,
                                     self.grid.backward_map,
                                     self.grid.backward_map_bffr)

    @ti.kernel
    def doubleAdvectKern2D(self,
                           f: Wrapper,
                           f_n: Wrapper,
                           f_orig: Wrapper,
                           f_init: Wrapper,
                           d_f: Wrapper,
                           d_f_prev: Wrapper,
                           BM: Wrapper,
                           p_BM: Wrapper
                           ):
        """
        advect twice
        :param p_BM: previous backward mapper
        :param BM:
        :param d_f_prev:
        :param d_f:
        :param f_init:
        :param f_orig:
        :param f: the wrapper of input field(T, rho, velocity)
        :param f_n: the next buffer, where the advected results store
        :return:
        """
        drct = [1.0, -1.0]

        # BM = ti.static(self.grid.backward_map)
        # p_BM = ti.static(self.grid.backward_map_bffr)
        for I in ti.static(f):
            if I[0] == 0 or I[1] == 0 or I[0] == f.shape[0] - 1 or I[1] == f.shape[1] - 1:
                # on the boundary
                f_n[I] = f[I]
            else:
                for drct, w in ti.static(zip(self.dirs, self.ws)):
                    pos = f.getW(I + ti.Vector(drct))

                    pos1 = BM.interpolate(pos)
                    pos1 = self.clampPos(pos1)

                    pos2 = p_BM.interpolate(pos1)
                    pos2 = self.clampPos(pos2)

                    f_n[I] += (1.0 - self.blend_coefficient) * w * (
                            f_orig.interpolate(pos2) +
                            d_f.interpolate(pos1) +
                            d_f_prev.interpolate(pos2)
                    )
                    f_n[I] += self.blend_coefficient * w * (
                            f_init.interpolate(pos1) +
                            d_f.interpolate(pos1)
                    )

    @ti.kernel
    def doubleAdvectKern3D(self, f: Wrapper):
        """
        advect twice
        :param f: the wrapper of input field(T, rho, velocity)
        :return:
        """
        dir = [1.0, -1.0]

    @ti.func
    def advectDMC(self, M, dt):
        M_tmp = ti.static(self.grid.tmp_map)
        for I in ti.static(M):
            pos = M.getW(I)
            back_pos = self.solveODE_DMC(pos, dt)
            M_tmp[I] = M.interpolate(back_pos)

    @ti.func
    def solveODE_DMC(self, pos, dt):
        """
        Based on (12) ~ (14)
        :param pos:
        :param dt:
        :return:
        """
        vf = ti.static(self.grid.v_pair.cur)

        a = self.calculateA(pos)
        # trace DMC
        # TODO DMC requires CFL < 1.0
        vel = vf.interpolate(pos)
        dmc_trace_pos = pos - dt * vel

        back_trace_pos = self.advection_solver.backtrace(vf, pos, dt)
        for d in ti.static(self.dim):
            if ti.abs(a[d]) > ti.static(err):
                dmc_trace_pos[d] = pos[d] - (1.0 - ti.exp(-a[d] * dt)) * vel[d] / a[d]
            else:
                dmc_trace_pos[d] = back_trace_pos[d]

        return dmc_trace_pos

    @ti.func
    def calculateA(self, pos):
        """
        Based on (10), (11) in paper
        :param pos: world pos
        :return:
        """
        vf = ti.static(self.grid.v_pair.cur)

        vel = vf.interpolate(pos)

        new_pos = pos + vf.dx * ts.sign(vel)
        new_vel = vf.interpolate(new_pos)

        return (new_vel - vel) / (new_pos - pos + err)

    @ti.kernel
    def updateBackward(self, M: Matrix, dt: ti.f32):
        tmp_M = ti.static(self.grid.tmp_map)
        # DMC
        self.advectDMC(M, dt)

        M, tmp_M = tmp_M, M

    @ti.kernel
    def updateForward(self, M: Matrix, dt: ti.f32):
        """

        :param M: wrapper for mapper
        :param dt:
        :return:
        """
        vf = ti.static(self.grid.v_pair.cur)
        for I in ti.static(M):
            pos = M[I]
            # TODO maybe need clamp here
            M[I] = self.clampPos(self.advection_solver.backtrace(vf, pos, -dt))

    def decorator_track_delta(self, delta, track_what):
        """
        track the track_what and store its change in delta
        :param delta:
        :param track_what:
        :return:
        """
        raise DeprecationWarning

        def Inner(traced_func):
            def wrapper(*args, **kwargs):
                delta.copy(track_what)
                traced_func(*args, **kwargs)

            return wrapper

        return Inner

    @ti.kernel
    def estimateDistortion(self, BM: Wrapper, FM: Wrapper) -> Float:
        """
        Based on paper {#3.4} eqa (20)
        Get the largest Distortion
        :param BM:
        :param FM:
        :return:
        """
        ret = 0.0
        for I in ti.static(BM):
            # TODO origin code handles boundary
            p_init = BM.getW(I)
            p_frwd = FM.sample[I]
            p_bkwd = BM.interpolate(p_frwd)

            d = ts.distance(p_init, p_bkwd)

            p_bkwd = BM.sample[I]
            p_frwd = FM.interpolate(p_bkwd)

            d = ti.max(d, ts.distance(p_init, p_frwd))
            # TODO Taichi has thread local memory,
            # so this won't be too slow
            ret = ti.max(d, ret)

            if d > ret:
                ret = d

        return ret

    @ti.kernel
    def getMaxVel(self, v: Wrapper) -> Float:
        """
        Assyne v is FaceGrud
        :param v:
        :return:
        """
        ret = 0.0
        for d in ti.static(range(self.dim)):
            for I in ti.static(v.fields[d]):
                pass

    def schemeStep(self, ext_input: np.array):
        self.advect(self.cfg.dt)
        # self.decorator_track_delta(
        #     delta=self.grid.d_v_tmp,
        #     track_what=self.grid.v_pair.cur
        # )(self.externalForce)(ext_input, self.cfg.dt)
        # trace the d_v
        self.grid.d_v_tmp.copy(self.grid.v_pair.cur)
        self.externalForce(ext_input, self.cfg.dt)
        self.grid.d_v_tmp.subself(self.grid.v_pair.cur)

        self.grid.d_v_proj.copy(self.grid.v_pair.cur)
        self.project()
        d_vel = self.estimateDistortion(self.grid.backward_map, self.grid.forward_map)
        d_scalar = self.estimateDistortion(self.grid.backward_scalar_map, self.grid.forward_scalar_map)


        self.grid.d_v_proj.subself(self.grid.v_pair.cur)

        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
