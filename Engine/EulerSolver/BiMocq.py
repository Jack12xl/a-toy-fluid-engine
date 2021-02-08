import taichi as ti
import taichi_glsl as ts
import numpy as np
from .Euler_Scheme import EulerScheme
from utils import Vector, Matrix, Wrapper, Float, Int
from config import VisualizeEnum, SceneEnum, SchemeType, SimulateType

# paper ref: Bimocq 2019, from Qu ziyin .etal

err = 0.0001


class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.blend_coefficient = self.cfg.blend_coefficient

        self.LastVelRemeshFrame = 0
        self.LastScalarRemeshFrame = 0

        self.total_resampleCount = 0

        # self.doubleAdvect_kernel = None
        self.w_self = None
        # weight S
        self.ws = None
        # direction S
        self.dirs = None
        self.dA_d = 0.25  # neighbour of double Advect
        if self.dim == 2:
            self.ws = [0.125, 0.125, 0.125, 0.125, 0.5]
            self.dirs = [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25], [0.0, 0.0]]
        elif self.dim == 3:
            self.blend_coefficient = 0.5
            self.ws = 8 * [0.125]

            self.dirs = []
            for i in [-0.25, 0.25]:
                for j in [-0.25, 0.25]:
                    for k in [-0.25, 0.25]:
                        self.dirs.append([i, j, k])

        self.traceFunc = self.advection_solver.backtrace
        self.save_BM = self.cfg.bool_save and VisualizeEnum.BM in self.cfg.save_what
        self.save_FM = self.cfg.bool_save and VisualizeEnum.FM in self.cfg.save_what
        self.save_Distortion = self.cfg.bool_save and VisualizeEnum.Distortion in self.cfg.save_what

        self.saveFM = self.save_FM_2D if self.dim == 2 else self.save_FM_3D
        self.saveBM = self.saveBM2D if self.dim == 2 else self.saveBM3D

    @ti.pyfunc
    def clampPos(self, pos):
        """

        :param pos: world pos
        :return:
        """
        return ts.clamp(pos, 0.0, ti.Vector(self.cfg.res) * self.cfg.dx)

    def advect_velocity(self, dt):
        for v_pair in self.grid.advect_v_pairs:
            self.SemiLagAdvect(v_pair.cur, v_pair.nxt, dt)

    def SimpleEulerAdvect(self, dt):
        self.advect_velocity(dt)
        self.SemiLagAdvect(self.grid.density_pair.cur,
                           self.grid.density_pair.nxt,
                           dt)
        if self.cfg.SimType == SimulateType.Gas:
            self.SemiLagAdvect(self.grid.t_pair.cur,
                               self.grid.t_pair.nxt,
                               dt)
            self.grid.t_pair.swap()
        self.grid.swap_v()
        self.grid.density_pair.swap()

    def advect(self, dt):
        """
        Advect part for Bimocq
        :param dt:
        :return:
        """
        # max_vel = self.getMaxVel(self.grid.v_pair.cur)
        # print("Before update map: max abs Velocity : {}".format(max_vel))

        self.updateForward(self.grid.forward_map, dt)
        self.updateForward(self.grid.forward_scalar_map, dt)

        self.updateBackward(self.grid.backward_map, dt)
        self.updateBackward(self.grid.backward_scalar_map, dt)

        # d_vel = self.estimateDistortion(self.grid.backward_map, self.grid.forward_map)
        # print("Before advect vel distortion: {}".format(d_vel))
        # d_sca = self.estimateDistortion(self.grid.backward_scalar_map, self.grid.forward_scalar_map)
        # print("Before advect vel distortion: {}".format(d_sca))

        # advect velocity, temperature, density
        # max_vel = self.getMaxVel(self.grid.v_pair.cur)
        # print("before original advect:  max abs Velocity : {}".format(max_vel))

        # super(Bimocq_Scheme, self).advect(dt)
        # simple advect
        self.SimpleEulerAdvect(dt)

        # actually store the velocity before advection
        self.grid.v_presave.copy(self.grid.v_pair.nxt)
        # IntegrateMultiLevel {#3.5}
        self.IntegrateMultiLevel(dt)

    def IntegrateMultiLevel(self, dt):
        # fresh the buffer
        self.grid.v_pair.nxt.fill(ts.vecND(self.dim, 0.0))
        self.grid.t_pair.nxt.fill(ts.vecND(1, 0.0))
        self.grid.density_pair.nxt.fill(ts.vecND(3, 0.0))

        # max_vel = self.getMaxVel(self.grid.v_pair.cur)
        # print("before double advect v cur:  max abs Velocity : {}".format(max_vel))

        self.advectBimocq_velocity()
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
        self.grid.density_pair.swap()
        self.grid.t_pair.swap()
        self.grid.swap_v()

        for d, v_pair in enumerate(self.grid.advect_v_pairs):
            self.ErrorCorrectField(
                1,
                v_pair.cur,
                v_pair.nxt,
                self.grid.v_tmp.fields[d],
                self.grid.v_init.fields[d],
                self.grid.d_v.fields[d],
                self.grid.forward_map,
                self.grid.backward_map
            )
        # density
        self.ErrorCorrectField(
            3,
            self.grid.density_pair.cur,
            self.grid.density_pair.nxt,
            self.grid.rho_tmp,
            self.grid.rho_init,
            self.grid.d_rho,
            self.grid.forward_scalar_map,
            self.grid.backward_scalar_map
        )
        # temperature
        self.ErrorCorrectField(
            1,
            self.grid.t_pair.cur,
            self.grid.t_pair.nxt,
            self.grid.T_tmp,
            self.grid.T_init,
            self.grid.d_T,
            self.grid.forward_scalar_map,
            self.grid.backward_scalar_map
        )
        self.grid.density_pair.swap()
        self.grid.t_pair.swap()
        self.grid.swap_v()
        #
        # max_vel = self.getMaxVel(self.grid.v_pair.cur)
        # print("After correction:  max abs Velocity : {}".format(max_vel))

    @ti.kernel
    def SemiLagAdvect(self, f0: Wrapper, f1: Wrapper, dt: Float):
        """
        A simple semiLag with substep solve
        :param f0:
        :param f1:
        :param dt:
        :return:
        """
        vf = ti.static(self.grid.v_pair.cur)
        for I in ti.static(f0):
            pos = f0.getW(I)
            backpos = self.solveODE(pos, dt)
            f1[I] = f0.interpolate(backpos)

    def advectBimocq_velocity(self):
        for d, v_pair in enumerate(self.grid.advect_v_pairs):
            self.doubleAdvect_kernel(v_pair.cur,
                                     v_pair.nxt,
                                     self.grid.v_origin.fields[d],
                                     self.grid.v_init.fields[d],
                                     self.grid.d_v.fields[d],
                                     self.grid.d_v_prev.fields[d],
                                     self.grid.backward_map,
                                     self.grid.backward_map_bffr)

    @ti.kernel
    def doubleAdvect_kernel(self,
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
        for I in ti.static(f):
            # if I[0] == 0 or I[1] == 0 or I[0] == f.shape[0] - 1 or I[1] == f.shape[1] - 1:
            if f.GisNearBoundary(I, 2):
                # on the boundary
                f_n[I] = f[I]
            else:
                for drct, w in ti.static(zip(self.dirs, self.ws)):
                    # drct = [0.0, 0.0]
                    # w = 0.125
                    pos = f.getW(I + ti.Vector(drct))
                    # print("pos: ", pos)
                    pos1 = BM.interpolate(pos)
                    # print("pos1: before clamp", pos1)
                    pos1 = self.clampPos(pos1)
                    # print("pos1: after clamp", pos1)

                    pos2 = p_BM.interpolate(pos1)
                    pos2 = self.clampPos(pos2)

                    f_n[I] += (1.0 - self.blend_coefficient) * w * (
                            f_orig.interpolate(pos2) +
                            d_f.interpolate(pos1) +
                            d_f_prev.interpolate(pos2)
                    )
                    # add_what = self.blend_coefficient * w * (
                    #         f_init.interpolate(pos1) +
                    #         d_f.interpolate(pos1)
                    # )
                    # print(f_init.interpolate(pos1))
                    # print(pos, pos1, pos2, add_what)
                    f_n[I] += self.blend_coefficient * w * (
                            f_init.interpolate(pos1) +
                            d_f.interpolate(pos1)
                    )

    # @ti.kernel
    # def doubleAdvectKern3D(self, f: Wrapper):
    #     """
    #     advect twice
    #     :param f: the wrapper of input field(T, rho, velocity)
    #     :return:
    #     """
    #     dir = [1.0, -1.0]

    @ti.func
    def solveODE(self, pos, dt):
        vf = ti.static(self.grid.v_pair.cur)

        pos2 = self.traceFunc(vf, pos, dt)
        return pos2
        # ddt = dt
        # pos1 = self.traceFunc(vf, pos, ddt)
        # ddt /= 2.0
        # substeps = 2
        # pos2 = self.traceFunc(vf, pos, ddt)
        # pos2 = self.traceFunc(vf, pos2, ddt)

        # uncomment this to get more accurate results(takes much more time)
        # iter = 0
        # while ts.distance(pos2, pos1) > (err * self.cfg.dx) and iter < 6:
        #     pos1 = pos2
        #     ddt /= 2.0
        #     substeps *= 2
        #     pos2 = pos
        #     for _ in range(substeps):
        #         pos2 = self.traceFunc(vf, pos2, ddt)
        #     iter += 1
        # return pos2

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

        back_trace_pos = self.solveODE(pos, dt)
        for d in ti.static(range(self.dim)):
            if ti.abs(a[d]) > ti.static(err):
                dmc_trace_pos[d] = pos[d] - (1.0 - ti.exp(-a[d] * dt)) * vel[d] / a[d]
                # print(">, dt, a[d]", -a[d], dt)
                # print(">, exp", 1.0 - ti.exp(-a[d] * dt))
                # print(">", dmc_trace_pos[d])
            else:
                dmc_trace_pos[d] = back_trace_pos[d]
        # TODO
        # dmc_trace_pos = back_trace_pos
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
        # (10)
        # TODO fix this after glsl is merged
        new_pos = pos - self.cfg.dx * self.getSign(vel)
        # (11)
        new_vel = vf.interpolate(new_pos)
        # print("a", new_pos, pos, new_vel, vel)
        # (11)
        # return (new_vel - vel) / (new_pos - pos + err)
        return (new_vel - vel) / (new_pos - pos)

    @ti.kernel
    def BackwardIter(self, M: Wrapper, dt: Float):
        """
        Dual mesh characteristic in each iteration
        :param M: backward mapper
        :param dt:
        :return:
        """
        M_tmp = ti.static(self.grid.tmp_map)
        for I in ti.static(M):
            pos = M.getW(I)
            back_pos = self.solveODE_DMC(pos, dt)
            back_pos = self.clampPos(back_pos)
            M_tmp[I] = M.interpolate(back_pos)

    def updateBackward(self, M: Wrapper, dt: Float):
        """
        Dual mesh characteristic
        :param M: backward mapper
        :param dt:
        :return:
        """
        substep = self.grid.CFL[None]
        T = dt
        t = 0
        # back_step = ti.ceil(dt / substep)
        # DMC
        n_substep = int(T // substep) + 1
        substep = T / n_substep

        for _ in range(n_substep):
            self.BackwardIter(M, substep)
            M.copy(self.grid.tmp_map)

        # while t < T:
        #     if t + substep > T:
        #         substep = T - t
        #     self.BackwardIter(M, substep)
        #     M.copy(self.grid.tmp_map)
        #     t += substep
        # comment this if not need to visualize BM
        if self.save_Distortion:
            self.saveBM(M)

    @ti.kernel
    def saveBM2D(self, M: Wrapper):
        for I in ti.static(M):
            self.grid.BM[I] = ts.vec3(M[I], 0.0)

    @ti.kernel
    def saveBM3D(self, M: Wrapper):
        for I in ti.static(M):
            self.grid.BM[I] = M[I]

    @ti.kernel
    def updateForward(self, M: Wrapper, dt: Float):
        """

        :param M: wrapper for forward mapper
        :param dt:
        :return:
        """
        for I in ti.static(M):
            pos = M[I]
            M[I] = self.clampPos(self.solveODE(pos, -dt))
            # TODO comment this if not need to debug
            if self.save_FM:
                self.saveFM(M, I)

    @ti.func
    def save_FM_2D(self, M, I):
        self.grid.FM[I] = ts.vec3(M[I], 0.0)

    @ti.func
    def save_FM_3D(self, M, I):
        self.grid.FM[I] = M[I]

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
            if BM.GisNearBoundary(I, 3) == 0:
                p_init = BM.getW(I)
                p_frwd = FM.sample(I)
                p_bkwd = BM.interpolate(p_frwd)

                d = ts.distance(p_init, p_bkwd)

                p_bkwd = BM.sample(I)
                p_frwd = FM.interpolate(p_bkwd)

                d = ti.max(d, ts.distance(p_init, p_frwd))

                if self.save_Distortion:
                    self.grid.distortion[I] = ts.vec3(d)

                # TODO Taichi has thread local memory,
                # so this won't be too slow
                ti.atomic_max(ret, d)

        return ret

    @ti.kernel
    def getMaxVel(self, v: Wrapper) -> Float:
        """
        Assume v is FaceGrid
        :param v:
        :return:
        """
        ret = 0.0
        for d in ti.static(range(self.dim)):
            for I in ti.static(v.fields[d]):
                v_abs = ti.abs(v.fields[d][I][0])
                ti.atomic_max(ret, v_abs)
        # for d in range(self.dim):
        #     np_f = v.fields[d].field.to_numpy()
        #     cur_max = np.max(np_f)
        #     ret = max(cur_max, ret)
        return ret

    def calCFL(self):
        max_vel = self.getMaxVel(self.grid.v_pair.cur)
        self.grid.CFL[None] = self.cfg.dx / abs(max_vel)
        # self.grid.CFL[None] = 0.01

    @ti.kernel
    def AccumulateField(self,
                        d_f: Wrapper,
                        d_f_src: Wrapper,
                        f1: Wrapper,
                        f2: Wrapper,
                        coeff: Float,
                        FM: Wrapper,
                        BM: Wrapper):
        """
        Based on Paper {# 3.3}
        Need three buffer
        !! should fill f1, f2 with zero ahead
        :param d_f: keep track of the difference
        :param FM: forward mapper
        :param d_f_src: the source of the difference(from projection, external force)
        :param f1: buffer 1
        :param f2: buffer 2
        :param coeff:
        :return:
        """
        # TODO error correction
        for I in ti.static(d_f_src):
            for drct, w in ti.static(zip(self.dirs, self.ws)):
                pos = d_f_src.getW(I + ti.Vector(drct))
                samplePos = FM.interpolate(pos)
                samplePos = self.clampPos(samplePos)
                f1[I] += w * d_f_src.interpolate(samplePos)

        for I in ti.static(d_f_src):
            for drct, w in ti.static(zip(self.dirs, self.ws)):
                pos = d_f_src.getW(I + ti.Vector(drct))
                samplePos = BM.interpolate(pos)
                samplePos = self.clampPos(samplePos)
                f2[I] += w * f1.interpolate(samplePos)

        for I in ti.static(f2):
            f2[I] -= d_f_src[I]
            f2[I] *= 0.5

        for I in ti.static(d_f_src):
            for drct, w in ti.static(zip(self.dirs, self.ws)):
                pos = d_f_src.getW(I + ti.Vector(drct))
                samplePos = FM.interpolate(pos)
                samplePos = self.clampPos(samplePos)
                d_f[I] += w * coeff * d_f_src.interpolate(samplePos)

    def AccumulateVelocity(self, diff_src, coeff):
        """

        :param diff_src:
        :param coeff:
        :return:
        """
        # should fill this buffer with zero
        self.grid.tmp_v.fill(ts.vecND(self.dim, 0.0))
        for i, v_pair in enumerate(self.grid.advect_v_pairs):
            """
            Here use v_pair.nxt, tmp_v as buffer
            """
            v_pair.nxt.fill(ts.vecND(1, 0.0))
            self.AccumulateField(
                self.grid.d_v.fields[i],
                diff_src.fields[i],
                v_pair.nxt,
                self.grid.tmp_v.fields[i],
                coeff,
                self.grid.forward_map,
                self.grid.backward_map
            )

    def AccumulateScalar(self,
                         t_diff_src,
                         rho_diff_src
                         ):
        self.grid.rho_tmp.fill(ts.vecND(3, 0.0))
        self.grid.T_tmp.fill(ts.vecND(1, 0.0))

        self.grid.density_pair.nxt.fill(ts.vecND(3, 0.0))
        self.grid.t_pair.nxt.fill(ts.vecND(1, 0.0))

        self.AccumulateField(
            self.grid.d_T,
            t_diff_src,
            self.grid.t_pair.nxt,
            self.grid.T_tmp,
            1.0,
            self.grid.forward_scalar_map,
            self.grid.backward_scalar_map
        )

        self.AccumulateField(
            self.grid.d_rho,
            rho_diff_src,
            self.grid.density_pair.nxt,
            self.grid.rho_tmp,
            1.0,
            self.grid.forward_scalar_map,
            self.grid.backward_scalar_map
        )

    def ErrorCorrectField(self,
                          f_dim: int,
                          f0: Wrapper,
                          f1: Wrapper,
                          f_tmp: Wrapper,
                          f_init: Wrapper,
                          d_f: Wrapper,
                          FM: Wrapper,
                          BM: Wrapper
                          ):
        """

        :param f_dim: dimension of the field we are dealing with
        :param BM:
        :param FM:
        :param d_f:
        :param f0: field after double advection
        :param f1:
        :param f_tmp: field to store the error term (25)
        :param f_init:
        :return:
        """
        f_tmp.fill(ts.vecND(f_dim, 0.0))
        f1.copy(f0)
        self.ECkern(f0, f1, f_tmp, f_init, d_f, FM, BM)
        self.clampExtreme(f0, f1, f_dim)

    @ti.kernel
    def ECkern(self,
               f0: Wrapper,
               f1: Wrapper,
               f_tmp: Wrapper,
               f_init: Wrapper,
               d_f: Wrapper,
               FM: Wrapper,
               BM: Wrapper
               ):
        """
        Error correction
        :param f0:
        :param f1:
        :param f_tmp:
        :param f_init:
        :return:
        """
        for I in ti.static(f0):
            if f0.GisNearBoundary(I, 2) == 0:
                for drct, w in ti.static(zip(self.dirs, self.ws)):
                    pos = f0.getW(I + ti.Vector(drct))
                    pos1 = self.clampPos(FM.interpolate(pos))
                    f_tmp[I] += w * (f0.interpolate(pos1) - d_f[I])
        # error term (25)
        for I in ti.static(f_tmp):
            f_tmp[I] -= f_init[I]
            f_tmp[I] *= 0.5

        for I in ti.static(f0):
            if f0.GisNearBoundary(I, 2) == 0:
                for drct, w in ti.static(zip(self.dirs, self.ws)):
                    pos = f0.getW(I + ti.Vector(drct))
                    pos1 = self.clampPos(BM.interpolate(pos))
                    f1[I] -= w * f_tmp.interpolate(pos1)

    @ti.pyfunc
    def getSign(self, x, edge=0):
        """
        ref ts.sign, for those from ts version <= 0.10.0, ts.sign is wrong
        :param x:
        :param edge:
        :return -1 <=0, 1 > 0
        """
        ret = x + edge  # type promotion
        ret = abs(x > edge) - abs(x <= edge)
        return ret

    @ti.kernel
    def clampExtreme(self,
                     f0: Wrapper,
                     f1: Wrapper,
                     e_dim: ti.template()):
        """
        do maximum suppresion {3.7.2}
        to f1
        on surrounding neighbours (for 2D, 3x3 for 3D, 3x3x3)
        based on f0

        :param e_dim: the element dimension
        :param f0: field before EC
        :param f1: field after EC
        :return:
        """
        # dim = ti.static(f0.field.n)
        for I in ti.static(f0):
            min_val = ts.vecND(e_dim, 1e+6)
            max_val = ts.vecND(e_dim, 0.0)

            kernel_scope = [[I[d] - 1, I[d] + 1 + 1] for d in range(self.dim)]
            # for d in ti.static(range(self.dim)):
            for J in ti.grouped(ti.ndrange(*kernel_scope)):
                val = abs(f0.sample(J))
                max_val = ti.max(max_val, val)
                min_val = ti.min(min_val, val)

            # TODO ts.sign is not right in version 0.10.0
            s = self.getSign(f1[I])
            f1[I] = s * ts.clamp(ti.abs(f1[I]), min_val, max_val)

    def reSampleVelBuffer(self):
        self.total_resampleCount += 1
        self.grid.v_origin.copy(self.grid.v_init)
        self.grid.v_init.copy(self.grid.v_pair.cur)
        self.grid.d_v_prev.copy(self.grid.d_v)

        self.grid.d_v.fill(ts.vecND(self.dim, 0.0))

        self.grid.backward_map_bffr.copy(self.grid.backward_map)

        self.grid.init_map(self.grid.forward_map)
        self.grid.init_map(self.grid.backward_map)

    def reSampleScalarBuffer(self):
        self.total_resampleCount += 1

        self.grid.rho_origin.copy(self.grid.rho_init)
        self.grid.rho_init.copy(self.grid.density_pair.cur)
        self.grid.d_rho_prev.copy(self.grid.d_rho)
        self.grid.d_rho.fill(ts.vecND(3, 0.0))

        self.grid.T_origin.copy(self.grid.T_init)
        self.grid.T_init.copy(self.grid.t_pair.cur)
        self.grid.d_T_prev.copy(self.grid.d_T)
        self.grid.d_T.fill(ts.vecND(1, 0.0))

        self.grid.backward_scalar_map_bffr.copy(self.grid.backward_scalar_map)

        self.grid.init_map(self.grid.forward_scalar_map)
        self.grid.init_map(self.grid.backward_scalar_map)

    @ti.kernel
    def blendVel(self,
                 v1: Wrapper,
                 v2: Wrapper):
        """

        :param v1: velocity field
        :param v2:
        :return:
        """
        for d in ti.static(range(self.dim)):
            for I in ti.static(v1.fields[d]):
                v1.fields[d][I] = 0.5 * (v1.fields[d][I] + v2.fields[d][I])

    def refill(self):
        for emitter in self.emitters:
            emitter.stepEmitHardCode(self.grid.v_pair.cur, self.grid.density_pair.cur, self.grid.t_pair.cur)
            # emitter.stepEmitHardCode(self.grid.v_init, self.grid.rho_init, self.grid.T_init)
            # emitter.stepEmitHardCode(self.grid.v_origin, self.grid.rho_origin, self.grid.T_origin)

    def materialize_emitter(self):
        for emitter in self.emitters:
            emitter.kern_materialize()
            # init the density and velocity for advection
            emitter.stepEmitHardCode(self.grid.v_pair.cur, self.grid.density_pair.cur, self.grid.t_pair.cur)
            emitter.stepEmitHardCode(self.grid.v_init, self.grid.rho_init, self.grid.T_init)
            emitter.stepEmitHardCode(self.grid.v_origin, self.grid.rho_origin, self.grid.T_origin)

    def schemeStep(self, ext_input: np.array):
        # T = 0.0
        substep = self.cfg.CFL * self.cfg.dx / self.getMaxVel(self.grid.v_pair.cur)

        n_substep = int(self.cfg.dt // substep) + 1
        substep = self.cfg.dt / n_substep
        for _ in range(n_substep):
            print("cur CFL: {}".format(substep * self.getMaxVel(self.grid.v_pair.cur) / self.cfg.dx))
            print("cur dt: {}".format(substep))
            self.substep(ext_input, substep)

        # # this method is not cool for taichi compilation
        # while T < self.cfg.dt:
        #     if T + substep > self.cfg.dt:
        #         substep = self.cfg.dt - T
        #     print("cur CFL: {}".format(substep * self.getMaxVel(self.grid.v_pair.cur) / self.cfg.dx))
        #     print("cur dt: {}".format(substep))
        #     self.substep(ext_input, substep)
        #     T += substep

    def substep(self, ext_input, subdt):
        """
        Run once in each frame
        :param subdt:
        :return:
        """
        self.calCFL()
        print("ODE Substep: {}".format(self.grid.CFL[None]))

        # if self.curFrame != 0:
        #     self.grid.v_pair.cur.copy(self.grid.v_tmp)

        self.advect(subdt)

        # trace the d_v
        # serve as v_save
        self.grid.d_v_tmp.copy(self.grid.v_pair.cur)
        self.grid.d_T_tmp.copy(self.grid.t_pair.cur)
        self.grid.d_rho_tmp.copy(self.grid.density_pair.cur)

        # max_vel = self.getMaxVel(self.grid.v_pair.cur)
        # print("after advection max abs Velocity : {}".format(max_vel))
        ### External Force
        self.externalForce(ext_input, subdt)

        self.grid.d_v_tmp.subself(self.grid.v_pair.cur)
        self.grid.d_T_tmp.subself(self.grid.t_pair.cur)
        self.grid.d_rho_tmp.subself(self.grid.density_pair.cur)

        # max_vel = self.getMaxVel(self.grid.v_pair.cur)
        # print("after external force  max abs Velocity : {}".format(max_vel))
        # track the delta v from projection
        self.grid.d_v_proj.copy(self.grid.v_pair.cur)
        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
        self.grid.boundaryZeroVelocity(howNear=2)

        # difference of backward and forward
        d_vel = self.estimateDistortion(self.grid.backward_map, self.grid.forward_map)
        d_scalar = self.estimateDistortion(self.grid.backward_scalar_map, self.grid.forward_scalar_map)
        max_vel = self.getMaxVel(self.grid.v_pair.cur)

        VelocityDistortion = d_vel / ((max_vel + err) * subdt)
        ScalarDistortion = d_scalar / ((max_vel + err) * subdt)

        # print("d_vel: {}".format(d_vel))
        # print("d_scalar: {}".format(d_scalar))
        print("Velocity Distortion : {}".format(VelocityDistortion))
        print("Scalar Distortion : {}".format(ScalarDistortion))
        # print("After project Max abs Velocity : {}".format(max_vel))

        vel_remapping = VelocityDistortion > self.cfg.vel_remap_threshold or (
                self.curFrame - self.LastVelRemeshFrame >= self.cfg.vel_remap_frequency)
        sca_remapping = ScalarDistortion > self.cfg.sclr_remap_threshold or (
                self.curFrame - self.LastScalarRemeshFrame >= self.cfg.sclr_remap_frequency)

        # substract
        self.grid.d_v_proj.subself(self.grid.v_pair.cur)
        #
        proj_coeff = 1.0 if vel_remapping else 2.0
        self.AccumulateVelocity(self.grid.d_v_tmp, 1.0)
        # max_vel = self.getMaxVel(self.grid.d_v)
        # print("Accumulate external d_v: {}".format(max_vel))
        self.AccumulateVelocity(self.grid.d_v_proj, proj_coeff)
        # max_vel = self.getMaxVel(self.grid.d_v_proj)
        # print("Accumulate project d_v: {}".format(max_vel))
        self.AccumulateScalar(self.grid.d_T, self.grid.d_rho)

        if vel_remapping:
            print("Remap velocity")
            self.LastVelRemeshFrame = self.curFrame
            self.reSampleVelBuffer()
            self.AccumulateVelocity(self.grid.d_v_proj, proj_coeff)
            # max_vel = self.getMaxVel(self.grid.d_v_proj)
            print("Accumulate project d_v: {}".format(max_vel))

        if sca_remapping:
            print("Remap scalar")
            self.LastScalarRemeshFrame = self.curFrame
            self.reSampleScalarBuffer()

        self.grid.v_tmp.copy(self.grid.v_pair.cur)

        # if self.curFrame != 0:
        #     self.blendVel(self.grid.v_pair.cur, self.grid.v_presave)

        print("Cur frame ", self.curFrame)
        print("")
