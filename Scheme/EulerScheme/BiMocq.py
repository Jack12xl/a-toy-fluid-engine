import taichi as ti
import taichi_glsl as ts
import numpy as np
from .Euler_Scheme import EulerScheme
from utils import Vector, Matrix, Wrapper, Float

# ref Bimocq 2019, from Qu ziyin
err = 0.0001


class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.blend_coefficient = 0.5

        self.curFrame = 0
        self.LastVelRemeshFrame = 0
        self.LastScalarRemeshFrame = 0

        self.total_resampleCount = 0

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
        return ts.clamp(pos, 0.0, ti.Vector(self.cfg.res) * self.cfg.dx)

    def advect(self, dt):
        self.updateForward(self.grid.forward_map, dt)
        self.updateForward(self.grid.forward_scalar_map, dt)

        self.updateBackward(self.grid.backward_map, dt)
        # self.grid.backward_map.copy(self.grid.tmp_map)
        self.grid.backward_map, self.grid.tmp_map = self.grid.tmp_map, self.grid.backward_map

        self.updateBackward(self.grid.backward_scalar_map, dt)
        # self.grid.backward_scalar_map.copy(self.grid.tmp_map)
        self.grid.backward_scalar_map, self.grid.tmp_map = self.grid.tmp_map, self.grid.backward_scalar_map

        # advect velocity, temperature, density
        super(Bimocq_Scheme, self).advect(dt)
        # actually store the velocity before advection
        self.grid.v_presave.copy(self.grid.v_pair.nxt)
        # IntegrateMultiLevel {3.5}
        self.grid.v_pair.nxt.fill(ts.vecND(self.dim, 0.0))
        self.grid.t_pair.nxt.fill(ts.vecND(1, 0.0))
        self.grid.density_pair.nxt.fill(ts.vecND(3, 0.0))
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
        self.grid.density_pair.swap()
        self.grid.t_pair.swap()
        self.grid.swap_v()

    def advectBimocq_velocity(self, v_pairs: list):
        for d, v_pair in enumerate(v_pairs):
            self.doubleAdvect_kernel(v_pair.cur,
                                     v_pair.nxt,
                                     self.grid.v_origin.fields[d],
                                     self.grid.v_init.fields[d],
                                     self.grid.d_v.fields[d],
                                     self.grid.d_v_prev.fields[d],
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
        for d in ti.static(range(self.dim)):
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

        # M, tmp_M = tmp_M, M

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
            p_frwd = FM.sample(I)
            p_bkwd = BM.interpolate(p_frwd)

            d = ts.distance(p_init, p_bkwd)

            p_bkwd = BM.sample(I)
            p_frwd = FM.interpolate(p_bkwd)

            d = ti.max(d, ts.distance(p_init, p_frwd))
            # TODO Taichi has thread local memory,
            # so this won't be too slow
            ret = ti.atomic_max(d, ret)
            # if d > ret:
            #     ret = d

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
                ret = ti.atomic_max(v_abs, ret)

        return ret

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
                samplePos = BM.interpolate(pos)
                samplePos = self.clampPos(samplePos)
                d_f[I] += w * coeff * d_f_src.interpolate(samplePos)

    def AccumulateVelocity(self, diff_src, coeff):
        """

        :param diff_src:
        :param coeff:
        :return:
        """
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

    def reSampleVelBuffer(self):
        self.total_resampleCount += 1
        self.grid.v_origin = self.grid.v_init
        self.grid.v_init = self.grid.v
        self.grid.d_v_prev = self.grid.d_v

        self.grid.d_v.fill(ts.vecND(self.dim, 0.0))

        self.grid.backward_map_bffr = self.grid.backward_map

        self.grid.init_map(self.grid.forward_map)
        self.grid.init_map(self.grid.backward_map)

    @ti.kernel
    def blendVel(self,
                 v1: Wrapper,
                 v2: Wrapper):
        for d in ti.static(range(self.dim)):
            for I in ti.static(v1.fields[d]):
                v1[I] = 0.5 * (v1[I] + v2[I])

    def schemeStep(self, ext_input: np.array):
        if self.curFrame != 0:
            self.grid.v_pair.cur.copy(self.grid.v_tmp)

        self.advect(self.cfg.dt)
        # self.decorator_track_delta(
        #     delta=self.grid.d_v_tmp,
        #     track_what=self.grid.v_pair.cur
        # )(self.externalForce)(ext_input, self.cfg.dt)

        # trace the d_v
        self.grid.d_v_tmp.copy(self.grid.v_pair.cur)
        self.externalForce(ext_input, self.cfg.dt)
        self.grid.d_v_tmp.subself(self.grid.v_pair.cur)

        # track the delta v from projection
        self.grid.d_v_proj.copy(self.grid.v_pair.cur)
        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)

        d_vel = self.estimateDistortion(self.grid.backward_map, self.grid.forward_map)
        d_scalar = self.estimateDistortion(self.grid.backward_scalar_map, self.grid.forward_scalar_map)
        max_vel = self.getMaxVel(self.grid.v_pair.cur)

        VelocityDistortion = d_vel / (max_vel * self.cfg.dt + err)
        ScalarDistortion = d_scalar / (max_vel * self.cfg.dt + err)

        print("Velocity Distortion : {}".format(VelocityDistortion))
        print("Scalar Distortion : {}".format(ScalarDistortion))
        print("Max abs Velocity : {}".format(max_vel))

        vel_remapping = VelocityDistortion > 1.0 or (self.curFrame - self.LastVelRemeshFrame >= 8)
        rho_remapping = ScalarDistortion > 1.0 or (self.curFrame - self.LastScalarRemeshFrame >= 20)
        # substract
        self.grid.d_v_proj.subself(self.grid.v_pair.cur)
        #
        proj_coeff = 1.0 if vel_remapping else 2.0
        self.AccumulateVelocity(self.grid.d_v_tmp, 1.0)
        self.AccumulateVelocity(self.grid.d_v_proj, proj_coeff)

        if vel_remapping:
            self.LastVelRemeshFrame = self.curFrame
            self.reSampleVelBuffer()
            self.AccumulateVelocity(self.grid.d_v_proj, proj_coeff)

        self.grid.v_tmp.copy(self.grid.v_pair.cur)
        if self.curFrame != 0:
            self.blendVel(self.grid.v_pair.cur, self.grid.v_presave)

        self.curFrame += 1
        print("Cur frame ", self.curFrame)
