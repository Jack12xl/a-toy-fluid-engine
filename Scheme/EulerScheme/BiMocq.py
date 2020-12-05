import taichi as ti
import taichi_glsl as ts
import numpy as np
from .Euler_Scheme import EulerScheme
from config import VisualizeEnum, SceneEnum, SchemeType, SimulateType
from utils import Vector, Matrix, Wrapper

# ref Bimocq 2019, By qi ziyin
err = 0.0001


class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.blend_coefficient = 0.5
        pass

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

    def advectBimocq_velocity(self, vf: Wrapper):
        pass

    @ti.kernel
    def advect_twice_kern(self, f: Wrapper):
        """
        advect twice
        :param f:
        :return:
        """

        pass

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
            if ti.abs(a[d]) > err:
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
            M[I] = self.advection_solver.backtrace(vf, pos, -dt)

    def schemeStep(self, ext_input: np.array):
        self.advect(self.cfg.dt)
        self.externalForce(ext_input, self.cfg.dt)

        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
