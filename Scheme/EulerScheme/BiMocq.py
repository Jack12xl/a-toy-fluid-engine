import taichi as ti
import taichi_glsl as ts
import numpy as np
from .Euler_Scheme import EulerScheme
from config import VisualizeEnum, SceneEnum, SchemeType, SimulateType
from utils import Vector, Matrix

# ref Bimocq 2019, By qi ziyin

class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)
        pass

    def advect(self, dt):
        self.updateForward(self.grid.forward_map)
        self.updateForward(self.grid.forward_scalar_map)

        self.advect_velocity(dt)

        self.advection_solver.advect(self.grid.v_pair.cur, self.grid.density_pair.cur, self.grid.density_pair.nxt, dt)

        if self.cfg.SimType == SimulateType.Gas:
            self.advection_solver.advect(self.grid.v_pair.cur,
                                         self.grid.t_pair.cur,
                                         self.grid.t_pair.nxt,
                                         dt)
            self.grid.t_pair.swap()
        self.grid.swap_v()
        self.grid.density_pair.swap()

    def advectDMC(self, dt):
        pass

    def updateBackward(self, dt):
        pass

    def updateForward(self, M: Matrix, dt: ti.f32):
        """

        :param M: wrapper for mapper
        :param dt:
        :return:
        """
        vf = ti.static(self.grid.v_pair.cur)
        for I in ti.static(M):
            pos = M[I]
            M[I] = self.advection_solver.backtrace(vf, pos, dt)

    def schemeStep(self, ext_input: np.array):
        self.advect(self.cfg.dt)

        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
