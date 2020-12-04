import taichi as ti
import taichi_glsl as ts
import numpy as np
from .Euler_Scheme import EulerScheme
from config import VisualizeEnum, SceneEnum, SchemeType, SimulateType


# ref Bimocq 2019, By qi ziyin

class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)
        pass

    def advect(self, dt):
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

    def updateForward(self, dt):
        pass

    def schemeStep(self, ext_input: np.array):
        self.updateForward(self.cfg.dt)
        self.updateBackward(self.cfg.dt)

        self.advect(self.cfg.dt)

        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
