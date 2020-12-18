from .Euler_Scheme import EulerScheme
import taichi as ti
import numpy as np


@ti.data_oriented
class AdvectionProjectionEulerScheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

    def schemeStep(self, ext_input: np.array):
        self.advect(self.cfg.dt)
        self.externalForce(ext_input, self.cfg.dt)

        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
