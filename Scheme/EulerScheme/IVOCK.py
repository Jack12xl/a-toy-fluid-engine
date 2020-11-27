from .Euler_Scheme import EulerScheme
import taichi as ti
import taichi_glsl as ts
import numpy as np


@ti.data_oriented
class IVOCK_EulerScheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

    def advect(self, dt):

        pass

    def schemeStep(self, ext_input: np.array):
        self.grid.calVorticity(self.grid.v_pair.cur)
        # TODO vorticity enhancement on vorticity

        # advect velocity
        for v_pair in self.grid.advect_v_pairs:
            self.advection_solver.advect(self.grid.v_pair.cur, v_pair.cur, v_pair.nxt,
                                         self.cfg.dt)

        self.advect()
        pass