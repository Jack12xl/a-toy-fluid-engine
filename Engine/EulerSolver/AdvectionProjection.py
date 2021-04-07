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

        #debug
        np_pressure = self.grid.p_pair.cur.field.to_numpy()
        print(f'before project, pressure mean: {np.mean(np_pressure)}')
        self.project()
        np_pressure = self.grid.p_pair.cur.field.to_numpy()
        print(f'after project, pressure mean: {np.mean(np_pressure)}')

        np_v = self.grid.v_pair.cur.field.to_numpy()
        print(f'before subtract, v mean: {np.mean(np_v)}')
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)
        np_v = self.grid.v_pair.cur.field.to_numpy()
        print(f'after project, v mean: {np.mean(np_v)}')