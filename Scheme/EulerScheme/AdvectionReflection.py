from .Euler_Scheme import EulerScheme
import taichi as ti
import taichi_glsl as ts
import numpy as np
import utils

@ti.data_oriented
class AdvectionReflectionEulerScheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

    def step(self, ext_input: np.array):
        self.boundarySolver.step_update_sdfs(self.boundarySolver.colliders)
        self.boundarySolver.kern_update_marker()
        for colld in self.boundarySolver.colliders:
            colld.surfaceshape.update_transform(self.cfg.dt)

        # ref: https://github.com/ShaneFX/GAMES201/blob/master/HW01/Smoke3d/smoke_3D.py
        self.advect(self.cfg.half_dt)
        self.externalForce(ext_input, self.cfg.half_dt)
        utils.copy_ti_field(self.grid.tmp_v, self.grid.v_pair.cur)
        # cur_v = tmp_v = u^{~1/2}, in Fig.2 in advection-reflection solver paper

        self.project()
        self.grid.subtract_gradient(self.grid.tmp_v, self.grid.p_pair.cur)
        # after projection, tmp_v = u^{1/2}, cur_v = u^{~1/2}
        utils.reflect(self.grid.v_pair.cur, self.grid.tmp_v)

        self.advect(self.cfg.half_dt)
        self.externalForce(ext_input, self.cfg.half_dt)
        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)

        self.boundarySolver.ApplyBoundaryCondition()

        self.render_frame()
        if len(self.boundarySolver.colliders):
            self.render_collider()