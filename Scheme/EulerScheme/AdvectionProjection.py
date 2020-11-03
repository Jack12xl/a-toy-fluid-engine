from Euler_Scheme import EulerScheme
import taichi as ti
import taichi_glsl as ts
import numpy as np

@ti.data_oriented
class AdvectionProjectionEulerScheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

    def step(self, ext_input: np.array):
        self.boundarySolver.step_update_sdfs(self.boundarySolver.colliders)
        self.boundarySolver.kern_update_marker()
        for colld in self.boundarySolver.colliders:
            colld.surfaceshape.update_transform(self.cfg.dt)

        self.advect(self.cfg.dt)
        self.externalForce(ext_input, self.cfg.dt)
        self.project()
        self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)

        self.boundarySolver.ApplyBoundaryCondition()

        self.render_frame()
        if len(self.boundarySolver.colliders):
            self.render_collider()