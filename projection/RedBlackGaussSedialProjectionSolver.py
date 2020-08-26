import taichi as ti
from .AbstractProjectionSolver import ProjectionSolver

#ref1 : https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf
#ref2 : https://github.com/ShaneFX/GAMES201/blob/330d9c75cacfad6901605d3f589eea11954d9a93/HW01/Smoke3d/smoke_3D.py
@ti.data_oriented
class RedBlackGaussSedialProjectionSolver(ProjectionSolver):

    def __init__(self, cfg, grid):
        super().__init__(cfg, grid)


    @ti.kernel
    def Gauss_Step(self,
                   pf: ti.template(),
                   new_pf: ti.template(),
                   p_divs: ti.template()):
        # TODO: dimension independent coding
        for i, j in pf:
            if (i + j) % 2 == 0:
                pl = self.grid.sample(pf, i - 1, j)
                pr = self.grid.sample(pf, i + 1, j)
                pb = self.grid.sample(pf, i, j - 1)
                pt = self.grid.sample(pf, i, j + 1)
                div = p_divs[i, j]
                new_pf[i, j] = (pl + pr + pb + pt + self.cfg.jacobi_alpha * div) * self.cfg.jacobi_beta

        for i, j in pf:
            if (i + j) % 2 == 1:
                pl = self.grid.sample(new_pf, i - 1, j)
                pr = self.grid.sample(new_pf, i + 1, j)
                pb = self.grid.sample(new_pf, i, j - 1)
                pt = self.grid.sample(new_pf, i, j + 1)
                div = p_divs[i, j]
                new_pf[i, j] = (pl + pr + pb + pt + self.cfg.jacobi_alpha * div) * self.cfg.jacobi_beta

    def runPressure(self):
        self.cfg.jacobi_alpha = self.cfg.poisson_pressure_alpha
        self.cfg.jacobi_beta = self.cfg.poisson_pressure_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Gauss_Step(self.grid.p_pair.cur, self.grid.p_pair.nxt, self.grid.v_divs, )
            self.grid.p_pair.swap()

    def runViscosity(self):
        self.cfg.jacobi_alpha = self.cfg.poisson_viscosity_alpha
        self.cfg.jacobi_beta = self.cfg.poisson_viscosity_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Gauss_Step(self.grid.p_pair.cur, self.grid.p_pair.nxt, self.grid.v_divs, )
            self.grid.p_pair.swap()