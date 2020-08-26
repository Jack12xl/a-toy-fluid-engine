import taichi as ti
from .AbstractProjectionSolver import ProjectionSolver

@ti.data_oriented
class JacobiProjectionSolver(ProjectionSolver):

    def __init__(self, cfg, grid):
        super().__init__(cfg, grid)

    @ti.kernel
    def calDivergence(self, vf: ti.template(), vd: ti.template()):
        for i, j in vf:
            vl = self.sample(vf, i - 1, j)[0]
            vr = self.sample(vf, i + 1, j)[0]
            vb = self.sample(vf, i, j - 1)[1]
            vt = self.sample(vf, i, j + 1)[1]
            vc = self.sample(vf, i, j)
            # boundary
            # TODO
            if i == 0:
                vl = -vc[0]
            if i == self.cfg.res[0] - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            if j == self.cfg.res[1] - 1:
                vt = -vc[1]
            vd[i, j] = (vr - vl + vt - vb) * self.cfg.half_inv_dx

    @ti.kernel
    def Jacobi_Step(self, pf: ti.template(), new_pf: ti.template(), p_divs: ti.template()):
        for i, j in pf:
            pl = self.grid.sample(pf, i - 1, j)
            pr = self.grid.sample(pf, i + 1, j)
            pb = self.grid.sample(pf, i, j - 1)
            pt = self.grid.sample(pf, i, j + 1)
            div = p_divs[i, j]
            new_pf[i, j] = (pl + pr + pb + pt + self.cfg.jacobi_alpha * div) * self.cfg.jacobi_beta


    def runPressure(self):
        self.cfg.jacobi_alpha = self.cfg.poisson_pressure_alpha
        self.cfg.jacobi_beta = self.cfg.poisson_pressure_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Jacobi_Step(self.grid.p_pair.cur, self.grid.p_pair.nxt, self.grid.v_divs, )
            self.grid.p_pair.swap()

    def runViscosity(self):
        self.cfg.jacobi_alpha = self.cfg.poisson_viscosity_alpha
        self.cfg.jacobi_beta = self.cfg.poisson_viscosity_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Jacobi_Step(self.grid.p_pair.cur, self.grid.p_pair.nxt, self.grid.v_divs, )
            self.grid.p_pair.swap()


