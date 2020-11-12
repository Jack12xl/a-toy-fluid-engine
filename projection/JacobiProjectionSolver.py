import taichi as ti
import taichi_glsl as ts
from .AbstractProjectionSolver import ProjectionSolver
from utils import Float

@ti.data_oriented
class JacobiProjectionSolver(ProjectionSolver):

    def __init__(self, cfg, grid):
        super().__init__(cfg, grid)


    @ti.kernel
    def Jacobi_Step(self,
                    pf: ti.template(),
                    new_pf: ti.template(),
                    v_divs: ti.template(),
                    alpha: Float,
                    beta: Float):
        for I in ti.grouped(pf.field):

            pl = pf.sample(I + ts.D.zy)
            pr = pf.sample(I + ts.D.xy)
            pb = pf.sample(I + ts.D.yz)
            pt = pf.sample(I + ts.D.yx)
            div = v_divs[I]
            #TODO
            new_pf[I] = (pl + pr + pb + pt + alpha * div) * beta


    def runPressure(self):
        # self.cfg.jacobi_alpha = self.cfg.poisson_pressure_alpha
        # self.cfg.jacobi_beta = self.cfg.poisson_pressure_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Jacobi_Step(
                self.grid.p_pair.cur,
                self.grid.p_pair.nxt,
                self.grid.v_divs,
                self.cfg.poisson_pressure_alpha,
                self.cfg.poisson_pressure_beta
            )
            self.grid.p_pair.swap()

    def runViscosity(self):
        # self.cfg.jacobi_alpha = self.cfg.poisson_viscosity_alpha
        # self.cfg.jacobi_beta = self.cfg.poisson_viscosity_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Jacobi_Step(
                self.grid.p_pair.cur,
                self.grid.p_pair.nxt,
                self.grid.v_divs,
                self.cfg.poisson_viscosity_alpha,
                self.cfg.poisson_viscosity_beta
            )
            self.grid.p_pair.swap()


