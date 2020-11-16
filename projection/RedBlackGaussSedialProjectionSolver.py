import taichi as ti
import taichi_glsl as ts
from .AbstractProjectionSolver import ProjectionSolver
from utils import Float


# ref1 : https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf
# ref2 : https://github.com/ShaneFX/GAMES201/blob/330d9c75cacfad6901605d3f589eea11954d9a93/HW01/Smoke3d/smoke_3D.py
@ti.data_oriented
class RedBlackGaussSedialProjectionSolver(ProjectionSolver):

    def __init__(self, cfg, grid):
        super().__init__(cfg, grid)

    @ti.kernel
    def Gauss_Step(self,
                   pf: ti.template(),
                   new_pf: ti.template(),
                   p_divs: ti.template(),
                   alpha: Float,
                   beta: Float):

        for I in ti.grouped(pf.field):
            if ts.summation(I) % 2 == 0:
                div = p_divs[I]
                ret = 0.0
                for d in ti.static(range(self.cfg.dim)):
                    D = ti.Vector.unit(self.cfg.dim, d)

                    p0 = pf.sample(I + D)
                    p1 = pf.sample(I - D)

                    ret += p0 + p1
                new_pf[I] = (ret + alpha * div) * beta
                # pl = pf.sample(I + ts.D.zy)
                # pr = pf.sample(I + ts.D.xy)
                # pb = pf.sample(I + ts.D.yz)
                # pt = pf.sample(I + ts.D.yx)
                # div = p_divs[I]
                # new_pf[I] = (pl + pr + pb + pt + alpha * div) * beta

        for I in ti.grouped(pf.field):
            if ts.summation(I) % 2 == 1:
                div = p_divs[I]
                ret = 0.0
                for d in ti.static(range(self.cfg.dim)):
                    D = ti.Vector.unit(self.cfg.dim, d)

                    p0 = pf.sample(I + D)
                    p1 = pf.sample(I - D)

                    ret += p0 + p1
                new_pf[I] = (ret + alpha * div) * beta
                # pl = new_pf.sample(I + ts.D.zy)
                # pr = new_pf.sample(I + ts.D.xy)
                # pb = new_pf.sample(I + ts.D.yz)
                # pt = new_pf.sample(I + ts.D.yx)
                # div = p_divs[I]
                # new_pf[I] = (pl + pr + pb + pt + alpha * div) * beta

    def runPressure(self):
        # self.cfg.jacobi_alpha = self.cfg.poisson_pressure_alpha
        # self.cfg.jacobi_beta = self.cfg.poisson_pressure_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Gauss_Step(
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
            self.Gauss_Step(
                self.grid.p_pair.cur,
                self.grid.p_pair.nxt,
                self.grid.v_divs,
                self.cfg.poisson_viscosity_alpha,
                self.cfg.poisson_viscosity_beta
            )
            self.grid.p_pair.swap()
