import taichi as ti
import taichi_glsl as ts
from .AbstractProjectionSolver import ProjectionSolver
from .MGPCG import MGPCG


@ti.data_oriented
class ConjugateGradientProjectionSolver(ProjectionSolver):

    def __init__(self, cfg, grid, bdry):
        super().__init__(cfg, grid)
        self.core = MGPCG(
            dx=cfg.dx,
            dim=cfg.dim,
            N=cfg.res[0],
            n_mg_levels=4,
        )

    def runPressure(self):
        self.core.init(self.grid.v_divs, - 1)
        self.core.solve(max_iters=-1, verbose=True)
        self.core.get_result(self.grid.p_pair.cur)
        # self.grid.p_pair.swap()

    def runViscosity(self):
        pass
