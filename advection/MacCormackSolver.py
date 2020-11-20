import taichi as ti
import taichi_glsl as ts
from enum import Enum
from .AbstractAdvectionSolver import AdvectionSolver
from .SemiLagrangianSolver import SemiLagrangeSolver
from utils import Vector, Matrix


@ti.data_oriented
class MacCormackSolver(AdvectionSolver):

    def __init__(self, cfg, grid, bdrySdf, pixel_marker):
        super().__init__(cfg)
        self.RK = cfg.semi_order
        self.grid = grid
        self.pixel_marker = pixel_marker

        self.subsolver = SemiLagrangeSolver(cfg, grid, bdrySdf, pixel_marker)

    @ti.func
    def advect_func(self,
                    vec_field: ti.template(),
                    q_cur: ti.template(),
                    q_nxt: ti.template(),
                    # boundarySdf: Matrix,
                    dt: ti.template()):

        for I in ti.grouped(vec_field):
            # pos = I + 0.5
            pos = float(I) * vec_field.dx
            p_mid = self.subsolver.backtrace(vec_field, pos,
                                             # boundarySdf,
                                             dt)
            q_mid = q_cur.interpolate(p_mid)

            p_fin = self.subsolver.backtrace(vec_field, p_mid,
                                             # boundarySdf,
                                             -dt)
            q_fin = q_cur.interpolate(p_fin)

            q_nxt[I] = q_mid + 0.5 * (q_fin - q_cur[I])
            # clipping to prevent overshooting
            if ti.static(self.cfg.macCormack_clipping):
                # ref: advection.py from taichi class 4
                min_val, max_val = q_cur.sample_minmax(p_mid)
                cond = ts.vec(min_val < q_nxt[I] < max_val)

                for k in ti.static(range(cond.n)):
                    if not cond[k]:
                        q_nxt[I][k] = q_mid[k]
                # q_nxt[I] = q_mid


    @ti.kernel
    def advect(self,
               vec_field: ti.template(),
               q_cur: ti.template(),
               q_nxt: ti.template(),
               # boundarySdf: Matrix,
               dt: ti.template()):
        self.advect_func(vec_field, q_cur, q_nxt,
                         dt)
