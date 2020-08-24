from src import AdvectionSolver, Grid
import taichi as ti
from enum import Enum


class Order(Enum):
    RK_1 = 1
    RK_2 = 2
    RK_3 = 3

@ti.data_oriented
class SemiLagrangeSolver(AdvectionSolver):

    def __init__(self, cfg, intpltr: Grid):
        super().__init__(cfg)
        self.RK = cfg.RK
        self.interpolator = intpltr

    @ti.func
    def backtrace(self,
                  vel_field: ti.template(),
                  I: ti.Vector) :
        # TODO abstract grid

        # position in cell center grid
        p = ( I  + 0.5 ) * self.cfg.dx

        if ti.static(self.RK == Order.RK_1):
            p -= self.cfg.dt * vel_field[I]
        elif ti.static(self.RK == Order.RK_2):
            mid_p = p - 0.5 * self.cfg.dt * vel_field[I]
            p -= self.cfg.dt * self.interpolator.interpolate_value(vel_field, mid_p)
        elif ti.static(self.RK == Order.RK_3):
            v1 = vel_field[I]
            p1 = p - 0.5 * self.cfg.dt * v1

            v2 = self.interpolator.interpolate_value(vel_field, p1)
            p2 = p - 0.75 * self.cfg.dt * v2

            v3 = self.interpolator.interpolate_value(vel_field, p2)
            p -= self.cfg.dt * ( 2.0 / 9.0 * v1 + 1.0 / 3.0 * v2 + 4.0 / 9.0 * v3 )

        # TODO boundary handling

        return p

    @ti.kernel
    def advect(self,
               vec_field: ti.template(),
               q_cur: ti.template(),
               q_nxt: ti.template()):

        for I in ti.grouped(vec_field):
            # get predicted position
            coord = self.backtrace(vec_field, I)
            # sample its speed
            q_nxt[I] = self.interpolator.interpolate_value(q_cur, coord)

