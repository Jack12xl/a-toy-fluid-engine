import taichi as ti
from enum import Enum
from .AbstractSolver import AdvectionSolver

class SemiLagrangeOrder(Enum):
    RK_1 = 1
    RK_2 = 2
    RK_3 = 3

@ti.data_oriented
class SemiLagrangeSolver(AdvectionSolver):

    def __init__(self, cfg, intpltr):
        super().__init__(cfg)
        self.RK = cfg.semi_order
        self.interpolator = intpltr

    @ti.func
    def backtrace(self,
                  vel_field: ti.template(),
                  I,
                  dt) :
        # TODO abstract grid

        # position in cell center grid
        p = ( I  + 0.5 ) * self.cfg.dx

        if ti.static(self.RK == SemiLagrangeOrder.RK_1):
            p -= dt * vel_field[I]
        elif ti.static(self.RK == SemiLagrangeOrder.RK_2):
            mid_p = p - 0.5 * dt * vel_field[I]
            p -= dt * self.interpolator.interpolate_value(vel_field, mid_p)
        elif ti.static(self.RK == SemiLagrangeOrder.RK_3):
            v1 = vel_field[I]
            p1 = p - 0.5 * dt * v1

            v2 = self.interpolator.interpolate_value(vel_field, p1)
            p2 = p - 0.75 * dt * v2

            v3 = self.interpolator.interpolate_value(vel_field, p2)
            p -= dt * ( 2.0 / 9.0 * v1 + 1.0 / 3.0 * v2 + 4.0 / 9.0 * v3 )

        # TODO boundary handling

        return p

    @ti.kernel
    def advect(self,
               vec_field: ti.template(),
               q_cur: ti.template(),
               q_nxt: ti.template(),
               dt: ti.template() ):

        for I in ti.grouped(vec_field):
            # get predicted position
            coord = self.backtrace(vec_field, I, dt)
            # sample its speed
            q_nxt[I] = self.interpolator.interpolate_value(q_cur, coord)

