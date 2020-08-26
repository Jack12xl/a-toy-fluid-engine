import taichi as ti
from enum import Enum
from .AbstractAdvectionSolver import AdvectionSolver

class SemiLagrangeOrder(Enum):
    RK_1 = 1
    RK_2 = 2
    RK_3 = 3

@ti.data_oriented
class SemiLagrangeSolver(AdvectionSolver):

    def __init__(self, cfg, intpltr):
        super().__init__(cfg)
        self.RK = cfg.semi_order
        self.grid = intpltr

    @ti.func
    def backtrace(self,
                  vel_field: ti.template(),
                  pos,
                  dt) :
        # TODO abstract grid
        '''

        :param vel_field:
        :param pos: input backtrace coordinate in grid
        :param dt:
        :return: backtraced pos
        '''

        # position in cell center grid
        # p = (pos + 0.5) * self.cfg.dx
        if ti.static(self.RK == SemiLagrangeOrder.RK_1):
            pos -= dt * self.grid.interpolate_value(vel_field, pos)
        elif ti.static(self.RK == SemiLagrangeOrder.RK_2):
            mid_p = pos - 0.5 * dt * self.grid.interpolate_value(vel_field, pos)
            pos -= dt * self.grid.interpolate_value(vel_field, mid_p)
        elif ti.static(self.RK == SemiLagrangeOrder.RK_3):
            v1 = self.grid.interpolate_value(vel_field, pos)
            p1 = pos - 0.5 * dt * v1

            v2 = self.grid.interpolate_value(vel_field, p1)
            p2 = pos - 0.75 * dt * v2

            v3 = self.grid.interpolate_value(vel_field, p2)
            pos -= dt * ( 2.0 / 9.0 * v1 + 1.0 / 3.0 * v2 + 4.0 / 9.0 * v3 )

        # TODO boundary handling

        return pos

    @ti.func
    def advect_func(self,
                    vec_field: ti.template(),
                    q_cur: ti.template(),
                    q_nxt: ti.template(),
                    dt: ti.template()):
        for I in ti.grouped(vec_field):
            # get predicted position
            p = ( I  + 0.5 ) * self.cfg.dx
            coord = self.backtrace(vec_field, p, dt)
            # sample its speed
            q_nxt[I] = self.grid.interpolate_value(q_cur, coord)

        return q_nxt
    @ti.kernel
    def advect(self,
               vec_field: ti.template(),
               q_cur: ti.template(),
               q_nxt: ti.template(),
               dt: ti.template() ):
        self.advect_func(vec_field, q_cur, q_nxt, dt)


