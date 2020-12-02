import taichi as ti
from enum import Enum, IntEnum
from .AbstractAdvectionSolver import AdvectionSolver
from utils import Vector, Matrix
from config import PixelType


@ti.data_oriented
class SemiLagrangeSolver(AdvectionSolver):

    def __init__(self, cfg, intpltr, bdrySdf, pixel_marker):
        super().__init__(cfg, intpltr, bdrySdf, pixel_marker)

    @ti.func
    def advect_func(self,
                    vec_field: ti.template(),
                    q_cur: ti.template(),
                    q_nxt: ti.template(),
                    dt: ti.template()):
        """

        :param vec_field: the raw field
        :param q_cur:
        :param q_nxt:
        :param dt:
        :return:
        """
        for I in ti.grouped(q_cur):
            # if self.pixel_marker[I] != PixelType.Liquid:
            #     continue
            # get predicted position
            p = q_cur.getW(I)
            coord = self.backtrace(vec_field, p, dt)
            # sample its speed
            q_nxt[I] = q_cur.interpolate(coord)

        return q_nxt

    @ti.kernel
    def advect(self,
               vec_field: ti.template(),
               q_cur: ti.template(),
               q_nxt: ti.template(),
               # boundarySdf: Matrix,
               dt: ti.template()):
        self.advect_func(vec_field, q_cur, q_nxt,
                         dt)
