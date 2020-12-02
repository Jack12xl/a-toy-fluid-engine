from abc import ABCMeta, abstractmethod
from utils import filterUpCase
import taichi as ti
from utils import Vector, Matrix
from enum import Enum, IntEnum
from config import PixelType


class RK_Order(IntEnum):
    RK_1 = 1
    RK_2 = 2
    RK_3 = 3


class AdvectionSolver(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, cfg, intpltr, bdrySdf, pixel_marker):
        self.cfg = cfg
        self.RK = cfg.semi_order
        self.grid = intpltr
        self.bdrySdf = bdrySdf
        self.pixel_marker = pixel_marker

    @abstractmethod
    def advect(self, vec_field, q_cur, q_nxt, dt):
        pass

    def abbreviation(self):
        return filterUpCase(self.__class__.__name__)

    @ti.func
    def backtrace(self,
                  vel_field: Matrix,
                  pos: Vector,
                  # boundarySdf: Matrix,
                  dt):
        """

        :param vel_field:
        :param pos: input backtrace physical world position
        :param dt:
        :return: backtraced pos
        """

        # start
        start_pos = pos

        if ti.static(self.RK == RK_Order.RK_1):
            pos -= dt * vel_field.interpolate(pos)
        elif ti.static(self.RK == RK_Order.RK_2):
            mid_p = pos - 0.5 * dt * vel_field.interpolate(pos)
            pos -= dt * vel_field.interpolate(mid_p)
        elif ti.static(self.RK == RK_Order.RK_3):
            v1 = vel_field.interpolate(pos)
            p1 = pos - 0.5 * dt * v1

            v2 = vel_field.interpolate(p1)
            p2 = pos - 0.75 * dt * v2

            v3 = vel_field.interpolate(p2)
            pos -= dt * (2.0 / 9.0 * v1 + 1.0 / 3.0 * v2 + 4.0 / 9.0 * v3)

        # TODO boundary handling
        # 3.4.2.4
        phi0 = self.bdrySdf.interpolate(start_pos)
        phi1 = self.bdrySdf.interpolate(pos)
        if phi0 * phi1 < 0.0:
            w = ti.abs(phi1) / (ti.abs(phi0) + ti.abs(phi1))
            # pos = w * start_pos + (1.0 - w) * pos
            # why the previous line would trigger:
            # UnboundLocalError: local variable 'pos' referenced before assignment
            pos += w * start_pos
            pos *= (1.0 - w)
        return pos
