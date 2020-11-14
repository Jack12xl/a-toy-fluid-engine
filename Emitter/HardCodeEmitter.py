import taichi as ti
import taichi_glsl as ts
from .GridEmitter import GridEmitter
from geometry import Transform2, Velocity2
from geometry import Transform3, Velocity3
from utils import Vector, Matrix


@ti.data_oriented
class SquareEmitter2D(GridEmitter):
    """
    Directly HardCode the velocity and Density in an area
    """

    def __init__(self,
                 t,
                 v,
                 fluid_color):
        """
        
        :param t:
            translation: center position
            orientation: velocity direction

        :param v: 
        :param fluid_color: 
        """
        super(SquareEmitter2D, self).__init__(t, v, fluid_color)

    @ti.kernel
    def stepEmitForce(self,
                      vf: Matrix,
                      df: Matrix,
                      dt: ti.f32
                      ):
        """
        Explicitly left blank for HardCodeEmitter
        :param vf:
        :param df:
        :param dt:
        :return:
        """
        pass

    @ti.kernel
    def stepEmitHardCode(self, vf: Matrix, df: Matrix):
        """
        Hard code Velocity and Density in an area,
        A square this time
        :param vf:
        :param df:
        :return:
        """

        l_b = self.t.translation - self.t.localScale
        r_u = self.t.translation + self.t.localScale
        shape = ti.Vector(vf.shape)

        l_b = ti.cast(ts.clamp(l_b, 0, shape - 1), ti.i32)
        r_u = ti.cast(ts.clamp(r_u, 0, shape - 1), ti.i32)

        for I in ti.grouped(ti.ndrange((l_b.x, r_u.x), (l_b.y, r_u.y))):
            vf[I] = ts.vec2(0.0, 128.0)
            # here CFL u * dt / dx
            # vf * 0.03 / 1
            # should be 1 ~ 10
            df[I] = self.fluid_color
