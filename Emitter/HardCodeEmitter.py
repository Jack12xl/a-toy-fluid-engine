import taichi as ti
import taichi_glsl as ts
from .GridEmitter import GridEmitter
from geometry import Transform2, Velocity2
from geometry import Transform3, Velocity3
from utils import Vector, Matrix


@ti.data_oriented
class SquareEmitter(GridEmitter):
    """
    Directly HardCode the velocity and Density in an area
    Support both 2D and 3D now
    """

    def __init__(self,
                 t,
                 v,
                 jit_v,
                 fluid_color):
        """
        
        :param t:
            translation: center position
            orientation: velocity direction

        :param v: self moving velocity
        :param fluid_color: 
        """
        super(SquareEmitter, self).__init__(t, v, fluid_color)
        self.jit_v = jit_v

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

        l_b = ts.clamp(l_b, 0, shape - 1)
        r_u = ts.clamp(r_u, 0, shape - 1)

        r = [(int(l_b[i]), int(r_u[i]) + 1) for i in range(len(l_b))]

        for I in ti.grouped(ti.ndrange(*r)):
            vf[I] = self.jit_v
            # here CFL u * dt / dx
            # vf * 0.03 / 1
            # should be 1 ~ 10
            df[I] = self.fluid_color

