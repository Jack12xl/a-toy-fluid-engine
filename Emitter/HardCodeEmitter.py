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
        super(SquareEmitter, self).__init__(t, v, fluid_color)

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
    def stepEmitHardCode(self,
                         vf: Matrix,
                         df: Matrix,
                         dt: ti.f32):
        """
        Hard code Velocity and Density in an area,
        A square this time
        :param vf:
        :param df:
        :param dt:
        :return:
        """

        for I in ti.grouped(vf.field):
            pass


