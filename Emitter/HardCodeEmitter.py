import taichi as ti
import taichi_glsl as ts
from .GridEmitter import GridEmitter
from utils import Vector, Matrix
from Grid import GRIDTYPE


@ti.data_oriented
class SquareEmitter(GridEmitter):
    """
    Directly HardCode the
    velocity
    Density
    Temperature
    in an area
    Support both 2D and 3D now
    """

    def __init__(self,
                 t,
                 v,
                 jet_v,
                 jet_t,
                 fluid_color,
                 v_grid_type=GRIDTYPE.CELL_GRID):
        """
        
        :param t:
            translation: center position
            orientation: velocity direction

        :param v: self moving velocity
        :param fluid_color: 
        """
        super(SquareEmitter, self).__init__(t, v, fluid_color, v_grid_type)
        self.jet_v = jet_v
        self.jet_t = jet_t

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
    def stepEmitHardCodeCell(self, vf: Matrix, df: Matrix, tf: Matrix):
        """
        Hard code Velocity and Density in an area, for cell grid
        :param tf:
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
            vf[I] = self.jet_v
            tf[I] = ts.vec(self.jet_t)
            # here CFL u * dt / dx
            # vf * 0.03 / 1
            # should be 1 ~ 10
            df[I] = self.fluid_color

    @ti.kernel
    def stepEmitHardCodeFace(self, vf: Matrix, df: Matrix, tf: Matrix):
        """
        Hard code Velocity and Density in an area,
        :param tf:
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

        for d in ti.static(range(len(shape))):
            for I in ti.grouped(ti.ndrange(*r)):
                vf.fields[d][I][0] = self.jet_v[d]

        for I in ti.grouped(ti.ndrange(*r)):
            tf[I] = ts.vec(self.jet_t)
            # here CFL u * dt / dx
            # vf * 0.03 / 1
            # should be 1 ~ 10
            df[I] = self.fluid_color
