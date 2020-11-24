import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from Grid import GRIDTYPE


@ti.data_oriented
class GridEmitter(metaclass=ABCMeta):
    """
    An emitter: produce density(D), velocity(V) to the scene
    """

    def __init__(self,
                 t,
                 v,
                 fluid_color,
                 v_grid_type=GRIDTYPE.CELL_GRID):
        """

        :param t: self transform
        :param v: self velocity
        """
        # self.grid = datagrid
        # self.cfg = cfg
        self.v = v
        self.t = t
        self.fluid_color = fluid_color
        self.V_GRID_TYPE = v_grid_type

        if self.V_GRID_TYPE == GRIDTYPE.CELL_GRID:
            self.stepEmitHardCode = self.stepEmitHardCodeCell
        elif self.V_GRID_TYPE == GRIDTYPE.FACE_GRID:
            self.stepEmitHardCode = self.stepEmitHardCodeFace
        else:
            raise NotImplementedError

    def kern_materialize(self):
        self.v.kern_materialize()
        self.t.kern_materialize()
        pass

    @abstractmethod
    def stepEmitHardCodeCell(self,
                             vf,
                             df
                             ):
        """
        hard code Velocity and Density in an area
        :param vf: velocity field
        :param df: density field
        :return:
        """
        pass

    @abstractmethod
    def stepEmitHardCodeFace(self,
                             vf,
                             df
                             ):
        """
        hard code Velocity and Density in an area
        :param vf: velocity field
        :param df: density field
        :return:
        """
        pass

    @abstractmethod
    def stepEmitForce(self,
                      vf,
                      df,
                      dt
                      ):
        # TODO support for both mac grid and uniform grid
        """
        Emit by force
        :param vf: velocity field
        :param df: density field
        :return:
        """
        pass
