import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod

@ti.data_oriented
class GridEmitter(metaclass=ABCMeta):
    """
    An emitter: produce density(D), velocity(V) to the scene
    """

    def __init__(self,
                 # datagrid,
                 # cfg,
                 t,
                 v,
                 fluid_color):
        """

        :param cfg:
        :param t: self transform
        :param v: self velocity
        """
        # self.grid = datagrid
        # self.cfg = cfg
        self.v = v
        self.t = t
        self.fluid_color = fluid_color

    def kern_materialize(self):
        self.v.kern_materialize()
        self.t.kern_materialize()
        pass

    @abstractmethod
    def stepEmitHardCode(self,
                         vf,
                         df,
                         dt):
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
        """
        Emit by force
        :param vf: velocity field
        :param df: density field
        :return:
        """
        pass




