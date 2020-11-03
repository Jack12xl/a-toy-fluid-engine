import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from geometry import Transform2, Velocity2
from utils import Vector

@ti.data_oriented
class GridEmitter(metaclass=ABCMeta):
    """
    An emitter: produce density(D), velocity(V) to the scene
    """

    def __init__(self,
                 # datagrid,
                 cfg,
                 t: Transform2,
                 v: Velocity2):
        """

        :param cfg:
        :param t: self transform
        :param v: self velocity
        """
        # self.grid = datagrid
        self.cfg = cfg
        self.v = v
        self.t = t

    def kern_materialize(self):
        pass

    @abstractmethod
    def stepEmitHardCode(self,
                         vf,
                         df):
        """
        Emit Velocity and Density and hard code
        :param vf: velocity field
        :param df: density field
        :return:
        """
        pass

    @abstractmethod
    def stepEmitForce(self,
                      vf,
                      df
                      ):
        """
        Emit by force
        :param vf: velocity field
        :param df: density field
        :return:
        """
        pass




@ti.data_oriented
class ForceEmitter(GridEmitter):
    """
    Simulate the velocity and density by force
    """

    def __init__(self,
                 # datagrid,
                 t: Transform2,
                 v: Velocity2,
                 cfg,
                 force_radius,
                 ):
        """

        :param datagrid:
        :param cfg:
        :param t:
            orientation : emit direction
            scale : emit strength
            translation : self position
        :param v:

        """
        super().__init__(
            # datagrid,
            cfg, t, v)

        if force_radius == None:
            self.inv_force_radius = 1.0 / force_radius
        else:
            self.inv_force_radius = 1.0 / (cfg.res[0] / 3.0)

    @ti.kernel
    def stepEmitHardCode(self,
                         # vf
                         ):
        """
        Left blank on purpose for force emitter
        :return:
        """
        pass

    @ti.kernel
    def stepEmitForce(self,
                      vf: Vector,
                      df: Vector,
                      dt: ti.f32,
                      ):
        """
        ref1 : taichi stable fluid
        ref2 : GPU GEM
        :return:
        """
        emit_force = ts.vec(ti.cos(self.t.orientation), ti.sin(self.t.orientation)) * self.t.localScale
        for I in ti.grouped(vf.field):
            den = df[I]

            d2 = (I + 0.5 - self.t.translation).norm_sqr()
            momentum = ti.exp(- dt * self.inv_force_radius) * emit_force

            vf[I] += momentum
            den += ti.exp(- d2 * self.inv_force_radius)
            df[I] = min(den, self.cfg.fluid_color)