import taichi as ti
import taichi_glsl as ts
from .GridEmitter import GridEmitter
from geometry import Transform2, Velocity2
from geometry import Transform3, Velocity3
from utils import Vector, Matrix


@ti.data_oriented
class ForceEmitter2(GridEmitter):
    """
    Simulate the velocity and density by force
    """

    def __init__(self,
                 # datagrid,
                 # cfg,
                 t: Transform2,
                 v: Velocity2,
                 fluid_color,
                 force_radius
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
            # cfg,
            t, v, fluid_color)

        self.inv_force_radius = 1.0 / (force_radius + 0.001)

    @ti.kernel
    def stepEmitHardCode(self,
                         vf: Matrix,
                         df: Matrix
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
            # add 0.5 can get less artifacts... strange
            factor = ti.exp(- d2 * self.inv_force_radius)
            momentum = (factor * emit_force) * dt

            vf[I] += momentum
            den += factor * self.fluid_color
            df[I] = min(den, self.fluid_color)


@ti.data_oriented
class ForceEmitter3(GridEmitter):
    def __init__(self,
                 # cfg,
                 t: Transform3,
                 v: Velocity3,
                 fluid_color,
                 force_radius,
                 ):
        """

                :param cfg:
                :param t: self transform
                :param v: self velocity
        """
        super(ForceEmitter3, self).__init__(t,
                                            v,
                                            fluid_color=fluid_color)

        self.inv_force_radius = 1.0 / (force_radius + 0.0001)

    @ti.kernel
    def stepEmitForce(self,
                      vf: Vector,
                      df: Vector,
                      dt: ti.f32
                      ):
        """
        3D
        :param vf:
        :param df:
        :param dt:
        :return:
        """

        # TODO
        theta = self.t.orientation[0]
        phi = self.t.orientation[1]
        emit_force = ts.vec3(
            ti.sin(phi) * ti.cos(theta),
            ti.sin(phi) * ti.sin(theta),
            ti.cos(phi)
        ) * self.t.localScale

        for I in ti.grouped(vf.field):
            den = df[I]

            d2 = (I + 0.5 - self.t.translation).norm_sqr()
            factor = ti.exp(- d2 * self.inv_force_radius)
            momentum = factor * emit_force * dt

            vf[I] += momentum
            den += factor * self.cfg.fluid_color
            df[I] = min(den, self.cfg.fluid_color)

    @ti.kernel
    def stepEmitHardCode(self,
                         vf: Matrix,
                         df: Matrix,
                         ):
        """
        Left blank on purpose for force emitter
        :return:
        """
        pass
