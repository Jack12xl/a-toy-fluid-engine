import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from .Sampler import LinearSampler2D, LinearSampler3D


@ti.data_oriented
class DataGrid(metaclass=ABCMeta):
    """
    the abstract class to store data,
    a wrapper for ti.field

    """

    def __init__(self,
                 data_field: ti.template(), dim=3, dx=ts.vec3(1.0), o=ts.vec3(0.0)):
        """

        :param data_field:
        :param dim:
        :param h: grid length
        :param o: offset
        """
        self._field = data_field
        # self.dim = len(self.shape)
        if dim == 2:
            self._sampler = LinearSampler2D(self.field)
        elif dim == 3:
            self._sampler = LinearSampler3D(self.field)
        else:
            raise NotImplemented

        self.dx = dx
        self.inv_dx = 1.0 / dx
        self.o = o

    @ti.pyfunc
    def __getitem__(self, I):
        return self.field[I]

    @ti.pyfunc
    def __setitem__(self, I, value):
        self.field[I] = value

    # @ti.pyfunc
    def loop_range(self):
        return self._field.loop_range()

    @property
    @ti.pyfunc
    def shape(self):
        return self._field.shape

    @property
    @ti.pyfunc
    def field(self):
        return self._field

    @ti.pyfunc
    def interpolate(self, P):
        """
        sample on position P(could be float)
        :param P: coordinate in physical world
        :return: value on grid
        """
        # grid coordinate
        return self._sampler.lerp(self.getG(P))

    @ti.pyfunc
    def getW(self, G):
        """
        get world position from Grid Coordinate
        :param I:
        :return:
        """
        return float(G) * self.dx

    @ti.pyfunc
    def getG(self, W):
        """

        :param W: physical position
        :return:
        """
        return W * self.inv_dx

    @ti.pyfunc
    def sample(self, I):
        """

        :param I:
        :return:
        """
        return ts.sample(self.field, I)

    @ti.pyfunc
    def sample_minmax(self, P):
        """

        :param P: physical coordinate
        :return: grid value
        """
        g = P / self.dx
        return self._sampler.sample_minmax(g)

    @ti.pyfunc
    def fill(self, value):
        self.field.fill(value)
