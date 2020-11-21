import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from .Sampler import LinearSampler2D, LinearSampler3D


@ti.data_oriented
class Grid(metaclass=ABCMeta):
    """
    the abstract class for the wrapper
     that stores the data
    """

    def __init__(self,
                 dim,
                 dx=ts.vec3(1.0),
                 o=ts.vec3(0.0)):
        """

        :param dim:  dimension of the grid, expected to be 2 or 3
        :param dx:  the physical length of a cell
        :param o: offset on grid
        """
        self.dim = dim
        # should not be zero
        # or too small
        self.dx = dx
        self.inv_dx = 1.0 / self.dx
        self.o = o

        self._sampler = None
        if dim == 2:
            self._sampler = LinearSampler2D()
        elif dim == 3:
            self._sampler = LinearSampler3D()
        else:
            raise NotImplemented
        pass

    @abstractmethod
    def __getitem__(self, I):
        pass

    @abstractmethod
    def __setitem__(self, I, value):
        pass

    # @property
    # def shape(self):
    #     pass

    @abstractmethod
    def fill(self, value):
        pass

    @abstractmethod
    def interpolate(self, P):
        pass
