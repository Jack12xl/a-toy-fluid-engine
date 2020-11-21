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
        self.dx = dx
        self.o = o
        pass

    @abstractmethod
    def __getitem__(self, I):
        pass

    @abstractmethod
    def __setitem__(self, I, value):
        pass

    @property
    def shape(self):
        pass

    @abstractmethod
    def fill(self, value):
        pass
