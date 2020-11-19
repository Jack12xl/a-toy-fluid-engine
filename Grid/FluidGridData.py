import taichi as ti
from abc import ABCMeta, abstractmethod
from config import EulerCFG
from utils import bufferPair, Vector, Matrix


@ti.data_oriented
class FluidGridData(metaclass=ABCMeta):
    """
    The abstract class for different grid type
    (cellcentered, facecentered)
    Contains data
    """

    def __init__(self, cfg: EulerCFG):
        self.cfg = cfg
        self.dim = cfg.dim

        self.calVorticity = None
        if self.dim == 2:
            self.calVorticity = self.calVorticity2D
        elif self.dim == 3:
            self.calVorticity = self.calVorticity3D

    @abstractmethod
    def calDivergence(self, vf: ti.template(), vd: ti.template()):
        """
        self-explained
        :param vf: field
        :param vd: field divergence
        :return:
        """
        pass

    @abstractmethod
    def calVorticity2D(self, vf: Matrix):
        pass

    @abstractmethod
    def calVorticity3D(self, vf: Matrix):
        pass

    @abstractmethod
    def subtract_gradient_pressure(self):
        pass

    @abstractmethod
    def materialize(self):
        """
        init some variable, especially for
        ti.field, since after materialize Taichi
        can not add more ti.field
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        reset to initial time
        :return:
        """
        pass