import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from .Sampler import LinearSampler2D, LinearSampler3D

@ti.data_oriented
class DataGrid(metaclass=ABCMeta):
    '''
    the abstract class to store data,
    a wrapper for ti.field

    '''
    def __init__(self,
                 data_field:ti.template(), dim=3):
        self._field = data_field
        # self.dim = len(self.shape)

        if dim == 2:
            self._sampler = LinearSampler2D(self.field)
        elif dim == 3:
            self._sampler = LinearSampler3D(self.field)
        else:
            raise NotImplemented

    @ti.pyfunc
    def __getitem__(self, I):
        return self.field[I]

    @ti.pyfunc
    def __setitem__(self, I, value):
        self.field[I] = value

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
        '''
        use bilinear to sample on position P(could be float)
        :param P:
        :return:
        '''

        return self._sampler.lerp(P)

    @ti.pyfunc
    def sample(self, I):
        return ts.sample(self.field, I)

    @ti.pyfunc
    def sample_minmax(self, P):

        return self._sampler.sample_minmax(P)

    @ti.pyfunc
    def fill(self, value):
        self.field.fill(value)
