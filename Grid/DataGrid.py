import taichi as ti
import taichi_glsl as ts
from basic_types import Vector, Float
from abc import ABCMeta, abstractmethod

@ti.data_oriented
class DataGrid(metaclass=ABCMeta):
    '''
    the abstract class to store data,
    a wrapper for ti.field

    '''
    def __init__(self,
                 data_field:ti.template()):
        self._field = data_field
        #TODO currently borrow from taichi_glsl
        self._sampler = None

        pass

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
    @ti.func
    def field(self):
        return self._field

    @ti.pyfunc
    def sample(self, I):
        return ts.bilerp(self.field, I)
