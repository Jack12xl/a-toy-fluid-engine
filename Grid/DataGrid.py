import taichi as ti
import taichi_glsl as ts
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
        # self.dim = len(self.shape)
        #TODO currently borrow from taichi_glsl
        self._sampler = None

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
    def LinearlyLerp(self, P):
        '''
        use bilinear to sample on position P(could be float)
        :param P:
        :return:
        '''
        #TODO support 3D
        I = int(P)
        w0 = ts.fract(P)
        w1 = 1 - w0
        dim = ti.static(len(self.shape))

        ret = ts.vecND(dim, 0.0)
        # for d in ti.static(range(self.dim)):
        if dim == 2:
            ret =  (self.sample(I + ts.D.xx) * w0.x * w0.y +
                    self.sample(I + ts.D.xy) * w0.x * w1.y +
                    self.sample(I + ts.D.yy) * w1.x * w1.y +
                    self.sample(I + ts.D.yx) * w1.x * w0.y)
        elif dim == 3:
            c00 = ts.sample(self.field, I + ts.D.yyy) * w1.x + ts.sample(self.field, I + ts.D.xyy) * w0.x
            c01 = ts.sample(self.field, I + ts.D.yyx) * w1.x + ts.sample(self.field, I + ts.D.xyx) * w0.x
            c10 = ts.sample(self.field, I + ts.D.yxy) * w1.x + ts.sample(self.field, I + ts.D.xxy) * w0.x
            c11 = ts.sample(self.field, I + ts.D.xyy) * w1.x + ts.sample(self.field, I + ts.D.xxx) * w0.x

            c0 = c00 * w1.y + c10 * w0.y
            c1 = c01 * w1.y + c11 * w0.y

            ret = c0 * w1.z + c1 * w1.z
        else:
            raise ValueError

        return ret
        # return ts.bilerp(self.field, P)

    @ti.pyfunc
    def sample(self, I):
        return ts.sample(self.field, I)

    @ti.pyfunc
    def sample_minmax(self, P):
        I = int(P)
        x = ts.fract(P)
        y = 1 - x

        a = ts.sample(self.field, I + ts.D.xx) * x.x * x.y
        b = ts.sample(self.field, I + ts.D.xy) * x.x * y.y
        c = ts.sample(self.field, I + ts.D.yy) * y.x * y.y
        d = ts.sample(self.field, I + ts.D.yx) * y.x * x.y

        return min(a, b, c, d), max(a, b, c, d)

    @ti.pyfunc
    def fill(self, value):
        self.field.fill(value)
