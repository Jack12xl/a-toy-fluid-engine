import taichi as ti
import taichi_glsl as ts
from abc import abstractmethod, ABCMeta


@ti.data_oriented
class Sampler(metaclass=ABCMeta):
    def __init__(self, field):
        self.field = field
        pass

    @abstractmethod
    def lerp(self, P):
        pass

    @abstractmethod
    def sample_minmax(self, P):
        pass


@ti.data_oriented
class LinearSampler2D(Sampler):
    def __init__(self, field):
        super(LinearSampler2D, self).__init__(field)

    @ti.pyfunc
    def lerp(self, P):
        # I = int(P)
        # w0 = ts.fract(P)
        # w1 = 1.0 - w0
        # return (self.sample(I + ts.D.xx) * w0.x * w0.y +
        #        self.sample(I + ts.D.xy) * w0.x * w1.y +
        #        self.sample(I + ts.D.yy) * w1.x * w1.y +
        #        self.sample(I + ts.D.yx) * w1.x * w0.y)
        return ts.bilerp(self.field, P)

    @ti.pyfunc
    def sample_minmax(self, P):
        I = ti.floor(P)

        a = ts.sample(self.field, I + ts.D.xx)
        b = ts.sample(self.field, I + ts.D.xy)
        c = ts.sample(self.field, I + ts.D.yy)
        d = ts.sample(self.field, I + ts.D.yx)

        return min(a, b, c, d), max(a, b, c, d)


@ti.data_oriented
class LinearSampler3D(Sampler):
    def __init__(self, field):
        super(LinearSampler3D, self).__init__(field)

    @ti.pyfunc
    def lerp(self, P):
        I = int(P)
        w0 = ts.fract(P)
        w1 = 1.0 - w0

        c00 = ts.sample(self.field, I + ts.D.yyy) * w1.x + ts.sample(self.field, I + ts.D.xyy) * w0.x
        c01 = ts.sample(self.field, I + ts.D.yyx) * w1.x + ts.sample(self.field, I + ts.D.xyx) * w0.x
        c10 = ts.sample(self.field, I + ts.D.yxy) * w1.x + ts.sample(self.field, I + ts.D.xxy) * w0.x
        c11 = ts.sample(self.field, I + ts.D.xyy) * w1.x + ts.sample(self.field, I + ts.D.xxx) * w0.x

        c0 = c00 * w1.y + c10 * w0.y
        c1 = c01 * w1.y + c11 * w0.y

        return c0 * w1.z + c1 * w1.z

    @ti.pyfunc
    def sample_minmax(self, P):
        I = ti.floor(P)

        a = ts.sample(self.field, I + ts.D.xxx)
        b = ts.sample(self.field, I + ts.D.xxy)
        c = ts.sample(self.field, I + ts.D.xyx)
        d = ts.sample(self.field, I + ts.D.yxx)

        e = ts.sample(self.field, I + ts.D.yyy)
        f = ts.sample(self.field, I + ts.D.xyy)
        g = ts.sample(self.field, I + ts.D.yyx)
        h = ts.sample(self.field, I + ts.D.yxy)

        return min(a, b, c, d, e, f, g, h), max(a, b, c, d, e, f, g, h)