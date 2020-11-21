import taichi as ti
import taichi_glsl as ts
from .Sampler import LinearSampler2D, LinearSampler3D
from .Grid import Grid


@ti.data_oriented
class FaceGrid(Grid):
    """
    wrapper for data on face center (e.g. velocity on MAC grid)
    - F -
    F C F
    - F -
    """

    def __init__(self, dtype, shape, dim, dx, o):
        super(FaceGrid, self).__init__(dim, dx, o)

        self.field = None
        if dim == 2:
            self._sampler = LinearSampler2D(self.field)
        elif dim == 3:
            self._sampler = LinearSampler3D(self.field)
        else:
            raise NotImplemented

        # self._shape = shape
        self.shape = shape
        self.fields = []
        for i in range(dim):
            res = ti.Vector(shape)
            res[i] += 1
            self.fields.append(ti.Vector.field(1, dtype=dtype, shape=res))

    @ti.pyfunc
    def __getitem__(self, I):
        pass

    @ti.pyfunc
    def __setitem__(self, I, value):
        pass

    # # TODO totally unnecessary
    # @property
    # @ti.pyfunc
    # def shape(self):
    #     return self._shape

    @ti.pyfunc
    def fill(self, value):
        for i, f in enumerate(self.fields):
            f.fill(ts.vec(value[i]))

    @ti.pyfunc
    def interpolate(self, P):
        pass

    @ti.pyfunc
    def __iter__(self):
        for I in ti.grouped(ti.ndrange(*self.shape)):
            yield I
