import taichi as ti
import taichi_glsl as ts
from .Sampler import LinearSampler2D, LinearSampler3D
from .Grid import Grid, GRIDTYPE
from .CellGrid import CellGrid
from utils import Wrapper

@ti.data_oriented
class FaceGrid(Grid):
    """
    wrapper for data on face center (e.g. velocity on MAC grid)
    - F -
    F C F
    - F -
    """

    def __init__(self, dtype, shape, dim, dx, o=ts.vec3(0.5)):
        super(FaceGrid, self).__init__(dim, dx, o)

        self.GRID_TYPE = GRIDTYPE.FACE_GRID

        self.shape = shape
        self.fields = []
        for d in range(dim):
            res = ti.Vector(shape)
            res[d] += 1
            tmp_o = ts.vecND(self.dim, 0.5)
            tmp_o[d] = 0.0
            # use cell grid as the wrapper for u, v, w component
            self.fields.append(
                CellGrid(
                    ti.Vector.field(1, dtype=dtype, shape=res),
                    dim,
                    dx=dx,
                    o=tmp_o
                )
            )

    @ti.pyfunc
    def __getitem__(self, I):
        """
        the index should be based on the whole resolution, not each face dimension
        so here we easily interpolate from the mac grid
        :param I:
        :return:
        """
        return self.interpolate(self.getW(I))

    @ti.pyfunc
    def __setitem__(self, I, value):
        raise NotImplementedError

    @ti.pyfunc
    def __iter__(self):
        for I in ti.grouped(ti.ndrange(*self.shape)):
            yield I

    @ti.pyfunc
    def fill(self, value):
        for i, f in enumerate(self.fields):
            f.fill(ts.vec(value[i]))

    @ti.pyfunc
    def interpolate(self, P):
        ret = ts.vecND(self.dim, 0.0)
        for d in ti.static(range(self.dim)):
            # grid coordinate on each dimension grid
            ret[d] = self.fields[d].interpolate(P)[0]
        return ret

    @ti.pyfunc
    def sample(self, I):
        raise NotImplementedError

    @ti.pyfunc
    def sample_minmax(self, W):
        raise NotImplementedError

    @ti.kernel
    def copy(self, src: Wrapper):
        """
        should
        :param src:
        :return:
        """
        assert (self.shape == src.shape)
        for d in ti.static(range(self.dim)):
            for I in ti.static(self.fields[d]):
                self.fields[d][I] = src.fields[d][I]

    @ti.kernel
    def subself(self, src: Wrapper):
        """

        :param src:
        :return:
        """
        assert (self.shape == src.shape)
        for d in ti.static(range(self.dim)):
            for I in ti.static(self.fields[d]):
                self.fields[d][I] = src.fields[d][I] - self.fields[d][I]

    @ti.pyfunc
    def clampPos(self, P):
        """
        should always called in Cell Grid
        :param P: world pos
        :return:
        """
        raise DeprecationWarning
        raise NotImplementedError
