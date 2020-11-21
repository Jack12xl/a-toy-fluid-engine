import taichi as ti
import taichi_glsl as ts
from .Sampler import LinearSampler2D, LinearSampler3D
from .Grid import Grid


@ti.data_oriented
class CellGrid(Grid):
    """
    wrapper for data on cell center (e.g. pressure, curl)
    - F -
    F C F
    - F -
    """

    def __init__(self,
                 data_field: ti.template(), dim=3, dx=ts.vec3(1.0), o=ts.vec3(0.0)):
        """

        :param data_field:
        :param dim:
        :param dx: grid length
        :param o: offset
        """
        super(CellGrid, self).__init__(dim, dx, o)

        self._field = data_field


    @ti.pyfunc
    def __getitem__(self, I):
        return self.field[I]

    @ti.pyfunc
    def __setitem__(self, I, value):
        self.field[I] = value

    # @ti.pyfunc
    def loop_range(self):
        return self._field.loop_range()

    @ti.pyfunc
    def __iter__(self):
        for I in ti.grouped(ti.ndrange(*self.shape)):
            yield I

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
        return self._sampler.lerp(self.field, self.getG(P))

    @ti.pyfunc
    def getW(self, G):
        """
        get world position from Grid Coordinate
        :param G:
        :return:
        """
        return (float(G) + self.o) * self.dx

    @ti.pyfunc
    def getG(self, W):
        """

        :param W: physical position
        :return:
        """
        return W * self.inv_dx - self.o

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
        g = self.getG(P)
        return self._sampler.sample_minmax(self.field, g)

    @ti.pyfunc
    def fill(self, value):
        self.field.fill(value)
