import taichi as ti
import taichi_glsl as ts
import math
from utils import Vector, Matrix, tiNormalize, Float
from config.base_cfg import error


class Transform3:
    def __init__(self,
                 translation=ts.vec3(0.0, 0.0, 0.0),
                 orientation=ts.vec2(0.0, 0.0),
                 localscale=ts.vec3(1.0)):
        """

        :param translation:
        :param orientation: theta : [0, pi], phi : [0, 2*pi]
        :param localscale:
        """
        self._translation = ti.Vector.field(3, dtype=ti.f32, shape=[])
        self._orientation = ti.Vector.field(2, dtype=ti.f32, shape=[])
        self._localScale = ti.Vector.field(3, dtype=ti.f32, shape=[])
        # use buffer for later materialization
        self.translation_buf = translation
        self.orientation_buf = orientation % (2 * math.pi)
        self.localscale_buf = localscale

    def __repr__(self):
        return '{} ( Trsln : {}, Ornttn: {}, lclScl: {})'.format(
            self.__class__.__name__,
            self.translation,
            self.orientation,
            self.localScale)

    @ti.pyfunc
    def kern_materialize(self):
        self._translation[None] = self.translation_buf
        self._orientation[None] = self.orientation_buf
        self._localScale[None] = self.localscale_buf

    @property
    @ti.pyfunc
    def translation(self) -> Vector:
        return self._translation[None]

    @translation.setter
    def translation(self, translation: ti.Vector):
        self._translation[None] = translation

    @property
    @ti.pyfunc
    def orientation(self) -> Float:
        return self._orientation[None]

    @orientation.setter
    def orientation(self, orientation: Vector):
        tmp = ts.vec2(0.0)
        tmp[0] = orientation[0] % math.pi
        tmp[1] = orientation[1] % (2 * math.pi)
        self._orientation[None] = tmp

    @property
    @ti.pyfunc
    def localScale(self) -> Float:
        return self._localScale[None]

    @localScale.setter
    def localScale(self, localScale: Vector):
        # clamp above zero
        self._localScale[None] = max(localScale, error)

    # TODO other method
