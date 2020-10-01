import taichi as ti
import math
from utils import Vector, Matrix, tiNormalize, Float
from config.base_cfg import error

## unity gameobject.transform
# ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi/blob/master/src/python/geometry.py
@ti.data_oriented
class Transform2:
    def __init__(self,
                 translation=ti.Vector([0.0, 0.0]),
                 orientation = 0.0,
                 localscale = 1.0):
        self._translation = ti.Vector.field(2, dtype=ti.f32, shape = [])
        self._orientation = ti.field(dtype=ti.f32, shape = [])
        self._localScale = ti.field(dtype=ti.f32, shape = [])

        self.translation_buf = translation
        self.orientation_buf = orientation % (2 * math.pi)
        self.localscale_buf = localscale

    def __repr__(self):
        return '{} ( Trsln : {}, Ornttn: {}, lclScl: {})'.format(
            self.__class__.__name__,
            self.translation,
            self.orientation,
            self.localScale)

    @ti.kernel
    def kern_materialize(self):
        self._translation[None] = self.translation_buf
        self._orientation[None] = self.orientation_buf
        self._localScale[None] = self.localscale_buf

    @property
    def translation(self) -> Vector:
        return self._translation[None]

    @property
    @ti.func
    def translation(self) -> Vector:
        return self._translation[None]


    @translation.setter
    def translation(self, translation:ti.Vector):
        self._translation[None] = translation



    @property
    def orientation(self) -> Float:
        return self._orientation[None]

    @orientation.setter
    def orientation(self, orientation:Float):
        self._orientation[None] = orientation % (2 * math.pi)

    @property
    @ti.func
    def orientation(self) -> Float:
        return self._orientation[None]

    @property
    def localScale(self) -> Float:
        return self._localScale[None]

    @property
    @ti.func
    def localScale(self) -> Float:
        return self._localScale[None]

    @localScale.setter
    def localScale(self, localScale:Float):
        # clamp above zero
        self._localScale[None] = max(localScale, error)

    @ti.func
    def to_local(self, p_world:Vector ) -> Vector:
        # translate
        out = float(p_world) - self.translation[None]
        # rotate back
        out = apply_rot(-self.orientation[None], out)
        # scale
        out /=  self.localScale[None]
        return out

    @ti.func
    def to_world(self, p_local:Vector ) -> Vector:
        # scale
        out = p_local * self.localScale
        # rotate
        out = apply_rot(self.orientation, out)
        # translate
        out += self.translation
        return out

    @ti.func
    def dir_2world(self, dir_local:Vector) -> Vector:
        out = apply_rot(self.orientation, dir_local)
        return tiNormalize(out)


@ti.func
def getRotMat2D(rotation)-> Matrix:
    return ti.Matrix([[ti.cos(rotation), -ti.sin(rotation)], [ti.sin(rotation), ti.cos(rotation)]])

@ti.func
def apply_rot(rot, p:Vector) -> Vector:
    cos = ti.cos(rot)
    sin = ti.sin(rot)
    return ti.Vector([cos * p[0] - sin * p[1], sin * p[0] + cos * p[1] ])


@ti.kernel
def test_rotate(a : ti.template()):

    a._orientation = ti.static(math.pi / 2)
    a._orientation = ti.static(math.pi)
    b = ti.Vector([0, 1])
    # print(a.to_local(b))

    print(a.to_local(b))


if __name__ == '__main__':
    ti.init(ti.cpu, debug=True)
    a = Transform2(ti.Vector([2.0, 4.0]), 15)
    a.kern_materialize()
    a.orientation = 100
    a.localScale = 2
    print(a.orientation)
    print(a.localScale)


    test_rotate(a)