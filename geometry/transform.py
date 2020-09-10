import taichi as ti
import math
from utils import Vector, Matrix, tiNormalize
from config.base_cfg import error


## unity gameobject.transform
# ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi/blob/master/src/python/geometry.py
@ti.data_oriented
class Transform2:
    def __init__(self, translation=ti.Vector([0.0, 0.0]), orientation = 0.0, localscale = 1.0):
        self._translation = translation
        self._orientation = orientation % (2 * math.pi)
        self._localscale = localscale

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation:ti.Vector):
        self._translation = translation

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation % (2 * math.pi)

    @property
    def localScale(self):
        return self._localscale

    @localScale.setter
    def localScale(self, localscale):
        self._localscale = max(localscale , error)

    @ti.func
    def to_local(self, p_world:Vector ) -> Vector:
        # translate
        out = p_world - self.translation
        # rotate back
        out = apply_rot(-self.orientation, out)
        # scale
        out /=  self.localScale
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
def test_rotate():
    a = Transform2()
    a.orientation = ti.static(math.pi / 2)
    a.orientation = ti.static(math.pi)
    b = ti.Vector([0, 1])
    # print(a.to_local(b))

    print(a.to_local(b))

if __name__ == '__main__':
    a = Transform2(20, 15)
    a.orientation = 100
    a.localScale = 2
    print(a.orientation)
    print(a.localScale)

    ti.init(ti.cpu, debug=True)
    test_rotate()