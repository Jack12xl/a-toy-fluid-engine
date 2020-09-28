import taichi as ti
from utils import Vector, Matrix, Float
from config.base_cfg import error

@ti.data_oriented
class Velocity2:

    @property
    def v_world(self) -> Vector:
        return self._v_world

    @v_world.setter
    def v_world(self, _v_world):
        self._v_world = _v_world

    @property
    def w_centroid(self) -> Float:
        return self._w_centroid

    @w_centroid.setter
    def w_centroid(self, _w_centroid:Float):
        self._w_centroid = _w_centroid

    def __init__(self,
                 velocity_to_world :Vector= ti.Vector([0,0]),
                 angular_velocity_to_centroid:Float = 0.0):
        self.v_world = velocity_to_world
        self.w_centroid = angular_velocity_to_centroid

    def __repr__(self):
        return '{} ( v_wrd : {}, w_cntr: {} )'.format(self.__class__.__name__, self.v_world, self.w_centroid)

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

if __name__ == '__main__':
    t = Velocity2(velocity_to_world=ti.Vector([2.0, 2.0]), angular_velocity_to_centroid=3.0)
    t.v_world = ti.Vector([3.0, 3.0])
    print(t)