import taichi as ti
from utils import Vector, Matrix, Float
from config.base_cfg import error

@ti.data_oriented
class Velocity2:

    @property
    @ti.pyfunc
    def v_world(self) -> Vector:
        return self._v_world[None]

    @v_world.setter
    def v_world(self, _v_world):
        self._v_world[None] = _v_world

    @property
    @ti.pyfunc
    def w_centroid(self) -> Float:
        return self._w_centroid[None]

    @w_centroid.setter
    def w_centroid(self, _w_centroid:Float):
        self._w_centroid[None] = _w_centroid

    def __init__(self,
                 velocity_to_world :Vector= ti.Vector([0,0]),
                 angular_velocity_to_centroid:Float = 0.0):
        self._v_world = ti.Vector.field(2, dtype=ti.f32, shape=[])
        self._w_centroid = ti.field(dtype=ti.f32, shape=[])

        self.v_world_buf = velocity_to_world
        self.w_centroid_buf = angular_velocity_to_centroid

    def __repr__(self):
        return '{} ( v_wrd : {}, w_cntr: {} )'.format(self.__class__.__name__, self.v_world, self.w_centroid)

    @ti.pyfunc
    def kern_materialize(self):
        self._v_world[None] = self.v_world_buf
        self._w_centroid[None] = self.w_centroid_buf

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

if __name__ == '__main__':
    t = Velocity2(velocity_to_world=ti.Vector([2.0, 2.0]), angular_velocity_to_centroid=3.0)
    t.kern_materialize()
    t.v_world = ti.Vector([3.0, 3.0])
    print(t)