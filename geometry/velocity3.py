import taichi as ti
import taichi_glsl as ts
from utils import Vector, Matrix, Float


@ti.data_oriented
class Velocity3:
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
    def w_centroid(self, _w_centroid: Vector):
        self._w_centroid[None] = _w_centroid

    def __init__(self,
                 velocity_to_world: Vector = ts.vec3(0.0),
                 angular_velocity_to_centroid: Vector = ts.vec2(0.0)):
        self._v_world = ti.Vector.field(3, dtype=ti.f32, shape=[])
        self._w_centroid = ti.Vector.field(3, dtype=ti.f32, shape=[])

        self.v_world_buf = velocity_to_world
        self.w_centroid_buf = angular_velocity_to_centroid

    def __repr__(self):
        return '{} ( v_wrd : {}, w_cntr: {} )'.format(self.__class__.__name__, self.v_world, self.w_centroid)

    @ti.pyfunc
    def kern_materialize(self):
        self._v_world[None] = self.v_world_buf
        self._w_centroid[None] = self.w_centroid_buf
