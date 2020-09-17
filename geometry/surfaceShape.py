from .surface import Surface
from .transform import Transform2
import taichi as ti
from basic_types import Vector, Matrix, Float
from utils import tiNormalize

@ti.data_oriented
class Ball(Surface):
    @property
    def mid(self):
        return self._mid

    @mid.setter
    def mid(self, _mid):
        self._mid = _mid

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, _radius):
        self._radius = abs(_radius)

    def parse_transform(self, transform: Transform2):
        self.mid = transform.translation
        self.radius = transform.localScale

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = _transform
        self.parse_transform(_transform)

    def __init__(self, transform: Transform2 = Transform2(), is_normal_flipped:bool = False):
        super(Ball, self).__init__(transform, is_normal_flipped)
        self.transform = transform

    @ti.func
    def closest_point_normal_local(self, local_p:Vector) -> Vector:
        return tiNormalize(local_p)

    @ti.func
    def closest_point_local(self, local_p) -> Vector:
        ## TODO if local_p is [0, 0]
        return tiNormalize(local_p)

    @ti.func
    def is_inside_local(self, local_p: Vector) -> bool:
        return local_p.norm() < ti.static(1.0)


