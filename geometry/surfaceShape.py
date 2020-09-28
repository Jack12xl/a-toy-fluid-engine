from .surface import Surface
from .transform import Transform2
import taichi as ti
from basic_types import Vector, Matrix, Float
from utils import tiNormalize
from abc import ABCMeta , abstractmethod

@ti.data_oriented
class SurfaceShape(Surface):
    '''
    class with velocity, mass
    more of interface for an instance
    '''
    @property
    def velocity(self) -> Vector:
        '''
        velocity w.r.t center of mass
        :return:
        '''
        return self._velocity

    @velocity.setter
    def velocity(self, _v: Vector):
        self._velocity = _v

    @property
    def omega(self):
        return self._angular_velocity

    @omega.setter
    def omega(self, angular_velocity):
        self._angular_velocity = angular_velocity

    @property
    def mass(self) -> ti.f32:
        return self._mass

    @mass.setter
    def mass(self, _m):
        self._mass = max(_m, 0.0001)

    def __init__(self, transform: Transform2 = Transform2(), is_normal_flipped:bool = False, mass:ti.f32= 1.0, angular_velocity:ti.f32 = 0.0):
        super(SurfaceShape, self).__init__(transform, is_normal_flipped)
        self.velocity = ti.Vector([0.0, 0.0])
        self.omega = angular_velocity
        self.mass = mass

    @abstractmethod
    def velocity_at_local_point(self, local_point: Vector):
        pass

    @abstractmethod
    def color_at_local_point(self, local_point: Vector):
        pass

@ti.data_oriented
class Ball(SurfaceShape):
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

    def __init__(self,
                 transform: Transform2 = Transform2(),
                 is_normal_flipped:bool = False,
                 mass:ti.f32 = 1.0,
                 angular_velocity: ti.f32= 0.0):
        super(Ball, self).__init__(
            transform,
            is_normal_flipped,
            mass,
            angular_velocity
        )

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

    @ti.func
    def velocity_at_local_point(self, local_point: Vector):
        #TODO
        return ti.Vector([0.0, 0.0])

    @ti.func
    def color_at_local_point(self, local_point: Vector) -> Vector:
        c = ti.Vector([0.0, 0.0, 0.0])
        if (ti.abs(local_point[1]) < ti.static(0.1) and local_point[0] > ti.static(0.0) ):
            c = ti.Vector([0.7, 0.2, 0.2])
        else:
            c = ti.Vector([0.9, 0.9, 0.9])
        return c


