import taichi as ti
from .transform import Transform2
from abc import ABCMeta , abstractmethod
from utils import Vector, EuclideanDistance

# implementation ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi/blob/master/src/python/
# ref: << Fluid Engine Development>>
@ti.data_oriented
class Surface(metaclass=ABCMeta):
# topmost class in geometry
    def __init__(self, transform: Transform2 = Transform2(), is_normal_flipped:bool = False):
        self.transform = transform
        self.is_normal_flipped = is_normal_flipped

    @ti.func
    def closest_point(self, world_p:Vector) -> Vector :
        return self.transform.to_world(self.closest_point_local(self.transform.to_local(world_p)))

    @abstractmethod
    def closest_point_local(self, local_p) -> Vector:
        '''
        in local space given local point, return the closest point to it
        :param point:
        :return:
        '''
        pass

    @ti.func
    def closest_normal(self, world_p:Vector) -> Vector :
        out = self.transform.dir_2world(self.closest_point_normal(self.transform.to_local(world_p) ))
        if (ti.static(self.is_normal_flipped)):
            out *= -1.0
        return out

    @abstractmethod
    def closest_point_normal(self, local_p:Vector ) -> Vector:
        pass

    @ti.func
    def closest_distance(self, world_p:Vector) -> Vector:
        closest_p = self.closest_point(world_p)
        return EuclideanDistance(world_p, closest_p)