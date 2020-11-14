import taichi as ti
from .transform2 import Transform2
from abc import ABCMeta , abstractmethod
from utils import Vector, Float, EuclideanDistance

# implementation ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi/blob/master/src/python/
# ref: << Fluid Engine Development>>
@ti.data_oriented
class Surface(metaclass=ABCMeta):
# topmost class in geometry
    def __init__(self, transform: Transform2 = Transform2(), is_normal_flipped:bool = False):
        self._transform = transform
        self.is_normal_flipped = is_normal_flipped

    @ti.pyfunc
    def kern_materialize(self):
        self.transform.kern_materialize()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = _transform

    @ti.func
    def closest_point(self, world_p:Vector) -> Vector :
        return self.transform.to_world(self.closest_point_local(self._transform.to_local(world_p)))

    @abstractmethod
    def closest_point_local(self, local_p) -> Vector:
        '''
        in local space given local point, return the closest point to it
        :param local_p:
        :return:
        '''
        pass

    @ti.func
    def closest_normal(self, world_p:Vector) -> Vector :
        out = self.transform.dir_2world(self.closest_point_normal_local(self._transform.to_local(world_p)))
        if (ti.static(self.is_normal_flipped)):
            out *= -1.0
        return out

    @abstractmethod
    def closest_point_normal_local(self, local_p:Vector) -> Vector:
        pass

    @ti.func
    def closest_distance(self, world_p:Vector) -> Vector:
        closest_p = self.closest_point(world_p)
        return EuclideanDistance(world_p, closest_p)

    @abstractmethod
    def is_inside_local(self, local_p: Vector) -> bool:
        pass

    @ti.pyfunc
    def is_inside_world(self, world_p: Vector) -> bool:
        local_p = self.transform.to_local(world_p)
        return self.is_inside_local(local_p)

    @ti.func
    def translateTo(self, offset: Vector):
        self.transform += ti.static(offset)

@ti.data_oriented
class ImplicitSurface(Surface):
    # reference 3.1.4.3
    def __init__(self, transform: Transform2, is_norm_flipped : bool = False):
        super(ImplicitSurface, self).__init__(transform, is_norm_flipped)

    @abstractmethod
    def sign_distance_local(self, local_point: Vector) -> Float:
        pass

    @ti.func
    def signed_distance(self, world_point: Vector) -> Float:
        local_point = self.transform.to_local(world_point)
        return self.transform.localScale * self.sign_distance_local(local_point)

    @ti.func
    def is_inside_local(self, local_p: Vector) -> bool:
        return self.sign_distance_local(local_p) < 0.0

@ti.data_oriented
class SurfaceToImplict(ImplicitSurface):
    def __init__(
            self,
            surface : Surface,
            # transform : Transform2 = Transform2(),
            is_normal_flipped : bool = False):
        _transform = surface.transform
        super(SurfaceToImplict, self).__init__(_transform, is_normal_flipped)
        self._surface = surface

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, _surface):
        self._surface = _surface

    @ti.func
    def closest_point_local(self, local_p) -> Vector:
        return self._surface.closest_point_local(local_p)

    @ti.func
    def closest_point_normal_local(self, local_p:Vector) -> Vector:
        return self._surface.closest_point_normal_local(local_p)

    @ti.func
    def sign_distance_local(self, local_point: Vector) -> Float:
        p = self.surface.closest_point_local(local_point)
        return - EuclideanDistance(p, local_point) \
            if \
                self.surface.is_inside_local(local_point) \
            else \
                EuclideanDistance(p, local_point)

    @ti.pyfunc
    def signed_vector_l(self, world_point: Vector) -> Vector:
        """
        To
        :param world_point:
        :return:
        """
        # p = self.surface.closest_point_local(local_point)
        pass

    @ti.func
    def is_inside_local(self, local_p: Vector) -> bool:
        return self.surface.is_inside_local(local_p)


if __name__ == '__main__':
    ti.init(ti.gpu, debug=True)



