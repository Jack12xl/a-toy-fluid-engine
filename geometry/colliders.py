# ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi
from abc import ABCMeta , abstractmethod
from utils import Vector, Float
import taichi as ti
from .surface import Surface, SurfaceToImplict, ImplicitSurface
from .surfaceShape import SurfaceShape

class Collider(metaclass=ABCMeta):
    '''
    class that integrated surfaceShape
    3.4.1.1
    conversion to signed-distance field enable cached the inside/outside testing
    and closest distance measuring into a grid
    => accelerate the collider queries
    '''
    def __init__(self, surface: SurfaceShape):
        self._surface = surface
        self._implicit = SurfaceToImplict(surface) if not isinstance(surface , ImplicitSurface) else surface

    @property
    def surfaceshape(self) -> SurfaceShape:
        return self._surface

    @property
    def implict_surface(self) -> ImplicitSurface:
        return self._implicit

    @abstractmethod
    def update(self, time_interval: float):
        pass

    @abstractmethod
    def velocity_at(self, point: Vector) -> Vector:
        pass


@ti.data_oriented
class RigidBodyCollider(Collider):
    def __init__(self, surface):
        super().__init__(surface)

    def update(self, time_interval: float):
        pass

    @ti.func
    def velocity_at(self, world_point: Vector) -> Vector:
        sfs = self.surfaceshape
        return sfs.velocity_at_local_point(sfs.transform.to_local(world_point))
