# ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi
from abc import ABCMeta , abstractmethod
from utils import Vector, Float
import taichi as ti
from .surface import Surface, SurfaceToImplict, ImplicitSurface
from .surfaceShape import SurfaceShape

class Collider(metaclass=ABCMeta):
    '''
    class containing speed
    currently only for 2d
    '''
    def __init__(self, surface: SurfaceShape):
        self._surface = surface
        self._implicit = SurfaceToImplict(surface) if not isinstance(surface , ImplicitSurface) else surface

    @property
    def surface(self) -> Surface:
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

    def velocity_at(self, point: Vector) -> Vector:

