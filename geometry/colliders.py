# ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi
from abc import ABCMeta, abstractmethod
from utils import Vector, Float
import taichi as ti
from .surface import SurfaceToImplict, ImplicitSurface
from .surfaceShape import SurfaceShape, Ball
from .transform2 import Transform2
from .velocity import Velocity2


class Collider(metaclass=ABCMeta):
    """
    class that integrated surfaceShape
    3.4.1.1
    conversion to signed-distance field enable cached the inside/outside testing
    and closest distance measuring into a grid
    => accelerate the collider queries
    """

    def __init__(self, surface_shape: SurfaceShape):
        self._surface = surface_shape
        self._implicit = SurfaceToImplict(surface_shape) if not isinstance(surface_shape,
                                                                           ImplicitSurface) else surface_shape

    @ti.pyfunc
    def kern_materialize(self):
        self.surfaceshape.kern_materialize()

    @property
    def surfaceshape(self) -> SurfaceShape:
        return self._surface

    @property
    def implicit_surface(self) -> ImplicitSurface:
        return self._implicit

    @abstractmethod
    def update(self, time_interval: float):
        pass

    @abstractmethod
    def velocity_at(self, point: Vector) -> Vector:
        pass

    @ti.pyfunc
    def is_inside_collider(self, world_p: Vector) -> bool:
        local_p = self.surfaceshape.transform.to_local(world_p)
        return self.implicit_surface.is_inside_local(local_p)

    @ti.func
    def color_at_world(self, world_p: Vector) -> Vector:
        local_p = self.surfaceshape.transform.to_local(world_p)
        return self.surfaceshape.color_at_local_point(local_p)

    @ti.pyfunc
    def reset(self):
        self.kern_materialize()


@ti.data_oriented
class RigidBodyCollider(Collider):
    def __init__(self, surface_shape):
        super().__init__(surface_shape)

    def update(self, time_interval: float):
        pass

    @ti.func
    def velocity_at(self, world_point: Vector) -> Vector:
        sfs = self.surfaceshape
        return sfs.velocity_at_local_point(sfs.transform.to_local(world_point))


if __name__ == '__main__':
    ti.init(ti.cpu, debug=True)
    colld_ball = RigidBodyCollider(Ball(
        transform=Transform2(translation=ti.Vector([300, 150]), localscale=16),
        velocity=Velocity2(velocity_to_world=ti.Vector([0.0, 0.0]), angular_velocity_to_centroid=10.0)))

    colld_ball.kern_materialize()
    world_p = ti.Vector([315.0, 150.0])
    print(colld_ball.implicit_surface.transform)


    @ti.kernel
    def test():
        # print(colld_ball.implict_surface.transform)
        # bad value

        print(colld_ball.is_inside_collider(world_p))


    test()
