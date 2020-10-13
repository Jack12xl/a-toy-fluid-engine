from .surface import Surface
from .transform import Transform2
from .velocity import Velocity2
import taichi as ti
from utils import Vector, Matrix, Float
from utils import tiNormalize
from abc import ABCMeta , abstractmethod


@ti.data_oriented
class SurfaceShape(Surface):
    '''
    class with velocity, mass
    more of interface for an instance
    '''
    @property
    def Velocity(self) -> Velocity2:
        '''
        velocity w.r.t center of mass
        :return:
        '''
        return self._velocity

    @Velocity.setter
    def Velocity(self, _v: Velocity2):
        self._velocity = _v


    @property
    def mass(self) -> ti.f32:
        return self._mass

    @mass.setter
    def mass(self, _m):
        self._mass = max(_m, 0.0001)

    def __init__(self,
                 transform: Transform2 = Transform2(),
                 velocity: Velocity2 = Velocity2(),
                 is_normal_flipped:bool = False,
                 mass:ti.f32= 1.0
                 ):
        super(SurfaceShape, self).__init__(transform, is_normal_flipped)
        self.Velocity = velocity
        self.mass = mass

    @abstractmethod
    def velocity_at_local_point(self, local_point: Vector):
        pass

    @abstractmethod
    def color_at_local_point(self, local_point: Vector):
        pass

    @ti.kernel
    def update_transform(self, delta_time: Float):
        #TODO seems too intuitive
        self.transform.orientation = self.transform.orientation + self.Velocity.w_centroid * delta_time
        self.transform.translation = self.transform.translation + self.Velocity.v_world * delta_time


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
                 velocity: Velocity2 = Velocity2(),
                 is_normal_flipped:bool = False,
                 mass:ti.f32 = 1.0,
                 ):
        super(Ball, self).__init__(
            transform,
            velocity,
            is_normal_flipped,
            mass
        )

    @ti.func
    def closest_point_normal_local(self, local_p:Vector) -> Vector:
        return tiNormalize(local_p)

    @ti.func
    def closest_point_local(self, local_p) -> Vector:
        ## TODO if local_p is [0, 0]
        return tiNormalize(local_p)

    @ti.pyfunc
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
            # plot a red line
            c = ti.Vector([0.7, 0.2, 0.2])
        else:
            c = ti.Vector([0.9, 0.9, 0.9])
        return c


if __name__ == '__main__':
    ti.init(ti.gpu, debug=True)
    m_ball = Ball(transform=Transform2(ti.Vector([2.0, 2.0]), localscale=5.0))
    m_ball.kern_materialize()

    @ti.kernel
    def test_kern():
        print(m_ball.is_inside_world(ti.Vector([2.0, 2.0])))

    test_kern()
    m_ball.transform.translation = ti.Vector([1.5, 1.5])
    # print(m_ball.transform._translation[None])
