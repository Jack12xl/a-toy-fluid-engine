from .surface import Surface
from .transform import Transform2
import taichi as ti
from basic_types import Vector, Matrix, Float
from utils import tiNormalize
from .surfaceShape import Ball

@ti.data_oriented
class RoatateBall(Ball):
    def __init__(self,
                 transform: Transform2 = Transform2(),
                 is_normal_flipped: bool = False,
                 angular_velocity: float = 0.0):
        @property
        def omega(self):
            return self._angular_velocity

        @omega.setter
        def omega(self, angular_velocity):
            self._angular_velocity = angular_velocity

        super(Ball, self).__init__(transform, is_normal_flipped)
        self.omega = angular_velocity

