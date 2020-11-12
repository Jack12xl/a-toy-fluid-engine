import taichi as ti
import taichi_glsl as ts
import math
from utils import Vector, Matrix, tiNormalize, Float
from config.base_cfg import error

class Transform3:
    def __init__(self,
                 translation=ts.vec3(0.0, 0.0, 0.0),
                 orientation = ts.vec2(0.0, 0.0),
                 localscale = ts.vec3(1.0)):
        self._translation = ti.Vector.field(3, dtype=ti.f32, shape = [])
        self._orientation = ti.Vector.field(2, dtype=ti.f32, shape = [])
        self._localScale = ti.Vector.field(3, dtype=ti.f32, shape = [])
        # use buffer for later materialization
        self.translation_buf = translation
        self.orientation_buf = orientation % (2 * math.pi)
        self.localscale_buf = localscale