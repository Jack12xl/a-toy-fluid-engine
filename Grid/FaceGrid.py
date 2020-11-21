import taichi as ti
import taichi_glsl as ts
from .Sampler import LinearSampler2D, LinearSampler3D
from .Grid import Grid

@ti.data_oriented
class FaceGrid(Grid):
    """
    wrapper for data on face center (e.g. velocity on MAC grid)
    - F -
    F C F
    - F -
    """