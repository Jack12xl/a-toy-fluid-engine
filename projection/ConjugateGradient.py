import taichi as ti
import taichi_glsl as ts
from .AbstractProjectionSolver import ProjectionSolver
import taichi_glsl as ts

@ti.data_oriented
class ConjugateGradient(ProjectionSolver):

    def __init__(self, cfg, grid):
        super().__init__(cfg, grid)


