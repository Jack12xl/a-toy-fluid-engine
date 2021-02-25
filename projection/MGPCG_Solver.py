import taichi as ti
import taichi_glsl as ts
from .AbstractProjectionSolver import ProjectionSolver


@ti.data_oriented
class MGPCG_solver(ProjectionSolver):
    def __init__(self, cfg, grid, bdrt):
        pass
