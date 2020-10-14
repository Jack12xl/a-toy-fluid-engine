import taichi as ti
from abc import ABCMeta , abstractmethod
from Grid import Grid
from utils import Vector, Matrix
from config import PixelType
from geometry import Collider
from typing import List

# ref : https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi/blob/master/src/python/Eulerian_method.py
class GridBoudaryConditionSolver(metaclass = ABCMeta):
    '''
    solve boundary condition and manager C
    '''
    def __init__(self, cfg, grid: Grid):
        self.cfg = cfg
        self.grid = grid

        self.collider_sdf_field = ti.field(dtype = ti.f32, shape= self.cfg.res)
        self.collider_velocity_field = ti.Vector.field(cfg.dim, dtype = ti.f32, shape= self.cfg.res)

        self.marker_field = ti.field(dtype = ti.i32 , shape= self.cfg.res)
        self.marker_bffr_field = ti.field(dtype = ti.i32 , shape= self.cfg.res)
        self.colliders = self.cfg.Colliders

    @ti.kernel
    def kern_update_marker(self):
        #TODO support more type
        for I in ti.grouped(self.marker_field):
            if (self.collider_sdf_field[I] <= 0.0):
                # in collider
                self.marker_field[I] = int(PixelType.Collider)
            else:
                self.marker_field[I] = int(PixelType.Liquid)

    def step_update_sdfs(self, colliders:List[Collider]):
        self.collider_sdf_field.fill(1e8)
        for cllider in colliders:
            self.kern_update_sdf(cllider)

    @ti.kernel
    def kern_update_sdf(self, collid: ti.template()):
        sdf = ti.static(self.collider_sdf_field)
        vf = ti.static(self.collider_velocity_field)
        for I in ti.grouped(sdf):
            sdf[I] = min(sdf[I], collid.implict_surface.signed_distance(I))


    @abstractmethod
    def ApplyBoundaryCondition(self):
        pass
    
@ti.data_oriented
class StdGridBoundaryConditionSolver(GridBoudaryConditionSolver):
    def __init__(self, cfg, grid:Grid):
        super(StdGridBoundaryConditionSolver, self).__init__(cfg, grid)

    def ApplyBoundaryCondition(self):
        pass

    @ti.kernel
    def kernBoundaryCondition(self):
        # no flux
        # slip
        # for I in ti.grouped(self.grid.v):

        pass


