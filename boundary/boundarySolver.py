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
        self.marker_field = ti.field(dtype = ti.i32 , shape= self.cfg.res)
        self.marker_bffr_field = ti.field(dtype = ti.i32 , shape= self.cfg.res)
        self.colliders = self.cfg.Colliders

    @ti.kernel
    def kern_update_marker(self):
        for I in ti.grouped(self.marker_field):
            if (self.collider_sdf_field[I] <= 0.0):
                # in collider
                self.marker_field[I] = int(PixelType.Collider)
            else:
                self.marker_field[I] = int(PixelType.Liquid)

    def update_sdfs(self, colliders:List[Collider]):
        self.collider_sdf_field.fill(1e8)
        for cllider in colliders:
            self.kern_update_sdf(cllider.implict_surface)

    @ti.kernel
    def kern_update_sdf(self, implct_surf: ti.template()):
        sdf = ti.static(self.collider_sdf_field)

        for I in ti.grouped(sdf):
            self.collider_sdf_field[I] = min(sdf[I], implct_surf.signed_distance(I))

    @abstractmethod
    def ApplyBoundaryCondition(self):
        pass
    
@ti.data_oriented
class StdGridBoundaryConditionSolver(GridBoudaryConditionSolver):
    def __init__(self, cfg, grid:Grid):
        super(StdGridBoundaryConditionSolver, self).__init__(cfg, grid)

    def ApplyBoundaryCondition(self):
        pass
