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

        self.collidr_sdf_field = ti.field(dtype = ti.i32, shape= self.cfg.dim)
        self.marker_field = ti.field(dtype = ti.i32 , shape= self.cfg.dim)
        self.marker_bffr_field = ti.field(dtype = ti.i32 , shape= self.cfg.dim)


    @ti.kernel
    def kern_update_marker(self):
        for I in ti.grouped(self.marker_field):
            if (self.collidr_sdf_field[I] < 0.0):
                # in collider
                self.marker_field[I] = PixelType.Collider
            else:
                self.marker_field[I] = PixelType.Fluid

    def update_colliders(self, colliders:List[Collider]):
        self.collidr_sdf_field.fill(1e8)
        for cllider in colliders:
            self.kern_update_collider(cllider)

    @ti.kernel
    def kern_update_collider(self, cld: ti.template()):
        sdf = ti.static(self.collidr_sdf_field)
        for I in ti.grouped(sdf):
            self.collidr_sdf_field[I] = min(sdf[I], cld.implict_surface())

    @abstractmethod
    def ApplyBoundaryCondition(self):
        pass
    
    
class StdGridBoundaryConditionSolver(GridBoudaryConditionSolver):
    def __init__(self, cfg, grid:Grid):
        super(StdGridBoundaryConditionSolver, self).__init__(cfg, grid)

    def ApplyBoundaryCondition(self):
        pass
