import taichi as ti
from abc import ABCMeta , abstractmethod
from Grid import Grid

class GridBoudaryConditionSolver(metaclass = ABCMeta):
    '''
    solve boundary condition and manager C
    '''
    def __init__(self, cfg, grid: Grid):
        self.cfg = cfg
        self.grid = Grid

    def ApplyBoundaryCondition(self, ):
        pass


