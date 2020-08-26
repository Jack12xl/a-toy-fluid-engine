from abc import ABCMeta , abstractmethod

class ProjectionSolver(metaclass = ABCMeta):

    @abstractmethod
    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid

    @abstractmethod
    def runPressure(self):
        pass

    @abstractmethod
    def runViscosity(self):
        pass