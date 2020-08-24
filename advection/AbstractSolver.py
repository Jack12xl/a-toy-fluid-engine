from abc import ABCMeta , abstractmethod
import taichi as ti

class AdvectionSolver(metaclass = ABCMeta):

    @abstractmethod
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def advect(self, vec_field, q_cur, q_nxt, dt):
        pass