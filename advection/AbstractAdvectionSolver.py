from abc import ABCMeta , abstractmethod
from utils import filterUpCase


class AdvectionSolver(metaclass = ABCMeta):

    @abstractmethod
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def advect(self, vec_field, q_cur, q_nxt, dt):
        pass

    def abbreviation(self):
        return filterUpCase(self.__class__.__name__)