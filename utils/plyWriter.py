import taichi as ti
import numpy as np
from config import EulerCFG
from .basic_types import Wrapper

class plyWriter(ti.PLYWriter):
    """
    Simple use
    """
    def __init__(self, cfg: EulerCFG, *args, **kwargs):
        super(plyWriter, self).__init__(args, kwargs)

        self.cfg = cfg
        self.res = self.cfg.res
        self.dim = self.cfg.dim

        self.ti_pos = ti.field.Vector(self.dim, dt=ti.f32, shape=self.cfg.res)
        self.np_pos = None

        self.num_vertices = np.prod(self.res)

    @ti.kernel
    def read_pos(self, f: Wrapper):
        for I in ti.static(f):
            self.ti_pos[I] = f.getW(I)

    def set_pos(self):
        self.np_pos = np.reshape(self.ti_pos.to_numpy(), (self.num_vertices, self.dim))

    def materialize(self):
        pass