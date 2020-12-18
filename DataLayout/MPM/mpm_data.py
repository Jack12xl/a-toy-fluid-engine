import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import mpmCFG
from utils import Int, Float, Matrix


@ti.data_oriented
class mpmData(metaclass=ABCMeta):
    """
    the default data for MPM
    """

    def __init__(self, cfg: mpmCFG):
        self.cfg = cfg

        self.dim = cfg.dim
        self.n_particles = ti.field(dtype=Int)

        # Particle

        # position
        self.p_x = ti.Vector.field(self.dim, dtype=Float)
        # velocity
        self.v = ti.Vector.field(self.dim, dtype=Float)
        # affine velocity field
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        # deformation gradient
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        # material
        self.material_id = ti.field(dtype=Int)
        self.color = ti.field(dtype=Int)
        # plastic deformation volume ratio
        self.Jp = ti.field(dtype=Int)

        # Grid

        self.g_v = ti.Vector.field(self.dim, dtype=Float)
        self.g_m = ti.field(dtype=Float)

        pass
