import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import mpmCFG, DataLayout
from utils import Int, Float, Matrix
from Grid import CellGrid

@ti.data_oriented
class mpmLayout(metaclass=ABCMeta):
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
        self.p_v = ti.Vector.field(self.dim, dtype=Float)
        # affine velocity field
        self.p_C = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        # deformation gradient
        self.p_F = ti.Matrix.field(self.dim, self.dim, dtype=Float)
        # material
        self.p_material_id = ti.field(dtype=Int)
        self.p_color = ti.field(dtype=Int)
        # plastic deformation volume ratio
        self.p_Jp = ti.field(dtype=Int)

        # Grid
        # velocity
        self.g_v = CellGrid(
            ti.Vector.field(self.dim, dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )
        # mass
        self.g_m = CellGrid(
            ti.field(dtype=Float),
            self.dim,
            dx=ts.vecND(self.dim, self.cfg.dx),
            o=ts.vecND(self.dim, 0.0)
        )

        pass

    def materialize(self):
        self._particle = ti.root.dense(ti.i, self.n_particles)
        _indices = ti.ij if self.dim == 2 else ti.ijk
        if self.cfg.layout_method == DataLayout.FLAT:
            self._particle.place(self.p_x)
            self._particle.place(self.p_v)
            self._particle.place(self.p_C)
            self._particle.place(self.p_F)
            self._particle.place(self.p_material_id)
            self._particle.place(self.p_color)
            self._particle.place(self.p_Jp)

            self._grid = ti.root.dense(_indices, self.cfg.res)
            self._grid.place(self.g_v.field)
            self._grid.place(self.g_m.field)
        elif self.cfg.layout_method == DataLayout.H1:
            self._particle.place(self.p_x,
                                 self.p_v,
                                 self.p_C,
                                 self.p_F,
                                 self.p_material_id,
                                 self.p_color,
                                 self.p_Jp)
            # TODO finish the setup
        pass

    def grid2zero(self):
        self.g_m.fill(0.0)
        self.g_v.fill(ts.vecND(self.dim, 0.0))

    @ti.kernel
    def P2G(self, dt: Float):
        """
        Particle to Grid
        :param dt:
        :return:
        """
        p_x = ti.static(self.p_x)
        g_m = ti.static(self.g_m)
        g_v = ti.static(self.g_v)
        p_F = ti.static(self.p_F)
        for I in p_x:
            base = ti.floor(g_m.getG(p_x[I] - 0.5 * g_m.dx)).cast(int)
            fx = g_m.getG(p_x[I]) - base.cast(float)
            # Here we adopt quadratic kernels
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw =

            U, sig, V = ti.svd(p_F[I])
