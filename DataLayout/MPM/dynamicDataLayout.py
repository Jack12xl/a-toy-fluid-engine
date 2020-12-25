import taichi as ti
import taichi_glsl as ts
from config.CFG_wrapper import mpmCFG, DLYmethod
from utils import Int, Float, Matrix, Vector
from Grid import CellGrid
from .dataLayout import mpmLayout


@ti.data_oriented
class mpmDynamicLayout(mpmLayout):
    """
    TODO
    To support dynamic particles adding
    """

    def __init__(self, cfg: mpmCFG):
        super(mpmDynamicLayout, self).__init__(cfg)

        self.max_n_particle = self.cfg.max_n_particle
        self.p_chunk_size = self.cfg.p_chunk_size


    def materialize(self):
        # TODO
        self._particle = ti.root.dynamic(ti.i, self.max_n_particle, self.p_chunk_size)

    @ti.kernel
    def seed(self, n_p: Int, mat: Int, color: Int):
        for P in range(self.n_max_particle[None],
                       self.n_max_particle[None] + n_p):
            self.p_material_id[P] = mat
            x = self.source_bound[0] + ts.randND(self.dim)
            self.seed_particle(P, x, mat, color, self.source_velocity[None])

    @ti.func
    def seed_particle(self, P, x, mat, color, velocity):
        self.p_x[P] = x
        self.p_v[P] = velocity
        self.p_F[P] = ti.Matrix.identity(Float, self.dim)
        self.p_color[P] = color
        self.p_material_id[P] = mat
