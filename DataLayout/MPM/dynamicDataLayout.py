import taichi as ti
import taichi_glsl as ts
from config.CFG_wrapper import mpmCFG, DLYmethod, MaType
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
        print("Use dynamic layout !")
        # pid -> pos
        self.pid = ti.field(Int)
        self.p_chunk_size = self.cfg.p_chunk_size
        # TODO why...
        self.grid_size = 4096

    def materialize(self):
        # TODO

        _indices = ti.ij if self.dim == 2 else ti.ijk
        # offset : map to the whole grid center
        # self.offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = tuple(0 // 2 for _ in range(self.dim))

        # grid
        grid_block_size = 128
        self.leaf_block_size = 64 // 2 ** self.dim

        self._grid = ti.root.pointer(_indices, self.grid_size // grid_block_size)
        block = self._grid.pointer(_indices,
                                   grid_block_size // self.leaf_block_size)

        def block_component(c):
            block.dense(_indices, self.leaf_block_size).place(c, offset=self.offset)

        # def block_component(c):
        #     new_grid = ti.root.pointer(_indices, self.grid_size // grid_block_size)
        #     new_block = new_grid.pointer(_indices, grid_block_size // self.leaf_block_size)
        #     new_block.dense(_indices, self.leaf_block_size).place(c, offset=self.offset)


        # assign
        block_component(self.g_m.field)
        for v in self.g_v.field.entries:
            block_component(v)

        # TODO what's this...
        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size ** self.dim * 8).place(
            self.pid, offset=self.offset + (0, ))

        self._particle = ti.root.dynamic(ti.i, self.max_n_particle, self.p_chunk_size)
        # particle
        self._particle.place(self.p_x,
                             self.p_v,
                             self.p_C,
                             self.p_F,
                             self.p_material_id,
                             self.p_color,
                             self.p_Jp)

    @ti.kernel
    def build_pid(self):
        ti.block_dim(64)
        for P in self.p_x:
            # TODO check this
            base = int(ti.floor(self.g_m.getG(self.p_x[P]) - 0.5))
            ti.append(self.pid.parent(), base - ti.Vector(list(self.offset)),
                      P)

    @ti.kernel
    def P2G(self, dt: Float):
        """
        Dynamic way of P2G
        :param dt:
        :return:
        """
        # TODO what's this...
        ti.no_activate(self._particle)
        ti.block_dim(256)

        ti.block_local(*self.g_v.field.entries)
        ti.block_local(self.g_m.field)
        for I in ti.grouped(self.pid):
            P = self.pid[I]
            self.P2G_func(dt, P)

    @ti.kernel
    def G2P(self, dt: Float):
        """

        :param dt:
        :return:
        """

        ti.block_dim(256)

        ti.block_local(*self.g_v.field.entries)
        ti.no_activate(self._particle)
        for I in ti.grouped(self.pid):
            P = self.pid[I]
            self.G2P_func(dt, P)

    def G2zero(self):
        """
        TODO what is this
        :return:
        """
        self._grid.deactivate_all()
        self.build_pid()

    def add_cube(self,
                 l_b: Vector,
                 cube_size: Vector,
                 mat: MaType,
                 density: Int,
                 velocity: Vector,
                 color=0xFFFFFF,
                 ):
        """
        Use density since there is no limit for particle number
        :param l_b:
        :param cube_size:
        :param mat:
        :param density:
        :param velocity:
        :param color:
        :return:
        """
        vol = 1.0
        for d in range(self.dim):
            vol *= cube_size[d]

        n_p = int(density * vol / self.cfg.dx ** self.dim + 1)
        super(mpmDynamicLayout, self).add_cube(l_b,
                                               cube_size,
                                               mat,
                                               n_p,
                                               velocity,
                                               color)
        return n_p

