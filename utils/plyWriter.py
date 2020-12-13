import taichi as ti
import numpy as np
from .basic_types import Wrapper
import os


class plyWriter(ti.PLYWriter):
    """
    Simple use
    """

    def __init__(self, fluid_color, cfg):
        self.cfg = cfg
        self.res = self.cfg.res
        self.dim = self.cfg.dim
        self.num_vertices = np.prod(self.res)

        self.fluid_color = fluid_color

        super(plyWriter, self).__init__(self.num_vertices)

        self.ti_pos = ti.field.Vector(self.dim, dt=ti.f32, shape=self.cfg.res)
        self.np_pos = None

        self.ti_den = ti.field(dt=ti.f32, shape=self.cfg.res)
        self.np_den = None

        self.series_prefix = os.path.join(self.cfg.profile_name, 'PLYs')

    @ti.kernel
    def read_pos(self, f: Wrapper):
        for I in ti.static(f):
            self.ti_pos[I] = f.getW(I)

    @ti.kernel
    def read_den(self, rho_f: Wrapper):
        for I in ti.static(rho_f):
            self.ti_den[I] = rho_f[I][0] / self.fluid_color[0]

    def set_pos(self):
        self.np_pos = np.reshape(self.ti_pos.to_numpy(), (self.num_vertices, 3))

    def set_den(self):
        self.np_den = np.reshape(self.ti_den.to_numpy(), (self.num_vertices, 1))

    def save_frame(self, frame_number, rho_f: Wrapper):
        if frame_number == 0:
            self.read_pos(rho_f)
            self.set_pos()
        self.add_vertex_pos(self.np_pos[:, 0], self.np_pos[:, 1], self.np_pos[:, 2])
        self.read_den(rho_f)
        self.add_vertex_alpha(self.np_den)
        self.export_frame_ascii(frame_number, self.series_prefix)
