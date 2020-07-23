from .Grid import Grid
import taichi as ti
import numpy as np

@ti.data_oriented
class EulerScheme():
    def __init__(self, cfg:dict, ):
        self.cfg = cfg
        self.grid = Grid(cfg)

        self.clr_bffr = ti.Vector(3, dt=ti.f32, shape=cfg['res'])

    @ti.kernel
    def advect_q(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j in vf:
            coord = ( ti.Vector([i,j]) + 0.5 ) * self.cfg['dx'] - vf[i, j] * self.cfg['dt']
            new_qf[i,j] = self.grid.interpolate_value(qf, coord)

    @ti.kernel
    def project(self):
        pass

    @ti.kernel
    def fill_color(self, vf: ti.template()):
        for i, j in vf:
            v = vf[i, j]
            self.clr_bffr[i, j] = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])

    @ti.kernel
    def apply_impulse(vf: ti.template(), dyef: ti.template(),
                      imp_data: ti.ext_arr()):

        for i, j in vf:
            mdir = ti.Vector([imp_data[0], imp_data[1]])
            omx, omy = imp_data[2], imp_data[3]


            pass

        pass

    def step(self, mouse_data:np.array):
        self.advect_q(self.grid.v_pair.cur, self.grid.v_pair.cur, self.grid.v_pair.nxt)
        self.advect_q(self.grid.v_pair.cur, self.grid.dye_pair.cur, self.grid.dye_pair.nxt)
        self.grid.v_pair.swap()
        self.grid.dye_pair.swap()

        # add impulse
        self.fill_color(self.grid.dye_pair.cur)

        pass