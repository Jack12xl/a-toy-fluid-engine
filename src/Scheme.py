from .Grid import Grid
import taichi as ti
import numpy as np


@ti.data_oriented
class EulerScheme():
    def __init__(self, cfg, ):
        self.cfg = cfg
        self.grid = Grid(cfg)

        self.clr_bffr = ti.Vector(3, dt=ti.f32, shape=cfg.res)

    @ti.kernel
    def advect_q(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j in vf:
            coord = ( ti.Vector([i,j]) + 0.5 ) * self.cfg.dx - vf[i, j] * self.cfg.dt
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
    def apply_impulse(self, vf: ti.template(), dyef: ti.template(),
                      imp_data: ti.ext_arr()):

        for i, j in vf:
            mdir = ti.Vector([imp_data[0], imp_data[1]])
            omx, omy = imp_data[2], imp_data[3]
            # move to cell center
            dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
            d2 = dx * dx + dy * dy
            # ref: https://developer.download.nvidia.cn/books/HTML/gpugems/gpugems_ch38.html
            # apply the force
            factor = ti.exp(-d2 * self.cfg.inv_force_radius)
            momentum = mdir * self.cfg.f_strength_dt * factor

            vf[i, j] += momentum
            # add dye
            dc = dyef[i, j]
            # TODO what the hell is this?
            if mdir.norm() > 0.5:
                dc += ti.exp(-d2 * self.cfg.inv_dye_denom) * ti.Vector(
                    [imp_data[4], imp_data[5], imp_data[6]])
            dc *= self.cfg.dye_decay
            dyef[i, j] = dc


    def step(self, mouse_data:np.array):
        self.advect_q(self.grid.v_pair.cur, self.grid.v_pair.cur, self.grid.v_pair.nxt)
        self.advect_q(self.grid.v_pair.cur, self.grid.dye_pair.cur, self.grid.dye_pair.nxt)
        self.grid.v_pair.swap()
        self.grid.dye_pair.swap()

        # add impulse
        self.apply_impulse(self.grid.v_pair.cur, self.grid.dye_pair.cur, mouse_data)

        self.fill_color(self.grid.dye_pair.cur)
