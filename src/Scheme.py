from .Grid import Grid
import taichi as ti
import numpy as np
from config import VisualizeEnum, SceneEnum, SchemeType

@ti.data_oriented
class EulerScheme():
    def __init__(self, cfg, ):
        self.cfg = cfg
        self.grid = Grid(cfg)

        self.clr_bffr = ti.Vector(3, dt=ti.f32, shape=cfg.res)
        self.advection_solver = self.cfg.advection_solver(cfg, self.grid)


    def advect(self, dt):
        self.advection_solver.advect(self.grid.v_pair.cur, self.grid.v_pair.cur, self.grid.v_pair.nxt, dt)
        self.advection_solver.advect(self.grid.v_pair.cur, self.grid.dye_pair.cur, self.grid.dye_pair.nxt, dt)
        self.grid.v_pair.swap()
        self.grid.dye_pair.swap()

    def externalForce(self, ext_input, dt):
        if (self.cfg.SceneType == SceneEnum.MouseDragDye):
            # add impulse from mouse
            self.apply_mouse_input_and_render(self.grid.v_pair.cur, self.grid.dye_pair.cur, ext_input, dt)
        elif (self.cfg.SceneType == SceneEnum.ShotFromBottom):
            self.add_fixed_force_and_render(self.grid.v_pair.cur, dt)

    def project(self):
        self.grid.calDivergence(self.grid.v_pair.cur, self.grid.v_divs)

        self.grid.Jacobi_run_pressure()
        self.grid.Jacobi_run_viscosity()

    @ti.kernel
    def fill_color(self, vf: ti.template()):
        for i, j in vf:
            v = vf[i, j]
            self.clr_bffr[i, j] = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])

    @ti.kernel
    def fill_color_2d(self, vf: ti.template()):
        for i, j in vf:
            v = vf[i, j]
            self.clr_bffr[i, j] = ti.Vector([abs(v[0]), abs(v[1]), 0.25])

    @ti.kernel
    def apply_mouse_input_and_render(self, vf: ti.template(), dyef: ti.template(),
                                     imp_data: ti.ext_arr(),
                                     dt: ti.template()):

        for i, j in vf:
            mdir = ti.Vector([imp_data[0], imp_data[1]])
            omx, omy = imp_data[2], imp_data[3]
            # move to cell center
            dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
            d2 = dx * dx + dy * dy
            # ref: https://developer.download.nvidia.cn/books/HTML/gpugems/gpugems_ch38.html
            # apply the force
            factor = ti.exp(-d2 * self.cfg.inv_force_radius)
            momentum = mdir * self.cfg.f_strength * dt * factor

            vf[i, j] += momentum
            # add dye
            dc = dyef[i, j]
            # TODO what the hell is this?
            if mdir.norm() > 0.5:
                dc += ti.exp(-d2 * self.cfg.inv_dye_denom) * ti.Vector(
                    [imp_data[4], imp_data[5], imp_data[6]])
            dc *= self.cfg.dye_decay
            dyef[i, j] = dc


    @ti.kernel
    def add_fixed_force_and_render(self,
                                   vf: ti.template(),
                                   dt: ti.template()):
        for i, j in vf:
            dx, dy = i + 0.5 - self.cfg.source_x, j + 0.5 - self.cfg.source_y
            d2 = dx * dx + dy * dy
            momentum = (self.cfg.direct_X_force * ti.exp( -d2 * self.cfg.inv_force_radius ) - self.cfg.f_gravity) * dt
            vf[i, j] += momentum

            dc = self.grid.dye_pair.cur[i, j]
            dc += ti.exp( - d2 * self.cfg.inv_dye_denom) * self.cfg.fluid_color
            dc *= self.cfg.dye_decay
            self.grid.dye_pair.cur[i, j] = min(dc, self.cfg.fluid_color)

    def render_frame(self):
        if (self.cfg.VisualType == VisualizeEnum.Velocity):
            self.fill_color_2d(self.grid.v_pair.cur)
        elif (self.cfg.VisualType == VisualizeEnum.Dye):
            self.fill_color(self.grid.dye_pair.cur)


    def step(self, ext_input:np.array):
        if (self.cfg.run_scheme == SchemeType.Advection_Projection):
            self.advect(self.cfg.dt)
            self.externalForce(ext_input, self.cfg.dt)
            self.project()
            self.grid.subtract_gradient(self.grid.v_pair.cur, self.grid.p_pair.cur)

        elif (self.cfg.run_scheme == SchemeType.Advection_Reflection):
            self.advect(self.cfg.half_dt)
            self.externalForce(ext_input, self.cfg.half_dt)

        self.render_frame()
    def reset(self):
        self.grid.reset()
        self.clr_bffr.fill(ti.Vector([0, 0, 0]))