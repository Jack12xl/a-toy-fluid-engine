import taichi as ti
import taichi_glsl as ts
from .abstractRenderer import renderer
from config import PixelType, VisualizeEnum
from utils import cmapper, Matrix, Wrapper
import numpy as np


@ti.data_oriented
class renderer25D(renderer):
    def __init__(self,
                 cfg,
                 grid,
                 z_plane):
        """
        Simply visualize a scarf
        :param cfg:
        :param grid:
        """
        super(renderer25D, self).__init__(cfg, grid)
        self.rho_buf = ti.field(dtype=ti.f32, shape=cfg.res)
        self.rho_buf2 = ti.field(dtype=ti.f32, shape=[cfg.res[0], cfg.res[1]])
        # first materialize
        self.mapper = cmapper()
        self.z_plane = z_plane
        self.res = ts.vec3(cfg.res)

    @ti.kernel
    def render_collider(self, bdrySolver: ti.template()):
        pass

    @ti.kernel
    def get_rho(self, rho_f: Wrapper):
        for I in ti.static(rho_f):
            self.rho_buf[I] = rho_f[I][0]

    @ti.kernel
    def write_rho(self):
        for I in ti.grouped(self.clr_bffr):
            self.clr_bffr[I] = ts.vec3(self.rho_buf2[I])

    def vis_density(self, rho_f: Wrapper):

        self.get_rho(rho_f)
        np_rho = np.reshape(self.rho_buf.to_numpy(), self.cfg.res)

        img = np.sum(np_rho, axis=2)
        img /= np.max(img)
        self.rho_buf2.from_numpy(img)
        self.write_rho()
        # self.clr_bffr.fill(img)

        # for I in ti.static(rho_f):
        #     self.clr_bffr[I.xy] = ts.vec3(0.0)
        #
        # for I in ti.static(rho_f):
        #     self.clr_bffr[I.xy] += rho_f[I]

        # ret = 0.0
        # for I in ti.grouped(self.clr_bffr):
        #     ti.atomic_max(ret, self.clr_bffr[I].norm())
        # ret += 0.00001

        # ret = ts.vec3(0.0)
        # for I in ti.grouped(self.clr_bffr):
        #     v = self.clr_bffr[I]
        #     ti.atomic_max(ret, v)
        # ret += 0.00001

        # for I in ti.static(rho_f):
        #     self.clr_bffr[I.xy] /= self.res[2] / 16.0

        # for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
        #     self.clr_bffr[I.xy] = ti.abs(vf[I, self.z_plane])

    @ti.kernel
    def vis_v(self, vf: ti.template()):
        # velocity
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            self.clr_bffr[I] = ti.abs(vf[I, self.z_plane])
            # self.clr_bffr[I] = 0.01 * vf[I, self.z_plane] + ts.vec3(0.3)
            # self.clr_bffr[I.xy] = ti.abs(vf[I])

    @ti.kernel
    def vis_v_mag(self, vf: ti.template()):
        # velocity magnitude
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            v_norm = vf[I, self.z_plane].norm() * 0.4
            self.clr_bffr[I] = self.mapper.color_map(v_norm)
            # self.clr_bffr[I] = ts.vec3(v_norm)

    @ti.kernel
    def vis_vd(self, vf: ti.template()):
        # divergence
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            v = ts.vec3(vf[I, self.z_plane], 0.0, 0.0)
            self.clr_bffr[I] = 0.3 * v + ts.vec3(0.5)
            # v = ts.vec3(vf[I])
            # self.clr_bffr[I.xy] = ti.abs(v)

    @ti.kernel
    def vis_vt(self, vf: ti.template()):
        # visualize vorticity
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            v = vf[I, self.z_plane]
            self.clr_bffr[I] = 0.03 * v + ts.vec3(0.5)

    @ti.kernel
    def vis_t(self, tf: Matrix, MaxT: ti.f32):
        # visualize temperature
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            t = tf[I, self.z_plane][0]
            self.clr_bffr[I] = ts.vec3(t / MaxT)

    def render_frame(self, render_what: VisualizeEnum = None):
        if render_what is None:
            render_what = self.cfg.VisualType
        if render_what == VisualizeEnum.Velocity:
            self.vis_v(self.grid.v_pair.cur)
        elif render_what == VisualizeEnum.Density:
            self.vis_density(self.grid.density_pair.cur)
        elif render_what == VisualizeEnum.Divergence:
            self.vis_vd(self.grid.v_divs)
        elif render_what == VisualizeEnum.Vorticity:
            self.vis_vt(self.grid.v_curl)
        elif render_what == VisualizeEnum.VelocityMagnitude:
            self.vis_v_mag(self.grid.v_pair.cur)
        elif render_what == VisualizeEnum.Temperature:
            self.vis_t(self.grid.t, self.cfg.GasMaxT)

    def renderStep(self, bdrySolver):
        self.render_frame()

        if len(bdrySolver.colliders):
            self.render_collider(bdrySolver)
