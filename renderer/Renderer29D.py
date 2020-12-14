import taichi as ti
import taichi_glsl as ts
from .abstractRenderer import renderer
from config import PixelType, VisualizeEnum
from utils import cmapper, Matrix, Wrapper

@ti.data_oriented
class renderer29D(renderer):
    def __init__(self,
                 cfg,
                 grid
                 ):
        """
        Orthogonal view volumne render
        face towards the grid
        :param cfg:
        :param grid:
        """
        super(renderer29D, self).__init__(cfg, grid)
        self.mapper = cmapper()
        self.res = ts.vec3(cfg.res)

    @ti.func
    def fresh_buf(self):
        for I in ti.grouped(self.clr_bffr):
            self.clr_bffr[I] = ts.vec3(0.0)

    @ti.kernel
    def render_collider(self, bdrySolver: ti.template()):
        pass

    @ti.func
    def getMaxColor(self):
        ret = ts.vecND(self.cfg.dim, 0.0)
        for I in ti.grouped(self.clr_bffr):
            ti.atomic_max(ret, self.clr_bffr[I])
        return ret + 0.0001

    @ti.kernel
    def vis_density(self, rho_f: Wrapper):
        self.fresh_buf()
        for I in ti.static(rho_f):
            self.clr_bffr[I.xy] += rho_f[I]

        for I in ti.static(rho_f):
            self.clr_bffr[I.xy] /= self.res[2]
            self.clr_bffr[I.xy] *= 32.0

    @ti.kernel
    def vis_v(self, vf: Wrapper):
        self.fresh_buf()
        for I in ti.static(vf):
            self.clr_bffr[I.xy] += ti.abs(vf[I])

        for I in ti.static(vf):
            self.clr_bffr[I.xy] /= self.res[2]

    @ti.kernel
    def vis_v_mag(self, vf: Wrapper):
        self.fresh_buf()
        for I in ti.static(vf):
            self.clr_bffr[I.xy] += abs(vf[I])
        for I in ti.static(vf):
            para = self.clr_bffr[I.xy] / self.res[2]
            self.clr_bffr[I.xy] = self.mapper.color_map(para.norm())

    @ti.kernel
    def vis_vd(self, vf: Wrapper):
        self.fresh_buf()
        for I in ti.static(vf):
            self.clr_bffr[I.xy] += ts.vec3(vf[I], 0.0, 0.0)
        for I in ti.static(vf):
            para = self.clr_bffr[I.xy] / self.res[2]
            self.clr_bffr[I.xy] = para * 0.3 + ts.vec3(0.5)

    @ti.kernel
    def vis_vt(self, vf: Wrapper):
        self.fresh_buf()
        for I in ti.static(vf):
            self.clr_bffr[I.xy] += vf[I]
        for I in ti.static(vf):
            self.clr_bffr[I.xy] = 0.03 * self.clr_bffr[I.xy] / self.res[2] + ts.vec3(0.5)

    def renderStep(self, bdrySolver):
        self.render_frame()

        if len(bdrySolver.colliders):
            self.render_collider(bdrySolver)

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