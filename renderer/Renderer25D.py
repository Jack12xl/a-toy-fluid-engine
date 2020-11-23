import taichi as ti
import taichi_glsl as ts
from .abstractRenderer import renderer
from config import PixelType, VisualizeEnum
from utils import cmapper, Matrix

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
        self.mapper = cmapper()
        self.z_plane = z_plane
        self.res = ts.vec3(cfg.res)

    @ti.kernel
    def render_collider(self, bdrySolver: ti.template()):
        pass

    @ti.kernel
    def vis_density(self, vf: ti.template()):
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            self.clr_bffr[I.xy] = ti.abs(vf[I, self.z_plane])

    @ti.kernel
    def vis_v(self, vf: ti.template()):
        # velocity
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            self.clr_bffr[I] = 0.04 * ti.abs(vf[I, self.z_plane])
            # self.clr_bffr[I] = 0.01 * vf[I, self.z_plane] + ts.vec3(0.3)
            # self.clr_bffr[I.xy] = ti.abs(vf[I])

    @ti.kernel
    def vis_v_mag(self, vf: ti.template()):
        # velocity magnitude
        for I in ti.grouped(ti.ndrange(self.res.x, self.res.y)):
            v_norm = vf[I, self.z_plane].norm() * 0.02
            self.clr_bffr[I] = self.mapper.color_map(v_norm)

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

    def render_frame(self):
        if self.cfg.VisualType == VisualizeEnum.Velocity:
            self.vis_v(self.grid.v_pair.cur)
        elif self.cfg.VisualType == VisualizeEnum.Density:
            self.vis_density(self.grid.density_pair.cur)
        elif self.cfg.VisualType == VisualizeEnum.Divergence:
            self.vis_vd(self.grid.v_divs)
        elif self.cfg.VisualType == VisualizeEnum.Vorticity:
            self.vis_vt(self.grid.v_curl)
        elif self.cfg.VisualType == VisualizeEnum.VelocityMagnitude:
            self.vis_v_mag(self.grid.v)
        elif self.cfg.VisualType == VisualizeEnum.Temperature:
            self.vis_t(self.grid.t, self.cfg.GasMaxT)

    def renderStep(self, bdrySolver):
        self.render_frame()

        if len(bdrySolver.colliders):
            self.render_collider(bdrySolver)
