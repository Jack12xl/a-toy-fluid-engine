import taichi as ti
import taichi_glsl as ts
from .abstractRenderer import renderer
from config import PixelType, VisualizeEnum
from utils import cmapper

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
        self.dim = ts.vec3(cfg.dim)

    @ti.kernel
    def render_collider(self, bdrySolver: ti.template()):
        pass

    @ti.kernel
    def vis_density(self, vf: ti.template()):
        for I in ti.grouped(ti.ndrange(self.dim.x, self.dim.y, (self.z_plane, self.z_plane + 1))):
            self.clr_bffr[I] = ti.abs(vf[I])

    @ti.kernel
    def vis_v(self, vf: ti.template()):
        # velocity
        for I in ti.grouped(ti.ndrange(self.dim.x, self.dim.y, (self.z_plane, self.z_plane + 1))):
            self.clr_bffr[I] = 0.01 * vf[I] + ts.vec3(0.5)

    @ti.kernel
    def vis_v_mag(self, vf: ti.template()):
        # velocity magnitude
        for I in ti.grouped(ti.ndrange(self.dim.x, self.dim.y, (self.z_plane, self.z_plane + 1))):
            v_norm = vf[I].norm() * 0.004
            self.clr_bffr[I.xy] = self.mapper.color_map(v_norm)

    @ti.kernel
    def vis_vd(self, vf: ti.template()):
        # divergence
        for I in ti.grouped(ti.ndrange(self.dim.x, self.dim.y, (self.z_plane, self.z_plane + 1))):
            v = ts.vec(vf[I], 0.0, 0.0)
            self.clr_bffr[I.xy] = 0.3 * v + ts.vec3(0.5)

    @ti.kernel
    def vis_vt(self, vf: ti.template()):
        # visualize vorticity
        for I in ti.grouped(ti.ndrange(self.dim.x, self.dim.y, (self.z_plane, self.z_plane + 1))):
            v = ts.vec3(vf[I], 0.0)
            self.clr_bffr[I] = 0.03 * v + ts.vec3(0.5)

    def render_frame(self):
        if self.cfg.VisualType == VisualizeEnum.Velocity:
            self.vis_v(self.grid.v_pair.cur.field)
        elif self.cfg.VisualType == VisualizeEnum.Density:
            self.vis_density(self.grid.density_pair.cur)
        elif self.cfg.VisualType == VisualizeEnum.Divergence:
            self.vis_vd(self.grid.v_divs.field)
        elif self.cfg.VisualType == VisualizeEnum.Vorticity:
            self.vis_vt(self.grid.v_curl.field)
        elif self.cfg.VisualType == VisualizeEnum.VelocityMagnitude:
            self.vis_v_mag(self.grid.v.field)

    def renderStep(self, bdrySolver):
        self.render_frame()

        if len(bdrySolver.colliders):
            self.render_collider(bdrySolver)
