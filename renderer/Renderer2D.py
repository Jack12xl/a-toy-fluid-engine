import taichi as ti
import taichi_glsl as ts
from .abstractRenderer import renderer
from config import PixelType, VisualizeEnum
from utils import cmapper

@ti.data_oriented
class renderer2D(renderer):
    def __init__(self, cfg, grid):
        super(renderer2D, self).__init__(cfg, grid)
        self.mapper = cmapper()

    @ti.kernel
    def render_collider(self, bdrySolver: ti.template()):
        for I in ti.grouped(self.clr_bffr):
            if bdrySolver.marker_field[I] == int(PixelType.Collider):
                for it in ti.static(range(len(bdrySolver.colliders))):
                    # clld = bdrySolver.colliders[0]
                    # TODO render function should be optimized
                    clld = bdrySolver.colliders[it]
                    if clld.is_inside_collider(I):
                        self.clr_bffr[I] = clld.color_at_world(I)

    @ti.kernel
    def vis_density(self, vf: ti.template()):
        ti.cache_read_only(vf.field)
        for I in ti.grouped(vf.field):
            self.clr_bffr[I] = ti.abs(vf[I])

    @ti.kernel
    def vis_v(self, vf: ti.template()):
        # velocity
        for I in ti.grouped(vf):
            v = ts.vec(vf[I].x, vf[I].y, 0.0)
            # self.clr_bffr[I] = ti.Vector([abs(v[0]), abs(v[1]), 0.0])
            self.clr_bffr[I] = 0.01 * v + ts.vec3(0.5)

    @ti.kernel
    def vis_v_mag(self, vf: ti.template()):
        # velocity magnitude
        for I in ti.grouped(vf):
            v_norm = vf[I].norm() * 0.004
            self.clr_bffr[I] = self.mapper.color_map(v_norm)

    @ti.kernel
    def vis_vd(self, vf: ti.template()):
        # divergence
        for I in ti.grouped(vf):
            v = ts.vec(vf[I], 0.0, 0.0)
            self.clr_bffr[I] = 0.3 * v + ts.vec3(0.5)

    @ti.kernel
    def vis_vt(self, vf: ti.template()):
        # visualize vorticity
        for I in ti.grouped(vf):
            v = ts.vec(vf[I], 0.0, 0.0)
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