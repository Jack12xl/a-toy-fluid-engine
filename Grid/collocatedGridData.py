import taichi as ti
from utils import bufferPair, clamp, lerp, Vector, Matrix
from .DataGrid import DataGrid
import taichi_glsl as ts


@ti.data_oriented
class collocatedGridData():
    '''
    class to store the grid data
    pressure, velocity both stores on grid cell corner
    '''

    # ref:https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
    def __init__(self, cfg, ):
        self.cfg = cfg
        self.dim = self.cfg.dim

        self.v = DataGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res))
        self.new_v = DataGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res))
        # another buffer for advection-reflection
        self.tmp_v = DataGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res))
        # velocity divergence
        self.v_divs = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))
        # velocity vorticity
        self.v_curl = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))

        self.p = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))
        self.new_p = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))
        # here density is just for visualization, which does not involve in calculation
        self.density_bffr = DataGrid(ti.Vector.field(3, dtype=ti.f32, shape=cfg.res))
        self.new_density_bffr = DataGrid(ti.Vector.field(3, dtype=ti.f32, shape=cfg.res))

        self.v_pair = bufferPair(self.v, self.new_v)
        self.p_pair = bufferPair(self.p, self.new_p)
        self.density_pair = bufferPair(self.density_bffr, self.new_density_bffr)
        # self.marker_pair = TexPair(self.marker, self.new_marker)

        if self.dim == 2:
            self.calVorticity = self.calVorticity2D
    @ti.kernel
    def calDivergence(self, vf: ti.template(), vd: ti.template()):
        for I in ti.grouped(vf.field):
            ret = 0.0
            vc = vf.sample(I)
            for d in ti.static(range(self.cfg.dim)):
                D = ti.Vector.unit(self.cfg.dim, d)
                v0 = vf.sample(I + D)[d]
                v1 = vf.sample(I - D)[d]
                # TODO boundary condition
                if I[d] == 0:
                    v1 = -vc[d]
                if I[d] == vf.shape[d] - 1:
                    v0 = -vc[d]
                ret += v0 - v1

            vd[I] = ret * 0.5

    @ti.kernel
    def calVorticity2D(self, vf: Matrix):
        for I in ti.grouped(vf.field):
            vl = vf.sample(I + ts.D.zy).y
            vr = vf.sample(I + ts.D.xy).y
            vb = vf.sample(I + ts.D.yz).x
            vt = vf.sample(I + ts.D.yx).x
            self.v_curl[I] = (vr - vl - vt + vb) * 0.5

    @ti.kernel
    def calVorticity3D(self, vf: Matrix):
        for I in ti.grouped(vf.field):
            pass

    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        for I in ti.grouped(pf.field):
            ret = ts.vecND(self.dim, 0.0)
            for d in ti.static(range(self.dim)):
                D = ti.Vector.unit(self.dim, d)

                p0 = pf.sample(I + D)
                p1 = pf.sample(I - D)

                ret[d] = p0 - p1
            vf[I] -= 0.5 * ret
            # pl = pf.sample(I + ts.D.zy)
            # pr = pf.sample(I + ts.D.xy)
            # pb = pf.sample(I + ts.D.yz)
            # pt = pf.sample(I + ts.D.yx)
            # vf[I] -= 0.5 * ts.vec(pr - pl, pt - pb)

    def subtract_gradient_pressure(self):
        self.subtract_gradient(self.v_pair.cur, self.p_pair.cur)

    def reset(self):
        self.v_pair.cur.fill(ti.Vector([0, 0]))
        self.p_pair.cur.fill(0.0)
        self.density_pair.cur.fill(ti.Vector([0, 0, 0]))
