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

        self.v = DataGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res))
        self.new_v = DataGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res))
        # another buffer for advection-reflection
        self.tmp_v = DataGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res))
        # velocity divergence
        self.v_divs = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))
        #velocity vorticity
        self.v_curl = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))

        self.p = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))
        self.new_p = DataGrid(ti.field(dtype=ti.f32, shape=cfg.res))
        # here density is just for visualization, which does not involve in calculation
        self.density_bffr = DataGrid(ti.Vector.field(3, dtype=ti.f32, shape=cfg.res))
        self.new_density_bffr = DataGrid(ti.Vector.field(3, dtype=ti.f32, shape=cfg.res))

        # self.marker = ti.field(dtype=ti.i32, shape=cfg.res)
        # self.new_marker = ti.field(dtype=ti.i32, shape=cfg.res)

        self.v_pair = bufferPair(self.v, self.new_v)
        self.p_pair = bufferPair(self.p, self.new_p)
        self.density_pair = bufferPair(self.density_bffr, self.new_density_bffr)
        # self.marker_pair = TexPair(self.marker, self.new_marker)


    @ti.func
    def sample(self, qf, u, v):
        raise DeprecationWarning
        # assure integer
        # i, j = int(u), int(v)
        I = int( ti.Vector([u, v]) )
        # clamp
        # i = clamp(i, 0, self.cfg.res[0] - 1)
        # j = clamp(j, 0, self.cfg.res[1] - 1)
        I = max( 0, min(self.cfg.res[0] - 1, I) )

        return qf[I]

    # @ti.func
    # def incell2grid(self, phy_coord:ti.template()) -> ti.template():
    #     grid_coord = phy_coord / self.cfg.dx - 0.5
    #     return grid_coord.cast(ti.i32)

    @ti.func
    def bilerp(self, values, phy_coord):
        '''
        get corresponding value given physical coordinate
        :param values: value field
        :param phy_coord:
        :return:
        '''
        raise DeprecationWarning
        # get coordinate in grid
        #TODO handle non-suqare cell
        grid_coord = phy_coord - 0.5
        #TODO handle 3D input

        # int -> floor https://github.com/taichi-dev/taichi/pull/1784
        iu, iv = ti.floor(grid_coord[0]), ti.floor(grid_coord[1])
        # fract
        fu, fv = grid_coord[0] - iu, grid_coord[1] - iv
        #TODO test +0, +1
        a = self.sample(values, iu + 0.5, iv + 0.5)
        b = self.sample(values, iu + 1.5, iv + 0.5)
        c = self.sample(values, iu + 0.5, iv + 1.5)
        d = self.sample(values, iu + 1.5, iv + 1.5)

        return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

    @ti.func
    def sample_minmax(self, vf, p):
        raise DeprecationWarning
        u, v = p
        s, t = u - 0.5, v - 0.5
        # int -> floor https://github.com/taichi-dev/taichi/pull/1784
        iu, iv = ti.floor(s), ti.floor(t)
        a = self.sample(vf, iu + 0.5, iv + 0.5)
        b = self.sample(vf, iu + 1.5, iv + 0.5)
        c = self.sample(vf, iu + 0.5, iv + 1.5)
        d = self.sample(vf, iu + 1.5, iv + 1.5)
        return min(a, b, c, d), max(a, b, c, d)


    @ti.kernel
    def calDivergence(self, vf: ti.template(), vd: ti.template()):
        for I in ti.grouped(vf.field):
            # vl = self.sample(vf, i - 1, j)[0]
            # vr = self.sample(vf, i + 1, j)[0]
            # vb = self.sample(vf, i, j - 1)[1]
            # vt = self.sample(vf, i, j + 1)[1]
            # vc = self.sample(vf, i, j)
            vl = vf.sample(I + ts.D.zy).x
            vr = vf.sample(I + ts.D.xy).x
            vb = vf.sample(I + ts.D.yz).y
            vt = vf.sample(I + ts.D.yx).y
            vc = vf.sample(I)
            # boundary
            #TODO
            # if i == 0:
            #     vl = -vc[0]
            # if i == self.cfg.res[0] - 1:
            #     vr = -vc[0]
            # if j == 0:
            #     vb = -vc[1]
            # if j == self.cfg.res[1] - 1:
            #     vt = -vc[1]
            # vd[i, j] = (vr - vl + vt - vb) * self.cfg.half_inv_dx
            if I.x == 0:
                vl = -vc.x
            if I.x == vf.shape[0] - 1:
                vr = -vc.x
            if I.y == 0:
                vb = -vc.y
            if I.y == vf.shape[1] - 1:
                vt = -vc.y
            vd[I] = (vr - vl + vt - vb) * 0.5

    @ti.kernel
    # ref: taichi official stable fluid
    def calVorticity(self, vf: Vector):
        for I in ti.grouped(vf.field):
            vl = vf.sample(I + ts.D.zy).x
            vr = vf.sample(I + ts.D.xy).x
            vb = vf.sample(I + ts.D.yz).y
            vt = vf.sample(I + ts.D.yx).y
            self.v_curl[I] = (vr - vl - vt + vb) * 0.5

    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        # for i, j in vf:
        #     pl = self.sample(pf, i - 1, j)
        #     pr = self.sample(pf, i + 1, j)
        #     pb = self.sample(pf, i, j - 1)
        #     pt = self.sample(pf, i, j + 1)
        #
        #     vf[i, j] = self.sample(vf, i, j) - \
        #                self.cfg.half_inv_dx * ti.Vector([pr - pl, pt - pb])
        ti.cache_read_only(pf.field)
        for I in ti.grouped(pf.field):
            pl = pf.sample(I + ts.D.zy)
            pr = pf.sample(I + ts.D.xy)
            pb = pf.sample(I + ts.D.yz)
            pt = pf.sample(I + ts.D.yx)
            vf[I] -= 0.5 * ts.vec(pr - pl, pt - pb)


    def subtract_gradient_pressure(self):
        self.subtract_gradient(self.v_pair.cur, self.p_pair.cur)

    def reset(self):
        self.v_pair.cur.fill(ti.Vector([0, 0]))
        self.p_pair.cur.fill(0.0)
        self.density_pair.cur.fill(ti.Vector([0, 0, 0]))