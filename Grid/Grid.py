import taichi as ti
from utils import TexPair, clamp, lerp
@ti.data_oriented
class Grid():
    # ref:https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
    def __init__(self, cfg, ):
        self.cfg = cfg

        self.v = ti.Vector.field(cfg.dim,  dtype=ti.f32, shape=cfg.res)
        self.new_v = ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res)
        self.v_divs = ti.field(dtype=ti.f32, shape=cfg.res)
        self.tmp_v = ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res)

        self.p = ti.field(dtype=ti.f32, shape=cfg.res)
        self.new_p = ti.field(dtype=ti.f32, shape=cfg.res)

        self.density_bffr = ti.Vector.field(3, dtype=ti.f32, shape=cfg.res)
        self.new_density_bffr = ti.Vector.field(3, dtype=ti.f32, shape=cfg.res)

        self.marker = ti.Vector.field(3, dtype=ti.i32, shape=cfg.res)

        self.v_pair = TexPair(self.v, self.new_v)
        self.p_pair = TexPair(self.p, self.new_p)
        self.density_pair = TexPair(self.density_bffr, self.new_density_bffr)



    @ti.func
    def sample(self, qf, u, v):
        # assure integer
        i, j = int(u), int(v)
        # clamp
        i = clamp(i, 0, self.cfg.res[0] - 1)
        j = clamp(j, 0, self.cfg.res[1] - 1)

        return qf[i, j]

    @ti.func
    def incell2grid(self, phy_coord:ti.template()) -> ti.template():
        grid_coord = phy_coord / self.cfg.dx - 0.5
        return grid_coord.cast(ti.i32)

    @ti.func
    def interpolate_value(self, values, phy_coord):
        '''
        get corresponding value given physical coordinate
        :param values: value field
        :param phy_coord:
        :return:
        '''

        # get coordinate in grid
        #TODO handle non-suqare cell
        grid_coord = phy_coord / self.cfg.dx - 0.5
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
        for i, j in vf:
            vl = self.sample(vf, i - 1, j)[0]
            vr = self.sample(vf, i + 1, j)[0]
            vb = self.sample(vf, i, j - 1)[1]
            vt = self.sample(vf, i, j + 1)[1]
            vc = self.sample(vf, i, j)
            # boundary
            #TODO
            if i == 0:
                vl = -vc[0]
            if i == self.cfg.res[0] - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            if j == self.cfg.res[1] - 1:
                vt = -vc[1]
            vd[i, j] = (vr - vl + vt - vb) * self.cfg.half_inv_dx


    @ti.kernel
    def Jacobi_Step(self, pf: ti.template(), new_pf: ti.template(), p_divs: ti.template()):
        for i, j in pf:
            pl = self.sample(pf, i - 1, j)
            pr = self.sample(pf, i + 1, j)
            pb = self.sample(pf, i, j - 1)
            pt = self.sample(pf, i, j + 1)
            div = p_divs[i, j]
            new_pf[i, j] = (pl + pr + pb + pt + self.cfg.jacobi_alpha * div) * self.cfg.jacobi_beta

    def Jacobi_run_pressure(self):
        self.cfg.jacobi_alpha = self.cfg.poisson_pressure_alpha
        self.cfg.jacobi_beta = self.cfg.poisson_pressure_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Jacobi_Step(self.p_pair.cur, self.p_pair.nxt, self.v_divs,)
            self.p_pair.swap()

    def Jacobi_run_viscosity(self):
        self.cfg.jacobi_alpha = self.cfg.poisson_viscosity_alpha
        self.cfg.jacobi_beta = self.cfg.poisson_viscosity_beta
        for _ in range(self.cfg.p_jacobi_iters):
            self.Jacobi_Step(self.p_pair.cur, self.p_pair.nxt, self.v_divs, )
            self.p_pair.swap()


    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        for i, j in vf:
            pl = self.sample(pf, i - 1, j)
            pr = self.sample(pf, i + 1, j)
            pb = self.sample(pf, i, j - 1)
            pt = self.sample(pf, i, j + 1)

            vf[i, j] = self.sample(vf, i, j) - \
                       self.cfg.half_inv_dx * ti.Vector([pr - pl, pt - pb])

    def subtract_gradient_pressure(self):
        self.subtract_gradient(self.v_pair.cur, self.p_pair.cur)

    def reset(self):
        self.v_pair.cur.fill(ti.Vector([0, 0]))
        self.p_pair.cur.fill(0.0)
        self.density_pair.cur.fill(ti.Vector([0, 0, 0]))