import taichi as ti
from utils import TexPair, clamp, lerp
@ti.data_oriented
class Grid():
    # ref:https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
    def __init__(self, cfg, ):
        self.cfg = cfg

        self.v = ti.Vector(cfg.dim,  dt=ti.f32, shape=cfg.res)
        self.new_v = ti.Vector(cfg.dim, dt=ti.f32, shape=cfg.res)
        self.v_divs = ti.var(dt=ti.f32, shape=cfg.res)

        self.p = ti.var(dt=ti.f32, shape=cfg.res)
        self.new_p = ti.var(dt=ti.f32, shape=cfg.res)

        self.dye_bffr = ti.Vector(3, dt=ti.f32, shape=cfg.res)
        self.new_dye_bffr = ti.Vector(3, dt=ti.f32, shape=cfg.res)

        self.v_pair = TexPair(self.v, self.new_v)
        self.p_pair = TexPair(self.p, self.new_p)
        self.dye_pair = TexPair(self.dye_bffr, self.new_dye_bffr)
        pass

    @ti.func
    def sample(self, qf, u, v):
        # assure integer
        i, j = int(u), int(v)
        # clamp
        i = clamp(i, 0, self.cfg.res[0])
        j = clamp(j, 0, self.cfg.res[1])

        return qf[i, j]

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
        iu, iv = int(grid_coord[0]), int(grid_coord[1])
        # fract
        fu, fv = grid_coord[0] - iu, grid_coord[1] - iv
        #TODO test +0, +1
        a = self.sample(values, iu + 0.5, iv + 0.5)
        b = self.sample(values, iu + 1.5, iv + 0.5)
        c = self.sample(values, iu + 0.5, iv + 1.5)
        d = self.sample(values, iu + 1.5, iv + 1.5)

        return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

