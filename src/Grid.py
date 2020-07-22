import taichi as ti

@ti.data_oriented
class Grid():
    def __init__(self, cfg : dict, ):
        self.cfg = cfg

        self.v = ti.Vector(cfg['dim'],  dt=ti.f32, shape=cfg['res'])
        self.new_v = ti.Vector(cfg['dim'], dt=ti.f32, shape=cfg['res'])
        self.v_divs = ti.var(dt=ti.f32, shape=cfg['res'])

        self.p = ti.var(dt=ti.f32, shape=cfg['res'])
        self.new_p = ti.var(dt=ti.f32, shape=cfg['res'])

        self.clr_bffr = ti.Vector(3, dt=ti.f32, shape=cfg['res'])
        self.dye_bffr = ti.Vector(3, dt=ti.f32, shape=cfg['res'])
        self.new_dye_bffr = ti.Vector(3, dt=ti.f32, shape=cfg['res'])

        pass

    def sample(self):
        pass

