from .Grid import Grid
import taichi as ti

@ti.data_oriented
class EulerScheme():
    def __init__(self, cfg:dict, ):
        self.cfg = cfg
        self.grid = Grid(cfg)



    def advect(self):
        pass

    def project(self):
        pass


    def step(self):
        pass