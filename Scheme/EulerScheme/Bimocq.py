import taichi as ti
from .Euler_Scheme import EulerScheme


# ref Bimocq 2019, By qi ziyin

class Bimocq_Scheme(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)
        pass

    def advect(self, dt):
        pass

    def advectDMC(self, dt):
        pass

    def updateBackward(self):
        pass

    def updateForward(self):
        pass

