from .Euler_Scheme import EulerScheme
import taichi as ti
import numpy as np

# for super resolution
# input velocity field and advect
@ti.data_oriented
class AdvectV(EulerScheme):
    def __init__(self, cfg):
        super().__init__(cfg)

    def schemeStep(self, ext_input: np.array):
        """

        :param ext_input: as the input velocity field
        :return:
        """
        self.grid.v_pair.cur.field.from_numpy(ext_input)
        self.advect(self.cfg.dt)