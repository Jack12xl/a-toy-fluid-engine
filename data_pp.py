# preprocess the data
import taichi as ti
import taichi_glsl as ts
import numpy as np
import os
from glob import glob
from utils import Float
from Grid import CellGrid, Wrapper


@ti.data_oriented
class dataPPer:
    def __init__(self, low_path: str, high_path: str):
        # read
        self.low_V_paths = sorted(glob(os.path.join(low_path, '*.npz')))
        self.high_V_paths = sorted(glob(os.path.join(high_path, '*.npz')))
        assert (len(self.low_V_paths) == len(self.high_V_paths) and len(self.low_V_paths) > 0)
        self.num = len(self.low_V_paths)
        print(f"We have {self.num} pair datasets.")

        tmp_low_v = np.load(self.low_V_paths[0])['v']
        tmp_high_v = np.load(self.high_V_paths[0])['v']
        self.dim = tmp_low_v.shape[-1]
        self.low_res = tmp_low_v.shape[:-1]
        self.high_res = tmp_high_v.shape[:-1]
        print(f"Low resolution: {self.low_res}")
        print(f"High resolution: {self.high_res}")

        self.clr_bffr = ti.Vector.field(3, dtype=Float, shape=self.high_res)

        self.l_up_field, self.h_field = [
            CellGrid(
                ti.Vector.field(self.dim, dtype=Float, shape=self.high_res),
                dim=self.dim,
                dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res),
                o=ts.vecND(self.dim, 0.5)
            )
            for _ in range(2)
        ]
        self.l_field = CellGrid(
            ti.Vector.field(self.dim, dtype=Float, shape=self.low_res),
            dim=self.dim,
            dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.low_res),
            o=ts.vecND(self.dim, 0.5)
        )

        self.gui = ti.GUI("test", self.high_res, fast_gui=True)

    def pp(self):
        """
        Pre process
        :return:
        """
        # for i in range(self.num):
        #     h_v = np.load(self.high_V_paths[i])['v']
        #     l_v = np.load(self.low_V_paths[i])['v']
        #
        #     self.l_field.field.from_numpy(l_v)
        #     self.h_field.field.from_numpy(h_v)
        #
        #     self.l2h(self.l_up_field, self.l_field)
        #
        #     # self.vis_v(self.l_field)
        #     self.gui.set_image(self.clr_bffr)
        #     self.gui.show()
        i = 256
        h_v = np.load(self.high_V_paths[i])['v']
        l_v = np.load(self.low_V_paths[i])['v']

        self.l_field.field.from_numpy(l_v)
        self.h_field.field.from_numpy(h_v)

        self.l2h(self.l_up_field, self.l_field)

        # self.vis_v(self.l_field)
        self.gui.set_image(self.clr_bffr)
        self.gui.show()

    @ti.kernel
    def l2h(self, l2h_f: Wrapper, l_f: Wrapper):
        for I_h in ti.static(l2h_f):
            W = l2h_f.getW(I_h)
            # to low resolution grid
            I_l = l_f.getG(W)
            # interpolate
            l2h_f[I_h] = l_f.interpolate(I_l)
            print("W: ", W, " I_l: ", I_l, " I_h: ", I_h, " v: ", l2h_f[I_h])
            self.clr_bffr[I_h] = ts.vec3(l2h_f[I_h].x, l2h_f[I_h].y, 0.0) + ts.vec3(0.5)

    @ti.kernel
    def vis_v(self, l_f: Wrapper):
        for I in ti.static(l_f):
            self.clr_bffr[I] = ts.vec3(l_f[I].x, l_f[I].y, 0.0)

if __name__ == "__main__":
    ti.init(ti.gpu)
    h_path = "./tmp_result/01-04-18-10-07-Euler2D-512x512-UniformGrid-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03/v512x512"
    l_path = "./tmp_result/01-04-17-39-21-Euler2D-128x128-UniformGrid-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03/v128x128"
    pper = dataPPer(l_path, h_path)
    pper.pp()
