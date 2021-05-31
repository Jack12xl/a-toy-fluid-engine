# preprocess the data
import taichi as ti
import taichi_glsl as ts
import numpy as np
import os
from glob import glob
from utils import Float, Matrix
from Grid import CellGrid, Wrapper


@ti.data_oriented
class dataPPer:
    """
    Assume collocated Grid
    """

    def __init__(self,
                 high_path: str,
                 save_path: str,
                 dry_run: bool = True):
        # read
        self.high_V_paths = sorted(glob(os.path.join(high_path, '*.npz')))
        self.save_path = save_path

        self.num = len(self.high_V_paths)
        print(f"We have {self.num} pair datasets.")

        self.dry_run = dry_run

        tmp_high_v = np.load(self.high_V_paths[0])['v']
        self.dim = tmp_high_v.shape[-1]
        self.high_res = tmp_high_v.shape[:-1]
        self.low_res = tuple([r // 2 for r in self.high_res])
        print(f"Target Low resolution: {self.low_res}")
        print(f"High resolution: {self.high_res}")

        self.clr_bffr = ti.Vector.field(3, dtype=Float, shape=self.low_res)

        self.h_v, self.rho = [
            CellGrid(
                ti.Vector.field(self.dim, dtype=Float, shape=self.high_res),
                dim=self.dim,
                dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res),
                o=ts.vecND(self.dim, 0.0)
            )
            for _ in range(2)
        ]
        self.h_dn_v, self.l_v = [
            CellGrid(
                ti.Vector.field(self.dim, dtype=Float, shape=self.low_res),
                dim=self.dim,
                dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.low_res),
                o=ts.vecND(self.dim, 0.0)
            )
            for _ in range(2)
        ]

        self.inv_d = ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res)
        # in fact v_curl stores the weight of curl
        self.v_curl = CellGrid(
            ti.field(dtype=Float, shape=self.high_res),
            dim=self.dim,
            dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res),
            o=ts.vecND(self.dim, 0.0)
        )

        self.gui = ti.GUI("test", self.low_res, fast_gui=False)

    def pp(self):
        """
        Pre process
        :return:
        """
        fn_root = os.path.join(self.save_path, "dataset")
        os.makedirs(os.path.join(fn_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(fn_root, "visualize"), exist_ok=True)
        for i in range(self.num):
            h_v = np.load(self.high_V_paths[i])['v']

            self.h_v.field.from_numpy(h_v)

            self.h2l(self.h_dn_v, self.h_v)

            fn = os.path.join(fn_root, "raw", f"{i:06d}.npz")
            self.save_results(fn)

            self.gui.set_image(self.clr_bffr)
            # self.gui.circles(center_pos / 512, radius=1.0, color=0xED553B)

            vis_fn = os.path.join(fn_root, "visualize", f"{i:06d}.png")
            self.gui.show(vis_fn)

    def save_results(self, fn: str):
        """
        Save upsampling results, selected patch_centers

        :return:
        """
        np_down_v = self.h_dn_v.field.to_numpy()

        np.savez_compressed(fn,
                            dn_v=np_down_v,
                            )

    @ti.kernel
    def h2l(self, h2l_f: Wrapper, h_f: Wrapper):
        for I_h in ti.static(h2l_f):
            W = h2l_f.getW(I_h)
            # to low resolution grid
            # interpolate
            h2l_f[I_h] = h_f.interpolate(W)
            # print("W: ", W, " I_l: ", I_l, " I_h: ", I_h, " v: ", l2h_f[I_h])
            self.clr_bffr[I_h] = ts.vec3(h2l_f[I_h].x, h2l_f[I_h].y, 0.0) + ts.vec3(0.5)

    @ti.kernel
    def vis_v(self, l_f: Wrapper):
        for I in ti.static(l_f):
            self.clr_bffr[I] = ts.vec3(l_f[I].x, l_f[I].y, 0.0) + ts.vec3(0.5)


if __name__ == "__main__":
    ti.init(ti.gpu)
    h_path = "./tmp_result/05-31-22-58-20-Euler2D-512x512-UniformGrid-AR-MCS-JPS-64it-RK3-curl0.0-dt-0.03/v512x512"
    # l_path = "./tmp_result/01-04-17-39-21-Euler2D-128x128-UniformGrid-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03/v128x128"
    save_path = "./tmp_result/05-31-22-58-20-Euler2D-256x256-UniformGrid-AP-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03"
    pper = dataPPer(h_path, save_path, False)
    pper.pp()
