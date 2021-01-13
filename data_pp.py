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
                 low_path: str,
                 high_path: str,
                 save_path: str,
                 dry_run: bool = True):
        # read
        self.low_V_paths = sorted(glob(os.path.join(low_path, '*.npz')))
        self.high_V_paths = sorted(glob(os.path.join(high_path, '*.npz')))
        self.save_path = save_path
        self.dry_run = dry_run

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

        self.l_up_v, self.h_v, self.rho = [
            CellGrid(
                ti.Vector.field(self.dim, dtype=Float, shape=self.high_res),
                dim=self.dim,
                dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res),
                o=ts.vecND(self.dim, 0.0)
            )
            for _ in range(3)
        ]
        self.l_v = CellGrid(
            ti.Vector.field(self.dim, dtype=Float, shape=self.low_res),
            dim=self.dim,
            dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.low_res),
            o=ts.vecND(self.dim, 0.0)
        )

        self.inv_d = ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res)
        # in fact v_curl stores the weight of curl
        self.v_curl = CellGrid(
            ti.field(dtype=Float, shape=self.high_res),
            dim=self.dim,
            dx=ts.vecND(self.dim, 1.0) / ti.Vector(self.high_res),
            o=ts.vecND(self.dim, 0.0)
        )
        if self.dim == 2:
            self.calVorticity = self.calVorticity2D

        elif self.dim == 3:
            self.calVorticity = self.calVorticity3D

        self.gui = ti.GUI("test", self.high_res, fast_gui=False)

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
            l_v = np.load(self.low_V_paths[i])['v']

            self.l_v.field.from_numpy(l_v)
            self.h_v.field.from_numpy(h_v)

            self.l2h(self.l_up_v, self.l_v)

            self.calVorticity(self.h_v)
            center_pos = self.Kai_Sampling()

            if not self.dry_run:
                fn = os.path.join(fn_root, "raw", f"{i:06d}.npz")
                self.save_results(fn, center_pos)

            self.gui.set_image(self.clr_bffr)
            self.gui.circles(center_pos / 512, radius=1.0, color=0xED553B)

            vis_fn = os.path.join(fn_root, "visualize", f"{i:06d}.png")
            self.gui.show(vis_fn if not self.dry_run else None)

    def save_results(self, fn: str, patch_centers: np.array):
        """
        Save upsampling results, selected patch_centers

        :return:
        """
        np_high_v = self.h_v.field.to_numpy()
        np_up_v = self.l_up_v.field.to_numpy()

        np.savez_compressed(fn,
                            up_v=np_up_v,
                            h_v=np_high_v,
                            c_pos=patch_centers
                            )

    @ti.kernel
    def l2h(self, l2h_f: Wrapper, l_f: Wrapper):
        for I_h in ti.static(l2h_f):
            W = l2h_f.getW(I_h)
            # to low resolution grid
            # interpolate
            l2h_f[I_h] = l_f.interpolate(W)
            # print("W: ", W, " I_l: ", I_l, " I_h: ", I_h, " v: ", l2h_f[I_h])
            self.clr_bffr[I_h] = ts.vec3(l2h_f[I_h].x, l2h_f[I_h].y, 0.0) + ts.vec3(0.5)

    @ti.kernel
    def calVorticity2D(self, vf: Matrix):
        curl_all = 0.0
        for I in ti.static(vf):
            vl = vf.sample(I + ts.D.zy).y
            vr = vf.sample(I + ts.D.xy).y
            vb = vf.sample(I + ts.D.yz).x
            vt = vf.sample(I + ts.D.yx).x

            res = ts.vec2(vr - vl, vb - vt) * self.inv_d
            curl = abs(res.sum())

            self.v_curl[I] = curl
            curl_all += curl

        # normalize
        for I in ti.static(vf):
            self.v_curl[I] /= (curl_all + 1e-6)

    @ti.kernel
    def calVorticity3D(self, vf: Matrix):
        curl_all = 0.0
        for I in ti.static(vf):
            curl = ts.vec3(0.0)
            # left & right
            v_l = vf.sample(I + ts.D.zyy)
            v_r = vf.sample(I + ts.D.xyy)
            # top & down
            v_t = vf.sample(I + ts.D.yxy)
            v_d = vf.sample(I + ts.D.yzy)
            # forward & backward
            v_f = vf.sample(I + ts.D.yyx)
            v_b = vf.sample(I + ts.D.yyz)

            curl[0] = (v_f.y - v_b.y) - (v_t.z - v_d.z)
            curl[1] = (v_r.z - v_l.z) - (v_f.x - v_b.x)
            curl[2] = (v_t.x - v_d.x) - (v_r.y - v_l.y)

            res = (curl * self.inv_d).norm()
            curl_all += res
            self.v_curl[I] = res

        # normalize
        for I in ti.static(vf):
            self.v_curl[I] /= (curl_all + 1e-6)

    def Kai_Sampling(self):
        """
        Calculate the index of patch center
        :return: an np.array idx_num x dim
        """
        curl_weight = self.v_curl.field.to_numpy()

        size = np.array(curl_weight.shape).prod()
        idx = np.arange(size)

        curl_weight = curl_weight.reshape([-1])

        idx_num = size // pow(5, self.dim)
        fluid_idx_num = int(0.95 * idx_num)
        non_fluid_idx_num = idx_num - fluid_idx_num
        # sample without repetition
        fluid_res = np.random.choice(idx, fluid_idx_num, replace=False, p=curl_weight)

        non_fluid_idx = np.delete(idx, fluid_res)
        non_fluid_res = np.random.choice(non_fluid_idx, non_fluid_idx_num, replace=False, p=None)

        def unravel(idx):
            return np.unravel_index(idx, self.high_res)

        unravel_vec = np.vectorize(unravel)

        res = np.concatenate((fluid_res, non_fluid_res))
        res = unravel_vec(res)
        res = np.swapaxes(res, 0, 1)

        return res

    @ti.kernel
    def vis_v(self, l_f: Wrapper):
        for I in ti.static(l_f):
            self.clr_bffr[I] = ts.vec3(l_f[I].x, l_f[I].y, 0.0) + ts.vec3(0.5)


if __name__ == "__main__":
    ti.init(ti.gpu)
    h_path = "./tmp_result/01-04-18-10-07-Euler2D-512x512-UniformGrid-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03/v512x512"
    l_path = "./tmp_result/01-04-17-39-21-Euler2D-128x128-UniformGrid-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03/v128x128"
    save_path = "./tmp_result/01-04-17-39-21-Euler2D-128x128-UniformGrid-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03"
    pper = dataPPer(l_path, h_path, save_path, False)
    pper.pp()
