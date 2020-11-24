import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix
from .CellGrid import CellGrid
from .FaceGrid import FaceGrid
from config import SimulateType
from .FluidGridData import FluidGridData


@ti.data_oriented
class MacGridData(FluidGridData):
    """
    staggered grid
    vector on the grid face
    """

    def __init__(self, cfg):
        super(MacGridData, self).__init__(cfg)
        # the distance between two neighbour when calculating divergence, vorticity
        self.inv_d = 1.0 / self.cfg.dx

        self.v = FaceGrid(ti.f32,
                          shape=cfg.res,
                          dim=cfg.dim,
                          dx=ts.vecND(self.dim, self.cfg.dx),
                          o=ts.vecND(self.dim, 0.5)
                          )
        self.new_v = FaceGrid(ti.f32,
                              shape=cfg.res,
                              dim=cfg.dim,
                              dx=ts.vecND(self.dim, self.cfg.dx),
                              o=ts.vecND(self.dim, 0.5)
                              )
        # buffer for advection-reflection
        self.tmp_v = FaceGrid(ti.f32,
                              shape=cfg.res,
                              dim=cfg.dim,
                              dx=ts.vecND(self.dim, self.cfg.dx),
                              o=ts.vecND(self.dim, 0.5)
                              )

        self.v_pair = bufferPair(self.v, self.new_v)
        self.p_pair = bufferPair(self.p, self.new_p)
        self.density_pair = bufferPair(self.density_bffr, self.new_density_bffr)
        self.t_pair = bufferPair(self.t, self.t_bffr)

        for d in range(self.dim):
            self.advect_v_pairs.append(
                bufferPair(self.v.fields[d], self.new_v.fields[d])
            )

    @ti.kernel
    def calDivergence(self, vf: ti.template(), vd: ti.template()):
        """

        :param vf: a facegrid
        :param vd:
        :return:
        """
        for I in ti.static(vf):
            ret = 0.0
            for d in ti.static(range(self.dim)):
                D = ti.Vector.unit(self.dim, d)
                v0 = vf.fields[d].sample(I + D)[0]
                v1 = vf.fields[d].sample(I)[0]
                # TODO boundary
                ret += v0 - v1
            vd[I] = ret * self.inv_d

    @ti.kernel
    def calVorticity2D(self, vf: Matrix):
        for I in ti.static(vf):
            # y
            vl = vf.fields[1].sample(I)[0]
            vr = vf.fields[1].sample(I + ts.D.xy)[0]
            # x
            vb = vf.fields[0].sample(I)[0]
            vt = vf.fields[0].sample(I + ts.D.yx)[0]
            self.v_curl[I] = (vr - vl - vt + vb) * self.inv_d

    @ti.kernel
    def calVorticity3D(self, vf: Matrix):
        for I in ti.static(vf):
            curl = ts.vec3(0.0)
            # left & right
            v_l = vf.fields[0].sample(I)
            v_r = vf.fields[0].sample(I + ts.D.xyy)
            # top & down
            v_t = vf.fields[1].sample(I)
            v_d = vf.fields[1].sample(I + ts.D.yzy)
            # forward & backward
            v_f = vf.fields[2].sample(I)
            v_b = vf.fields[2].sample(I + ts.D.yyz)

            curl[0] = (v_f.y - v_b.y) - (v_t.z - v_d.z)
            curl[1] = (v_r.z - v_l.z) - (v_f.x - v_b.x)
            curl[2] = (v_t.x - v_d.x) - (v_r.y - v_l.y)

            self.v_curl[I] = curl

    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        for d in ti.static(range(self.dim)):
            D = ti.Vector.unit(self.dim, d)
            for I in ti.static(pf):
                # TODO do not handle the right most boundary
                # but both left, right boundary gradient is zero anyway
                p0 = pf.sample(I)
                p1 = pf.sample(I - D)
                vf.fields[d][I][0] -= (p0 - p1) * self.inv_d

        # for I in ti.static(pf):
        #     for d in ti.static(range(self.dim)):
        #         D = ti.Vector.unit(self.dim, d)
        #
        #         p0 = pf.sample(I)
        #         p1 = pf.sample(I - D)
        #         # subtract delta_P / dx
        #         vf.fields[d][I][0] -= (p0 - p1) * self.inv_d

    def subtract_gradient_pressure(self):
        self.subtract_gradient(self.v_pair.cur, self.p_pair.cur)

    @ti.kernel
    def copy_v_field(self,
                     dst: ti.template(),
                     trgt: ti.template()):
        for d in ti.static(range(self.dim)):
            # dst_d = ti.static(dst.fields[d])
            # trgt_d = ti.static(trgt.fields[d])
            for I in ti.static(dst.fields[d]):
                dst.fields[d][I] = trgt.fields[d][I]

    @ti.kernel
    def reflect_v_field(self,
                        to_be_reflected: Matrix,
                        mid_point: Matrix):
        for d in ti.static(range(self.dim)):
            for I in ti.static(to_be_reflected.fields[d]):
                to_be_reflected.fields[d][I] = 2.0 * mid_point.fields[d][I] - to_be_reflected.fields[d][I]
