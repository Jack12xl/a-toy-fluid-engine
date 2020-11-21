import taichi as ti
from utils import bufferPair, Vector, Matrix
from .CellGrid import CellGrid
import taichi_glsl as ts
from config import SimulateType
from .FluidGridData import FluidGridData


@ti.data_oriented
class collocatedGridData(FluidGridData):
    """
    class to store the grid data
    pressure, velocity both stores on grid cell corner(cell centered grid)
    """

    # ref:https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
    def __init__(self, cfg, ):
        super(collocatedGridData, self).__init__(cfg)
        # the distance between two neighbour when calculating divergence, vorticity
        self.inv_d = 1.0 / (2 * self.cfg.dx)

        self.v = CellGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res), cfg.dim,
                          dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        self.new_v = CellGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res), cfg.dim,
                              dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        # another buffer for advection-reflection
        self.tmp_v = CellGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res), cfg.dim,
                              dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        # velocity divergence
        self.v_divs = CellGrid(ti.field(dtype=ti.f32, shape=cfg.res), cfg.dim, dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        # velocity vorticity
        if self.dim == 2:
            self.v_curl = CellGrid(ti.field(dtype=ti.f32, shape=cfg.res), cfg.dim, dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        elif self.dim == 3:
            self.v_curl = CellGrid(ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res), cfg.dim,
                                   dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))

        self.p = CellGrid(ti.field(dtype=ti.f32, shape=cfg.res), cfg.dim, dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        self.new_p = CellGrid(ti.field(dtype=ti.f32, shape=cfg.res), cfg.dim, dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        # here density is just for visualization, which does not involve in calculation
        self.density_bffr = CellGrid(ti.Vector.field(3, dtype=ti.f32, shape=cfg.res), cfg.dim,
                                     dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        self.new_density_bffr = CellGrid(ti.Vector.field(3, dtype=ti.f32, shape=cfg.res), cfg.dim,
                                         dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        # temperature
        self.t = CellGrid(ti.Vector.field(1, dtype=ti.f32, shape=cfg.res), cfg.dim, dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        self.t_bffr = CellGrid(ti.Vector.field(1, dtype=ti.f32, shape=cfg.res), cfg.dim,
                               dx=ts.vecND(self.dim, self.cfg.dx), o=ts.vecND(self.dim, 0.0))
        self.t_ambient = ti.field(dtype=ti.f32, shape=[])

        self.v_pair = bufferPair(self.v, self.new_v)
        self.p_pair = bufferPair(self.p, self.new_p)
        self.density_pair = bufferPair(self.density_bffr, self.new_density_bffr)
        self.t_pair = bufferPair(self.t, self.t_bffr)

        self.advect_pair = []

    @ti.kernel
    def calDivergence(self, vf: ti.template(), vd: ti.template()):
        for I in ti.static(vf):
            ret = 0.0
            vc = vf.sample(I)
            for d in ti.static(range(self.cfg.dim)):
                D = ti.Vector.unit(self.cfg.dim, d)
                v0 = vf.sample(I + D)[d]
                v1 = vf.sample(I - D)[d]
                # TODO boundary condition
                if I[d] == 0:
                    v1 = -vc[d]
                    # v1 = 0.0
                if I[d] == vf.shape[d] - 1:
                    v0 = -vc[d]
                    # v0 = 0.0
                ret += v0 - v1

            vd[I] = ret * self.inv_d

    @ti.kernel
    def calVorticity2D(self, vf: Matrix):
        for I in ti.static(vf):
            vl = vf.sample(I + ts.D.zy).y
            vr = vf.sample(I + ts.D.xy).y
            vb = vf.sample(I + ts.D.yz).x
            vt = vf.sample(I + ts.D.yx).x
            self.v_curl[I] = (vr - vl - vt + vb) * self.inv_d

    @ti.kernel
    def calVorticity3D(self, vf: Matrix):
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

            self.v_curl[I] = curl

    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        for I in ti.static(pf):
            ret = ts.vecND(self.dim, 0.0)
            for d in ti.static(range(self.dim)):
                D = ti.Vector.unit(self.dim, d)

                p0 = pf.sample(I + D)
                p1 = pf.sample(I - D)

                ret[d] = p0 - p1
            vf[I] -= self.inv_d * ret
            # pl = pf.sample(I + ts.D.zy)
            # pr = pf.sample(I + ts.D.xy)
            # pb = pf.sample(I + ts.D.yz)
            # pt = pf.sample(I + ts.D.yx)
            # vf[I] -= 0.5 * ts.vec(pr - pl, pt - pb)

    def subtract_gradient_pressure(self):
        self.subtract_gradient(self.v_pair.cur, self.p_pair.cur)

    def materialize(self):
        if self.cfg.SimType == SimulateType.Gas:
            self.t.fill(self.cfg.GasInitAmbientT)

    def reset(self):
        self.v_pair.cur.fill(ts.vecND(self.dim, 0.0))
        self.p_pair.cur.fill(0.0)
        self.density_pair.cur.fill(ti.Vector([0, 0, 0]))
        if self.cfg.SimType == SimulateType.Gas:
            self.t.fill(self.cfg.GasInitAmbientT)
