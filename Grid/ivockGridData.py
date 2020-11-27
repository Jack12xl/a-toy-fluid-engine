import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix, MultiBufferPair
from .FaceGrid import FaceGrid, CellGrid
from config import SimulateType
from .staggeredGridData import MacGridData


@ti.data_oriented
class IVOCKGridData(MacGridData):
    """
    The grid storage designed specifically for IVOCK scheme
    velocity on face center
    vorticity(curl) on cell edges

    curl vel curl
    vel  -   vel
    curl vel curl

    :return:
    """

    def __init__(self, cfg):
        super(MacGridData, self).__init__(cfg)

        self.new_v_curl = None
        if self.dim == 2:
            self.new_v_curl = CellGrid(
                ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res),
                cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx),
                o=ts.vecND(self.dim, 0.0)
            )
        elif self.dim == 3:
            self.new_v_curl = CellGrid(
                ti.Vector.field(cfg.dim, dtype=ti.f32, shape=cfg.res),
                cfg.dim,
                dx=ts.vecND(self.dim, self.cfg.dx),
                o=ts.vecND(self.dim, 0.0)
            )

        self.curl_pair = bufferPair(self.v_curl, self.new_v_curl)

    @ti.kernel
    def calVorticity2D(self, vf: Matrix):
        for I in ti.static(vf):
            # y
            vl = vf.fields[1].sample(I)[0]
            vr = vf.fields[1].sample(I + ts.D.xy)[0]
            # x
            vb = vf.fields[0].sample(I)[0]
            vt = vf.fields[0].sample(I + ts.D.yx)[0]
            self.v_curl[I][0] = (vr - vl - vt + vb) * self.inv_d
