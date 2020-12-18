import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix, MultiBufferPair, Wrapper
from FaceGrid import FaceGrid, CellGrid
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
        super(IVOCKGridData, self).__init__(cfg)

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
        # IVOCK would also advect the curl(vorticity)
        self.curl_pair = bufferPair(self.v_curl, self.new_v_curl)
        self.calCurl = self.calCurl2D if self.dim == 2 else self.dim == 3

    @ti.kernel
    def calCurl2D(self, vf: Wrapper, vc: Wrapper):
        """

        :param vf: velocity field
        :param vc: velocity curl
        :return:
        """
        for I in ti.static(vf):
            # y
            vl = vf.fields[1].sample(I)[0]
            vr = vf.fields[1].sample(I + ts.D.xy)[0]
            # x
            vb = vf.fields[0].sample(I)[0]
            vt = vf.fields[0].sample(I + ts.D.yx)[0]
            vc[I][0] = (vr - vl - vt + vb) * self.inv_d

    @ti.kernel
    def calCurl3D(self, vf: Wrapper, vc: Wrapper):
        """

        :param vf:
        :param vc:
        :return:
        """
        pass