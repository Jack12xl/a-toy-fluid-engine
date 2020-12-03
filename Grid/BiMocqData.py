import taichi as ti
import taichi_glsl as ts
from utils import bufferPair, Vector, Matrix, MultiBufferPair, Wrapper
from .FaceGrid import FaceGrid, CellGrid
from config import SimulateType
from .staggeredGridData import MacGridData


@ti.data_oriented
class BimMocqGridData(MacGridData):
    def __init__(self, cfg):
        super(BimMocqGridData, self).__init__(cfg)

        # 3.3 forward map
