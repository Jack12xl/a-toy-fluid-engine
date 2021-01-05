import taichi as ti
import taichi_glsl as ts
from Grid import FaceGrid
import numpy as np
from utils import bufferPair

ti.init(ti.gpu, debug=False)


@ti.kernel
def FillV(_v: ti.template()):
    """
    In this way, faceGrid would not be "Filled"
    :param _v:
    :return:
    """
    for I in ti.static(_v):
        _v[I] = ts.vec(233, 233)


@ti.kernel
def SupposedWay2FillV(_v: ti.template()):
    for d in ti.static(range(_v.dim)):
        for I in ti.static(_v.fields[d]):
            _v.fields[d][I] = ts.vec(I[d])


@ti.kernel
def SupposedWay2FillV_new(_v: ti.template()):
    for d in ti.static(range(_v.dim)):
        for I in ti.static(_v.fields[d]):
            _v.fields[d][I] = 2 * ts.vec(I[d])


@ti.kernel
def PrintV(_v: ti.template()):
    for I in ti.static(_v):
        print(_v[I])


if __name__ == "__main__":
    v = FaceGrid(ti.f32,
                 shape=[2, 2],
                 dim=2,
                 dx=ts.vecND(2, 1.0),
                 o=ts.vecND(2, 0.5)
                 )

    v_new = FaceGrid(ti.f32,
                     shape=[2, 2],
                     dim=2,
                     dx=ts.vecND(2, 1.0),
                     o=ts.vecND(2, 0.5)
                     )

    SupposedWay2FillV(v)
    SupposedWay2FillV_new(v)

    v_pair = bufferPair(v, v_new)

    advect_v_pairs = []
    for d in range(2):
        advect_v_pairs.append(
            bufferPair(v.fields[d], v_new.fields[d])
        )

    for d in range(2):
        advect_v_pairs[d].swap()

    PrintV(v_pair.cur)

    print("next")

    print(advect_v_pairs)
