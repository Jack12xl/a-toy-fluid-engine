import taichi as ti
import taichi_glsl as ts
# from Grid import FaceGrid
import numpy as np

ti.init(ti.gpu)

@ti.kernel
def getMaxVel(v: ti.template()) -> ti.f32:
    """
    Assume v is FaceGrid
    :param v:
    :return:
    """
    ret = 0.0
    for d in ti.static(range(2)):
        for I in ti.static(v.fields[d]):
            v_abs = ti.abs(v.fields[d][I][0])
            ret = ti.atomic_max(v_abs, ret)

    return ret

def getAnpmax(a):
    ret = 0.0
    for d in range(2):
        np_f = a.fields[d].field.to_numpy()
        cur_max = np.max(np_f)
        ret = max(cur_max, ret)

    return ret


@ti.kernel
def set(a:ti.template()):
    a.fields[1][ts.vec2(1)][0] = 2.33

@ti.kernel
def getBmax(b: ti.template()) -> ti.f32:
    ret = 0.0
    for I in ti.grouped(b):
        ret = ti.atomic_max(b[I][0], ret)

    return ret


def getBnumpymax(b):
    b_np = b.to_numpy()
    print(b_np.shape)
    return np.max(b_np)


@ti.kernel
def set_b(b:ti.template()):
    b[233, 233][0] = 2.0

if __name__ == "__main__":
    # a = FaceGrid(ti.f32,
    #              shape=[512, 512],
    #              dim=2,
    #              dx=ts.vecND(2, 0.1),
    #              o=ts.vecND(2, 0.5)
    #              )
    b = ti.Vector.field(1, dtype=ti.f32, shape=[512, 512])

    # set(a)
    # max_a = getMaxVel(a)
    # print(max_a)
    #
    # print(getAnpmax(a))

    set_b(b)
    b_max = getBmax(b)
    print(b_max)

    print(getBnumpymax(b))

