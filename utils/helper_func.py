import numpy as np
import config.base_cfg as base_cfg
import taichi as ti

def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

@ti.func
def clamp(v, start, end):
    '''
    clamp in closed interval
    :param v:
    :param start:
    :param end:
    :return:
    '''
    return max(start, min(end, v))

@ti.func
def lerp(vl, vr, frac):
    return frac * ( vr - vl ) + vl




def npNormalize(a, order=2, axis=0):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis)) + base_cfg.error
    # l2[l2 == 0] = 1
    return a / l2

@ti.kernel
def copy_ti_field( dst: ti.template(),
                   trgt:ti.template()):
    for I in ti.grouped(dst):
        dst[I] = trgt[I]

@ti.kernel
def reflect(to_be_reflected:ti.template(),
            mid_point: ti.template()):
    for (I) in ti.grouped(to_be_reflected):
        to_be_reflected[I] = 2.0 * mid_point[I] - to_be_reflected[I]



if __name__ == '__main__':
    A = np.array([[0,0],[0,0]])
    print(A)
    print(npNormalize(A))
    # print(A)
    # print(npNormalize(A, 0))
    # print(npNormalize(A, 1))
    # print(npNormalize(A, 2))