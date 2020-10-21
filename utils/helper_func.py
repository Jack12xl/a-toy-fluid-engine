import numpy as np
import config.base_cfg as base_cfg
import taichi as ti
from .basic_types import Vector

import math

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

@ti.func
def tiNormalize(v: Vector) -> Vector:
    return v / (v.norm() + base_cfg.error)

@ti.func
def EuclideanDistance(v1: Vector, v2: Vector):
    return (v1 - v2).norm()

@ti.kernel
def copy_ti_field( dst: ti.template(),
                   trgt:ti.template()):
    for I in ti.grouped(dst.field):
        dst[I] = trgt[I]

@ti.kernel
def reflect(to_be_reflected:ti.template(),
            mid_point: ti.template()):
    for (I) in ti.grouped(to_be_reflected.field):
        to_be_reflected[I] = 2.0 * mid_point[I] - to_be_reflected[I]


@ti.kernel
def test():
    a = ti.Vector([2.0, 2.0])
    b = ti.Vector([1.0, 1.0])
    print(EuclideanDistance(a,b))

if __name__ == '__main__':
    # A = np.array([[0,0],[0,0]])
    # print(A)
    # print(npNormalize(A))
    # print(A)
    # print(npNormalize(A, 0))
    # print(npNormalize(A, 1))
    # print(npNormalize(A, 2))
    test()