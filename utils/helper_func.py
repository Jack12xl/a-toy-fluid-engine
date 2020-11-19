import numpy as np
# import config.base_cfg as base_cfg
import taichi as ti
from .basic_types import Vector
import re
import math


class SetterProperty(object):
    """
    use this to reduce @property overhead
    hope it works
    """

    # ref: https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)


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
    raise DeprecationWarning
    return max(start, min(end, v))


@ti.func
def lerp(vl, vr, frac):
    raise DeprecationWarning
    return frac * (vr - vl) + vl


def npNormalize(a, order=2, axis=0):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis)) + 0.0001
    # l2[l2 == 0] = 1
    return a / l2


@ti.func
def tiNormalize(v: Vector) -> Vector:
    return v / (v.norm() + base_cfg.error)


@ti.func
def EuclideanDistance(v1: Vector, v2: Vector):
    return (v1 - v2).norm()


@ti.kernel
def copy_ti_field(dst: ti.template(),
                  trgt: ti.template()):
    for I in ti.grouped(dst.field):
        dst[I] = trgt[I]


@ti.kernel
def reflect(to_be_reflected: ti.template(),
            mid_point: ti.template()):
    for (I) in ti.grouped(to_be_reflected.field):
        to_be_reflected[I] = 2.0 * mid_point[I] - to_be_reflected[I]


def filterUpCase(c: str) -> str:
    return re.sub('[^A-Z]', '', c)


def getFieldMeanCpu(f: ti.field):
    return np.mean(f.to_numpy())


@ti.kernel
def test():
    a = ti.Vector([2.0, 2.0])
    b = ti.Vector([1.0, 1.0])
    print(EuclideanDistance(a, b))


if __name__ == '__main__':
    # A = np.array([[0,0],[0,0]])
    # print(A)
    # print(npNormalize(A))
    # print(A)
    # print(npNormalize(A, 0))
    # print(npNormalize(A, 1))
    # print(npNormalize(A, 2))
    test()
