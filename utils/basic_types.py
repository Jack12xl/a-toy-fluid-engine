# ref: https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi/blob/master/src/python/basic_types.py

import taichi as ti

Float = ti.f32
Int = ti.i32
# Bool = ti.i32   # 0 is False , else is True

Vector = ti.template()
Matrix = ti.template()

# wrapper, for example, defined in Grid/cellgrid,
Wrapper = ti.template()

Index = Vector

# ScalarGrid = Grid
# VectorGrid = Grid

INT_TYPEs = [ti.i8, ti.i16, ti.i32, ti.i64, ti.u16, ti.u8, ti.u32, ti.u64]
FLAOT_TYPEs = [ti.f32, ti.f64]


def in_int_type(dtype):
    return dtype in INT_TYPEs


def in_float_type(dtype):
    return dtype in FLAOT_TYPEs


def in_scalar_type(dtype):
    return in_int_type(dtype) or in_float_type(dtype)


def is_int(val):
    # return len(list(filter(lambda dtype : isinstance(val , dtype) , INT_TYPEs))) > 0
    return isinstance(val, int)


def is_float(val):
    # return len(list(filter(lambda dtype: isinstance(val , dtype) , FLAOT_TYPEs))) > 0
    return isinstance(val, float)


def is_bool(val):
    return isinstance(val, bool)


def is_scalar(val):
    return is_float(val) or is_int(val) or is_bool(val)
