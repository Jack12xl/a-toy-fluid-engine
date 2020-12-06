import taichi as ti
import taichi_glsl as ts

ti.init(ti.gpu)

tmp_v = ti.Vector.field(2, dtype=ti.f32, shape=(4, 4))
delta_v = ti.Vector.field(2, dtype=ti.f32, shape=(4, 4))


def track_delta(d_v, src_v, func):
    # func(args*)
    #
    # return inner
    pass


def decorator_track(d_v, src_v):
    def Inner(func):
        def wrapper(*args):
            copy(d_v, src_v)
            func(*args)
            subtract(d_v, src_v)

        return wrapper

    return Inner


@ti.kernel
def copy(dst: ti.template(), src: ti.template()):
    for I in ti.grouped(dst):
        dst[I] = src[I]


@ti.kernel
def subtract(dst: ti.template(), src: ti.template()):
    for I in ti.grouped(dst):
        dst[I] = src[I] - dst[I]


@ti.kernel
def print_field(src: ti.template()):
    for I in ti.grouped(src):
        print(src[I])


@decorator_track(delta_v, tmp_v)
@ti.kernel
def add2(dst: ti.template()):
    for I in ti.grouped(dst):
        dst[I] += ts.vec2(2.0, 2.0)


if __name__ == "__main__":
    # decorator_track(delta_v, src_v=tmp_v)(add2)(tmp_v)
    add2(tmp_v)
    print_field(delta_v)
