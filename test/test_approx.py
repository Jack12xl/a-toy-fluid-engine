import taichi as ti
import taichi_glsl as ts

ti.init(ti.gpu)


@ti.func
def t(a):
    return a == ti.approx(ts.vec3(233.0))


@ti.kernel
def test_approx():
    assert((233, 233) == (233, 233))
    a = ts.vec3(233.0)
    # print(t(a))


if __name__ == "__main__":
    test_approx()
