from utils import ti
import taichi_glsl as ts

ti.init(ti.gpu)

a = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def test_approx(_in: ti.template()):
    assert ((233, 233) == (233, 233))
    print(_in[None])
    # print(t(a))


if __name__ == "__main__":
    test_approx(a)
