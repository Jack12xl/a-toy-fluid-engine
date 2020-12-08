import taichi as ti
import taichi_glsl as ts

ti.init(ti.gpu)

q = ti.Vector.field(2, dtype=ti.f32, shape=(4, 4))
q_2 = ti.Vector.field(1, dtype=ti.f32, shape=(4, 4))



@ti.kernel
def test(
        f1: ti.template(),
        f2: ti.template()
):
    for I in ti.grouped(f1):
        f1[I] = f2[I]

test_func = test

if __name__ == "__main__":
    test_func(q, q_2)