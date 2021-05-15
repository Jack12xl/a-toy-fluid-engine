import taichi as ti
import taichi_glsl as ts

ti.init(ti.gpu)


@ti.kernel
def test_list():
    a = [1.0, -1.0]
    for i in ti.static(a):
        for j in ti.static(a):
            for k in ti.static(a):
                print(ts.vec(i, j, k))


@ti.kernel
def test_list2():
    a = [[1, 2], [2, 4], [3, 6]]
    b = [5, 6, 7]
    for i, j in ti.static(zip(a, b)):
        print(ti.Vector(i), j)


test_a = ti.Vector.field(2, dtype=ti.f32, shape=[16, 16])

@ti.kernel
def set_test():
    for I in ti.grouped(test_a):
        test_a[I] = I

@ti.kernel
def test_zip():
    # for II in (ti.static(ti.grouped(test_a), range(256))):
    #     print(233)
    for I in test_a:
        print(I)

if __name__ == "__main__":
    set_test()
    test_zip()
