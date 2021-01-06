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


if __name__ == "__main__":
    test_list2()
