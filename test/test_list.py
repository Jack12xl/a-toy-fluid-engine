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
    for i in ti.static(a):
        print(i)


if __name__ == "__main__":
    test_list2()
