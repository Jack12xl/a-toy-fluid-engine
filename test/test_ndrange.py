import taichi as ti
import taichi_glsl as ts


@ti.kernel
def fill(a: ti.template()):
    for I in ti.grouped(a):
        a[I] = I


@ti.kernel
def test_ndrange(a: ti.template()):
    shape = ts.vec3(a.shape)
    # print(shape.yz)
    for I in ti.grouped(ti.ndrange((64, 65), shape.y, shape.z)):
        print(a[I])


@ti.kernel
def test_ndrange2():
    l_b = ts.vec3(2, 2, 2)
    r_u = ts.vec3(4, 4, 4)
    r = [(l_b[i], r_u[i]) for i in range(len(l_b))]
    print(r)
    for I in ti.grouped(ti.ndrange(*r)):
        print(I)


@ti.kernel
def test_int(f: ti.template(), d: ti.template()):
    # f is a ti.vector.field
    # dim = ti.static(f.n) could work
    # a = ts.vecND(dim, 2)
    # dim = ti.static(d)
    a = ts.vecND(d, 2)  # would trigger errror
    print(a)


@ti.func
def stencil_range(l_b, r_u):
    return ti.ndrange(*[[l_b[d], r_u[d]] for d in range(3)])


@ti.kernel
def test():
    l_b = ts.vecND(3, 0)
    r_u = ts.vecND(3, 3)
    for offset in ti.grouped(stencil_range(l_b, r_u)):
        print(offset)


if __name__ == '__main__':
    ti.init(ti.gpu, debug=True)
    test()
    # a = ti.Vector.field(3, ti.float32, (128, 128, 128))
    # b = ti.Vector.field(2, ti.float32, (128, 128, 128))
    # # print(a.shape)
    # # fill(a)
    #
    # l_b = ts.vec3(2, 2, 2)
    # u_r = ts.vec3(4, 4, 4)
    #
    # # test_ndrange2()
    # test_int(a, 3)
    # test_int(b, 2)
    #
    # # test_ndrange(a)
