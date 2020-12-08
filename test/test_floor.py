import taichi as ti
import taichi_glsl as ts

ti.init(ti.cpu, debug=True)

@ti.kernel
def test_floor():
    I = ts.vec(-0.4, -0.6, 0.0, 0.7, 1.5)
    f = ti.floor(I)
    print(f)

@ti.kernel
def test_int():
    I = ts.vec(-0.4, -0.6, 0.0, 0.7, 1.5)
    f = int(I)
    print(ts.fract(I))
    print(f)

if __name__ == "__main__":
    print("floor: ")
    test_floor()
    print("int: ")
    test_int()