import taichi as ti
import taichi_glsl as ts
ti.init(ti.gpu)

@ti.kernel
def test_while():
    iter = 0
    while iter < 6:
        iter += 1
        for i in range(iter):
            print(i)
        print(iter)


if __name__ == "__main__":
    test_while()