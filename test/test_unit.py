import taichi as ti

ti.init(ti.gpu)

dim = 2
@ti.kernel
def play_unit():
    for i in ti.static(range(dim)):
        D = ti.Vector.unit(dim, i)
        print(D)


if __name__ == '__main__':
    play_unit()
