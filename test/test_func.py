import taichi as ti


@ti.kernel
def play():
    b = 3
    plus(b)
    print(b)

def plus(a):
    return a + 1


if __name__ == "__main__":
    ti.init(ti.gpu)
    play()
