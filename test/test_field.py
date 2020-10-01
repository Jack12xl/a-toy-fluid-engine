import taichi as ti
import math

ti.init(ti.cpu, debug=True)

t1 = ti.field(dtype=ti.f32, shape=2)
t2 = ti.Vector.field(2, dtype=ti.f32, shape=[1])

t1.fill(2.0)
t2.fill(ti.Vector([1.0, 1.0]))

@ti.kernel
def kern_init():
    t1[0] = 3.0
    t1[1] = 4.0

@ti.kernel
def kern_field():
    # t1[0] = 3
    # t1[1] = 4
    t2 *= 2.0
    # print(t2)

kern_init()

while True:


    kern_field()

    # print(t1)
    print(t2)