import taichi as ti
import taichi_glsl as ts

ti.init(ti.gpu, debug=False)


@ti.data_oriented
class f_wrapper():
    def __init__(self, f):
        self.field = f

    @ti.pyfunc
    def __getitem__(self, I):
        return self.field[I]

    @ti.pyfunc
    def __setitem__(self, I, value):
        pass
        # self.field[I] = value

    def loop_range(self):
        return self.field.loop_range()


@ti.kernel
def FillValue(f_w: ti.template()):
    for I in ti.grouped(f_w):
        f_w.field[I] = I


@ti.kernel
def PrintValue(f_w: ti.template()):
    for I in ti.grouped(f_w):
        print(f_w[I])


if __name__ == "__main__":
    a_field = ti.Vector.field(2, dtype=ti.f32, shape=[3, 3])
    field_wrapper = f_wrapper(a_field)
    FillValue(field_wrapper)
    PrintValue(field_wrapper)