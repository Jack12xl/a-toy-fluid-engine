import taichi as ti
import taichi_glsl as ts

ti.init(ti.gpu, debug=False)

class py_list_wrapper:
    def __init__(self):
        self.l = [233, 233, 233]

    def __getitem__(self, I):
        print("call list getitem")
        return self.l[I]

    def __setitem__(self, key, value):
        print("call list setitem")
        self.l[key] = value

@ti.data_oriented
class f_wrapper():
    def __init__(self, f):
        self.field = f

    @ti.pyfunc
    def __getitem__(self, I):
        print("get")
        return ts.vec2(self.field[I][1], self.field[I][0]) # get work fine
        # return self.field[I]
        # return ts.vec2(233, 233) # would cause internal error with lvalue

    @ti.pyfunc
    def __setitem__(self, I, value):
        # pass
        print("set")
        self.field[I] = value

    def loop_range(self):
        return self.field.loop_range()


@ti.kernel
def FillValue(f_w: ti.template()):
    for I in ti.grouped(f_w):
        f_w[I] = I

@ti.kernel
def FillSlice(f_w: ti.template()):
    f_w[ti.grouped(ts.vec2(2,2))] = ts.vec2(2,2)

@ti.kernel
def PrintValue(f_w: ti.template()):
    for I in ti.grouped(f_w):
        print("print", f_w[I])


if __name__ == "__main__":
    a_field = ti.Vector.field(2, dtype=ti.f32, shape=[3, 3])
    field_wrapper = f_wrapper(a_field)
    FillValue(field_wrapper)
    # FillSlice(field_wrapper)
    PrintValue(field_wrapper)

    l_wrapper = py_list_wrapper()
    l_wrapper[1] = 23333

