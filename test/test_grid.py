import taichi as ti
import taichi_glsl as ts


ti.init(ti.cpu)

if __name__ == '__main__':
    a_field = ti.Vector.field(2, dtype=ti.f32, shape=[3,3])
    a = ti.Vector([2, 3])
    print(ts.normalize(a))
    print(ts.D.xy)
    print(ts.D.yx)
    print(ts.D.xz)
    #print(a_field.shape)
    print(len(a))
    pass