import taichi as ti
import taichi_glsl as ts
from geometry import Transform3
import math

ti.init(ti.gpu, debug=False)

if __name__ == "__main__":
    t1 = Transform3(
        translation=ts.vec3(2, 3, 3),
        localscale=ts.vec3(8.0, 8.0, 4.0),
        orientation=ts.vec2(math.pi / 2.0, math.pi / 2.0)  # Up along Y axis
    )

    t2 = Transform3(
        translation=ts.vec3(2, 3, 3),
        localscale=ts.vec3(8.0, 8.0, 4.0),
        orientation=ts.vec2(math.pi / 2.0, math.pi / 2.0)  # Up along Y axis
    )

    print(t1._translation[0])