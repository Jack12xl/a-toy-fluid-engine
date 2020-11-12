import taichi as ti
import taichi_glsl as ts
import numpy as np

m_fluid_color = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3))
m_dye_decay = 0.99

m_f_gravity = ts.vec3(0.0, -9.8, 0.0)