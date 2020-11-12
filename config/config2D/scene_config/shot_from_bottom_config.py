import taichi as ti
import numpy as np

m_fluid_color = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3))
# m_f_strength = 10000.0
m_dye_decay = 0.99

m_f_gravity = ti.Vector([0.0, -9.8])