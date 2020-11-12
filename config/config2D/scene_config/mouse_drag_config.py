from config.euler_config.basic_config import m_dx as dx
from config.euler_config.basic_config import m_dt as dt
from config.euler_config.basic_config import m_res as res
import taichi as ti
import numpy as np

m_fluid_color = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3))
m_f_strength = 10000.0
m_dye_decay = 0.99

m_force_radius = res[0] / 3.0
m_inv_force_radius = 1.0 / m_force_radius
m_inv_dye_denom = 4.0 / (res[0] / 15.0)**2
m_f_strength_dt = m_f_strength * dt