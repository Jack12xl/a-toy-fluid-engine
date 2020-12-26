import taichi as ti
from config.CFG_wrapper import DLYmethod
import os

# Hello MPM
dim = 3
CFL = None

layout_method = DLYmethod.AoS

quality = 2

dt = 4e-3  # frame dt
max_n_particle = 27000 * quality ** 2

n_grid = 128 * quality
dx = 1.0 / n_grid
res = [n_grid, n_grid, n_grid]
screen_res = [512, 512]

p_vol = dx ** dim
p_rho = 1000

g_padding = [3, 3, 3]

E, nu = 1e6, 0.2

ti.init(arch=ti.gpu, debug=False, kernel_profiler=False, device_memory_GB=4.0)

profile_name = "MPM{}D-P-{}-G-{}-dt-{}".format(dim, max_n_particle, 'x'.join(map(str, res)), dt)
bool_save = True
save_frame_length = 192
save_root = './tmp_result'
save_path = os.path.join(save_root, profile_name)
