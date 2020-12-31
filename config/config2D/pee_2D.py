import taichi as ti
import os
from config.CFG_wrapper import DLYmethod, BC

# Hello MPM
dim = 2
CFL = None

layout_method = DLYmethod.AoS_TwinGrid

quality = 2

max_n_particle = 18000 * quality ** 2
# dt = 1e-4 / quality
dt = 4e-3


n_grid = 128 * quality
substep_dt = 1e-2 / n_grid

dx = 1.0 / n_grid
res = [n_grid, n_grid]
screen_res = [512, 512]

p_vol = (dx * 0.5) ** 2
p_rho = 400

g_padding = [3, 3]

E, nu = 3.537e5, 0.3

bdryCdtn = BC.slip

ti.init(arch=ti.gpu, debug=False, kernel_profiler=True)

from datetime import datetime
t = str(datetime.now())[5:-7].replace(' ', '-').replace(':', "-")
profile_name = t + "-pee-MPM{}D-P-{}-G-{}-dt-{}".format(dim, max_n_particle, 'x'.join(map(str, res)), dt)
bool_save = False
save_frame_length = 192
save_root = './tmp_result'
save_path = os.path.join(save_root, profile_name, 'simple')

bool_save_particle = False
particle_step = 1
particle_path = os.path.join(save_root, profile_name, 'particle')
