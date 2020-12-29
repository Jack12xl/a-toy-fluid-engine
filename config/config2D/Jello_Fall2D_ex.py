import taichi as ti
import os
from config.CFG_wrapper import DLYmethod

# Hello MPM
dim = 2
CFL = None

layout_method = DLYmethod.AoS_Dynamic

quality = 1

max_n_particle = 2 ** 21
p_chunk_size = 2 ** 19
# dt = 1e-4 / quality
dt = 4e-3

n_grid = 64 * quality
dx = 1.0 / n_grid
res = [n_grid, n_grid]
screen_res = [512, 512]

p_vol = (dx * 0.5) ** 2  # 4 in each cell
p_rho = 1

g_padding = [3, 3]

E, nu = 1e3, 0.2

ti.init(arch=ti.gpu, debug=False, kernel_profiler=True)

profile_name = "MPM{}D-G-{}-dt-{}".format(dim, 'x'.join(map(str, res)), dt)
bool_save = False
save_frame_length = 192
save_root = './tmp_result'
save_path = os.path.join(save_root, profile_name)
