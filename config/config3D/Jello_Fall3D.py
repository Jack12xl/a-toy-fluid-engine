import taichi as ti
from config.CFG_wrapper import DLYmethod

# Hello MPM
dim = 3
CFL = None

layout_method = DLYmethod.AoS

quality = 1

dt = 1e-4 / quality
n_particle = 18000 * quality ** 2

n_grid = 128 * quality
dx = 1.0 / n_grid
res = [n_grid, n_grid, n_grid]
screen_res = [512, 512]

p_vol = dx ** dim
p_rho = 1000

g_padding = [3, 3, 3]

E, nu = 1e6, 0.2

ti.init(arch=ti.gpu, debug=False, kernel_profiler=True)

profile_name = "MPM{}D-P-{}-G-{}-dt-{}".format(dim, n_particle, 'x'.join(map(str, res)), dt)
bool_save = False
