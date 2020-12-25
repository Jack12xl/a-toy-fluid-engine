import taichi as ti
from config.CFG_wrapper import DLYmethod

# Hello MPM
dim = 2
CFL = None

layout_method = DLYmethod.AoS

quality = 2

max_n_particle = 9000 * quality ** 2
# dt = 1e-4 / quality
dt = 4e-3

n_grid = 64 * quality
dx = 1.0 / n_grid
res = [n_grid, n_grid]
screen_res = [512, 512]

p_vol = (dx * 0.5) ** 2
p_rho = 1

g_padding = [3, 3]

E, nu = 1e3, 0.2

ti.init(arch=ti.gpu, debug=False, kernel_profiler=True)

profile_name = "MPM{}D-Q-{}-dt-{}".format(dim, quality, dt)
bool_save = False
