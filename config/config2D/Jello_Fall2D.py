# Hello MPM
dim = 2

quality = 1
dx = 1.0 / 9000.0
dt = 1e-4 / quality

p_vol = (dx * 0.5) ** 2
p_rho = 1

E, nu = 1e3, 0.2

profile_name = "MPM{}D-Q-{}-dt-{}".format(dim, quality, dt)
bool_save = False