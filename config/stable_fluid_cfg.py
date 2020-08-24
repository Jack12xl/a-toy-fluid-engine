import taichi as ti
from .class_cfg import SceneEnum, VisualizeEnum
from advection import SemiLagrangeOrder, SemiLagrangeSolver, MacCormackSolver

dim = 2
res = [600, 600]
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
dt = 0.03
p_jacobi_iters = 30
f_strength = 10000.0
dye_decay = 0.99
debug = True

force_radius = res[0] / 3.0
inv_force_radius = 1.0 / force_radius
inv_dye_denom = 4.0 / (res[0] / 15.0)**2
f_strength_dt = f_strength * dt

poisson_pressure_alpha = ti.static(- dx * dx)
poisson_pressure_beta  = ti.static(0.25)

dynamic_viscosity_coefficient = 0.1
poisson_viscosity_alpha = ti.static(dx * dx ) / (dt * dynamic_viscosity_coefficient )
poisson_viscosity_beta = 1.0 / (poisson_viscosity_alpha + 4)

jacobi_alpha = poisson_pressure_alpha
jacobi_beta = poisson_pressure_beta

SceneType = SceneEnum.MouseDragDye
VisualType = VisualizeEnum.Dye

# Advection
semi_order = SemiLagrangeOrder.RK_3
advection_solver = MacCormackSolver
macCormack_clipping = True

# scheme_setting = dict(
#     dim = dim,
#     res = res,
#     dx = dx,
#     inv_dx = inv_dx,
#     half_inv_dx = half_inv_dx,
#
#     dt = dt,
#     p_jacobi_iters = p_jacobi_iters,
#     f_strength = f_strength,
#     dye_decay = dye_decay,
#     debug = debug,
#
#     force_radius = force_radius,
#     inv_force_radius = inv_force_radius,
#     inv_dye_denom = inv_dye_denom,
#     f_strength_dt = f_strength * dt
#
# )
