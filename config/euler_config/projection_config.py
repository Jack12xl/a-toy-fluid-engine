import taichi as ti
from projection import JacobiProjectionSolver
# from basic_config2D import m_dx as dx
from config.config2D.basic_config2D import m_dx as dx

m_projection_solver = JacobiProjectionSolver
m_p_jacobi_iters = 30

m_poisson_pressure_alpha = ti.static(- dx * dx)
# dim == 2 should be 1.0/4.0
# dim == 3 1.0 / 6.0
# damn this takes me almost three days
m_poisson_pressure_beta = ti.static(1.0 / 4.0)

m_dynamic_viscosity_coefficient = 500
# m_poisson_viscosity_alpha = ti.static(dx * dx) / (dt * m_dynamic_viscosity_coefficient )
# m_poisson_viscosity_beta = 1.0 / (m_poisson_viscosity_alpha + 4)
m_poisson_viscosity_alpha = m_poisson_pressure_alpha
m_poisson_viscosity_beta = m_poisson_pressure_beta

m_jacobi_alpha = m_poisson_pressure_alpha
m_jacobi_beta = m_poisson_pressure_beta