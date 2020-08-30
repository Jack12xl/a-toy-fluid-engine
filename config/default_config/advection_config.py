from advection import SemiLagrangeOrder, SemiLagrangeSolver, MacCormackSolver

# Advection
m_semi_order = SemiLagrangeOrder.RK_2
m_advection_solver = MacCormackSolver
m_macCormack_clipping = True