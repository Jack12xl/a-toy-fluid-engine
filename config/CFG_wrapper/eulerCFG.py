from .fluidCFG import FluidCFG
from utils import SetterProperty
from projection import RedBlackGaussSedialProjectionSolver, JacobiProjectionSolver, ConjugateGradientProjectionSolver
import taichi as ti


class EulerCFG(FluidCFG):
    """
    Hold property especially for Euler-based simulation
    """

    def __init__(self, cfg):
        super(EulerCFG, self).__init__(cfg)

        self.run_scheme = cfg.run_scheme
        self.VisualType = cfg.VisualType
        self.SceneType = cfg.SceneType

        self.advection_solver = cfg.advection_solver
        self.projection_solver = cfg.projection_solver
        self.viscosity_coefficient = cfg.viscosity_coefficient

        self.dye_decay = cfg.dye_decay
        self.curl_strength = cfg.curl_strength


    @property
    def advection_solver(self):
        return self._advection_solver

    @advection_solver.setter
    def advection_solver(self, solver):
        self._advection_solver = solver
        self.semi_order = self.cfg.semi_order

    @property
    def projection_solver(self):
        return self._projection_solver

    @projection_solver.setter
    def projection_solver(self, solver):
        self._projection_solver = solver
        if self._projection_solver != ConjugateGradientProjectionSolver:

            self.p_jacobi_iters = self.cfg.p_jacobi_iters
            self.poisson_pressure_alpha = ti.static(- self.dx * self.dx)
            # 1 / 4 for 2D, 1 / 6 for 3D
            self.poisson_pressure_beta = ti.static(1.0 / 2 * self.dim)

            # viscosity
            self.viscosity_coefficient = self.cfg.dynamic_viscosity_coefficient

        else:
            # TODO MGPCG
            pass

    @property
    def viscosity_coefficient(self):
        return self._viscosity_coefficient

    @viscosity_coefficient.setter
    def viscosity_coefficient(self, v):
        # GPU GEM
        self._viscosity_coefficient = v
        self.poisson_viscosity_alpha = self.dx * self.dx / (self.dt, self._viscosity_coefficient)
        self.poisson_viscosity_beta = 1.0 / (self.poisson_viscosity_alpha + 4)
