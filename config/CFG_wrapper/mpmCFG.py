from .fluidCFG import FluidCFG
import taichi as ti
from utils import SetterProperty, plyWriter

class mpmCFG(FluidCFG):
    """
    Property for mpm-based simulation
    """

    def __init__(self, cfg):
        """

        :param cfg: A module,
                    contains the config for each each simulated scene
        """
        super(mpmCFG, self).__init__(cfg)

        self.n_particle = None
        self.n_grid = None
        self.quality = cfg.quality

        self.p_vol = cfg.p_vol
        self.p_rho = cfg.p_rho

        self.p_mass = self.p_vol * self.p_rho

        # Lame(not Lame... Well how to type that)
        self.mu_0 = None
        self.lambda_0 = None
        self._E = 1.0
        self._nu = 1.0
        self.E = cfg.E  # Young's modules
        self.nu = cfg.nu  # Poisson's ratio

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, q: int):
        self._quality = q
        self.n_particle = 9000 * q ** 2
        self.n_grid = 128 * q
        self.dt = 1e-4 / q


    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, _E: float):
        self._E = _E
        self.mu_0 = _E / (2 * (1 + self.nu))
        self.lambda_0 = _E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, _nu: float):
        self._nu = _nu
        self.mu_0 = self.E / (2 * (1 + _nu))
        self.lambda_0 = self.E * _nu / ((1 + _nu) * (1 - 2 * _nu))





