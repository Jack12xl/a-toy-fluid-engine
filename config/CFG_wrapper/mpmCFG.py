from .fluidCFG import FluidCFG
from enum import IntEnum


class DLYmethod(IntEnum):
    """
    Data layout method
    The way we set the memory structure of particle, grid ....
    """
    SoA = 0
    AoS = 1  # inspired by Taichi Elements


class MaType(IntEnum):
    """
    The material to support
    """
    elastic = 0
    liquid = 1
    snow = 2
    sand = 3


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
        # max 128 MB particles
        self.max_n_particle = cfg.max_n_particle if hasattr(cfg, 'max_n_particle') else 2 ** 27
        self.p_chunk_size = cfg.p_chunk_size if hasattr(cfg, 'p_chunk_size') else 2 ** 20

        self.n_grid = None
        self.quality = cfg.quality

        self.p_vol = cfg.p_vol
        self.p_rho = cfg.p_rho

        self.p_mass = self.p_vol * self.p_rho

        self.g_padding = cfg.g_padding

        self.layout_method = cfg.layout_method

        # Lame(not Lame... Well how to type that)
        self.mu_0 = None
        self.lambda_0 = None
        self._E = 1.0
        self._nu = 1.0
        self.E = cfg.E  # Young's modules
        self.nu = cfg.nu  # Poisson's ratio

        self.elastic_h = cfg.elastic_h if hasattr(cfg, "elastic_h") else 0.3


    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, q: int):
        self._quality = q
        self.n_particle = self.cfg.n_particle if hasattr(self.cfg, 'n_particle') else 9000 * q ** 2
        self.n_grid = self.cfg.n_grid if hasattr(self.cfg, 'n_grid') else 128 * q
        self.dt = self.cfg.dt if hasattr(self.cfg, 'dt') else 1e-4 / q

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
