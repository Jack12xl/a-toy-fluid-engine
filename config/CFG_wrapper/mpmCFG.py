from .fluidCFG import FluidCFG
from enum import IntEnum
import math
from utils import SetterProperty, plyWriter
import taichi as ti
import os


class DLYmethod(IntEnum):
    """
    Data layout method
    The way we set the memory structure of particle, grid ....
    """
    SoA = 0
    AoS_0 = 1  # support metal
    AoS_1 = 2  # inspired by Taichi Elements
    AoS_Dynamic = 3  # copied from Taichi Elements....


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
        # self.n_particle = None
        # max 128 MB particles
        self.max_n_particle = cfg.max_n_particle if hasattr(cfg, 'max_n_particle') else 2 ** 27
        self.p_chunk_size = cfg.p_chunk_size if hasattr(cfg, 'p_chunk_size') else 2 ** 19

        self.n_grid = None
        self.quality = cfg.quality

        self.substep_dt = cfg.substep_dt if hasattr(cfg, "substep_dt") else 1e-2 / self.n_grid

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
        # Sand parameters
        friction_angle = math.radians(45)
        sin_phi = math.sin(friction_angle)
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, q: int):
        self._quality = q
        # self.n_particle = self.cfg.n_particle if hasattr(self.cfg, 'n_particle') else 9000 * q ** 2
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

    @SetterProperty
    def bool_save(self, save):
        self.__dict__['bool_save'] = save
        print(">>>>>>>>>>")
        if save:
            # self.save_what = self.cfg.save_what
            self.save_frame_length = self.cfg.save_frame_length
            self.save_path = self.cfg.save_path
            print("Here we will simply save the particle ")
            # for save_thing in self.save_what:
            #     self.video_managers.append(ti.VideoManager(
            #         output_dir=os.path.join(self.cfg.save_path, str(save_thing)),
            #         framerate=self.cfg.frame_rate,
            #         automatic_build=False
            #     )
            #     )
            #     print(str(save_thing), end=" ")
            print("")
            # print("for {} frame with {} Frame Per Second".format(self.save_frame_length, self.cfg.frame_rate))
            print("We will save {} frame".format(self.save_frame_length))
            print("When done, plz go to {} for results !".format(self.cfg.save_path))

            # self.bool_save_ply = self.cfg.bool_save_ply
        else:
            print("Won't save results to disk this time !")
        print(">>>>>>>>>>")

    # @SetterProperty
    # def bool_save_ply(self, save):
    #     self.__dict__['bool_save_ply'] = save
    #     print(">>>>>>")
    #     if save:
    #         print("We will save ply every {} !".format(self.ply_frequency))
    #         self.PLYwriter = plyWriter(self)
    #         self.ply_frequency = self.cfg.ply_frequency
    #         print("When done, plz refer to {}".format(self.PLYwriter.series_prefix))
    #     print(">>>>>>")
