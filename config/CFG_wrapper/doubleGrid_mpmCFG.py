from .fluidCFG import FluidCFG
from enum import IntEnum
import math
from utils import SetterProperty
import taichi as ti
import os
from .mpmCFG import DLYmethod, MaType, BC, mpmCFG


class TwoGridmpmCFG(mpmCFG):
    """
    Property for two grid MPM
    """

    def __init__(self, cfg):
        super(TwoGridmpmCFG, self).__init__(cfg)
        # while max_n_particle stands for sand
        self.max_n_w_particle = cfg.max_n_w_particle if hasattr(cfg, "max_n_w_particle") else self.max_n_particle

        # w_K: bulk modules of water
        # gamma: more stiffness penalizes incompressibility
        self.w_k, self.w_gamma = 50, 3
        # n: sand porosity k_hat: permeability
        self.n, self.k_hat = 0.4, 0.2

        # coefficient of friction
        self.mu_b = 0.75

        self.a, self.b, self.c0, self.sC = -3.0, 0, 1e-2, 0.15

        # Hardening parameters @16 7.3
        self.h0, self.h1, self.h2, self.h3 = 35, 9, 0.2, 10
