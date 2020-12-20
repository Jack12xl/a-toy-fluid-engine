import taichi as ti
import taichi_glsl as ts
import numpy as np
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import mpmCFG
from utils import Vector, Float
from DataLayout.MPM import mpmLayout

"""
ref : 
    taichi elements
    mpm2d.py
"""


@ti.data_oriented
class mpmScheme(metaclass=ABCMeta):
    def __init__(self, cfg: mpmCFG):
        self.cfg = cfg

        self.dim = cfg.dim

        self.Layout = mpmLayout(cfg)
        self.curFrame = 0
        pass

    def materialize(self):
        # initial the ti.field
        self.Layout.materialize()
        pass

    def substep(self, dt: Float):
        self.Layout.G2zero()
        self.Layout.P2G(dt)
        self.Layout.G_Normalize_plus_Gravity(dt)
        self.Layout.G_boundary_condition()
        self.Layout.G2P(dt)

    def step(self):
        """
        Call once in each frame
        :return:
        """
        for _ in range(int(2e-3 // self.cfg.dt)):
            self.substep(self.cfg.dt)

        self.curFrame += 1
        print("frame {}".format(self.curFrame))

    def reset(self):
        """
        restart the whole process
        :return:
        """
        self.materialize()
        self.curFrame = 0
