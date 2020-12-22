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
        self.Layout.init_cube()
        pass

    def substep(self, dt: Float):
        # self.print_property(34)
        self.Layout.G2zero()
        # self.print_property(35)
        self.Layout.P2G(dt)
        # self.print_property(37)
        self.Layout.G_Normalize_plus_Gravity(dt)
        # self.print_property(39)
        self.Layout.G_boundary_condition()
        # self.print_property(41)
        self.Layout.G2P(dt)
        # self.print_property(43)

    @ti.kernel
    def print_property(self, prefix: ti.template(), v:ti.template()):
        p_x = ti.static(self.Layout.p_x)
        for P in p_x:
            print(prefix, v[P])
            # assert(ts.isnan(p_x[P][0]))
            # assert (ts.isnan(p_x[P][1]))
            # print(p_x[P])

    def step(self, print_stat=False):
        """
        Call once in each frame
        :return:
        """
        # TODO change dt
        for _ in range(int(2e-3 // self.cfg.dt)):
            self.substep(self.cfg.dt)
        if print_stat:
            ti.kernel_profiler_print()
            try:
                ti.memory_profiler_print()
            except:
                pass
            print(f'num particles={self.Layout.n_particles[None]}')
        self.curFrame += 1
        # print("frame {}".format(self.curFrame))

    def reset(self):
        """
        restart the whole process
        :return:
        """
        self.materialize()

        self.curFrame = 0

    def add_cube(self):
        pass
