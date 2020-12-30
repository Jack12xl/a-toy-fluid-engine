import taichi as ti
import taichi_glsl as ts
import numpy as np
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import mpmCFG, MaType, DLYmethod
from utils import Vector, Float
from DataLayout.MPM import mpmLayout, mpmDynamicLayout
import multiprocessing as mp

"""
ref : 
    taichi elements
    mpm2d.py
"""


@ti.data_oriented
class MPMSolver(metaclass=ABCMeta):
    def __init__(self, cfg: mpmCFG):
        self.cfg = cfg

        self.dim = cfg.dim

        if self.cfg.layout_method < int(DLYmethod.AoS_Dynamic):
            self.Layout = mpmLayout(cfg)
        else:
            self.Layout = mpmDynamicLayout(cfg)
        self.curFrame = 0
        self.writers = []

    def materialize(self):
        # initial the ti.field
        self.Layout.materialize()
        # self.Layout.init_cube()

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
    def print_property(self, prefix: ti.template(), v: ti.template()):
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
        T = 0.0
        sub_dt = self.cfg.substep_dt

        # #this way is not friendly for Taichi compilation(takes too long)
        # while T < self.cfg.dt:
        #     if T + sub_dt > self.cfg.dt:
        #         sub_dt = self.cfg.dt - T
        #
        #     self.substep(sub_dt)

        # print(self.cfg.dt // sub_dt)
        n_substep = int(self.cfg.dt // sub_dt) + 1
        sub_dt = self.cfg.dt / n_substep
        for _ in range(n_substep):
            self.substep(sub_dt)

        # if print_stat:
        #     ti.kernel_profiler_print()
        #     try:
        #         ti.memory_profiler_print()
        #     except:
        #         pass
        #     print(f'num particles={self.Layout.n_max_particle[None]}')
        self.curFrame += 1
        # print("frame {}".format(self.curFrame))

    def reset(self):
        """
        restart the whole process
        :return:
        """
        # TODO
        # self.Layout.init_cube()
        self.Layout.n_particle[None] = 0
        self.curFrame = 0

    def write_particles(self, fn: str):
        ps = self.Layout.particle_info()

        # p = mp.Process(target=self.Layout.dump, args=(fn, ps))
        # p.start()
        #
        # self.writers.append(p)
        self.Layout.dump(fn, ps)
