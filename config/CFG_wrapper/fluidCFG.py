import taichi as ti
from abc import ABCMeta, abstractmethod

from utils import SetterProperty


class FluidCFG(metaclass=ABCMeta):
    def __init__(self, cfg):
        """
        Here for clarification, we still wanna read configure parameter
        from another py file as module

        Note: do not change any parameter after Taichi reads it,
        because Taichi would treat it as a constant value
        (except that that parameter is init as ti.field)
        :param cfg: another py file as module
        """
        self.cfg = cfg

        self.dim = cfg.dim
        self.res = cfg.res
        self.screen_res = cfg.screen_res

        self.dt = cfg.dt
        # pixel length in physical world
        self.dx = cfg.dx

        self.Colliders = cfg.Colliders
        self.Emitters = cfg.Emitters

        self.profile_name = cfg.profile_name
        self.bool_save = cfg.bool_save


    @SetterProperty
    def dt(self, dt):
        self.__dict__['dt'] = dt
        self.__dict__['half_dt'] = 0.5 * dt
        # self.half_dt = 0.5 * self.dt

    @SetterProperty
    def dx(self, dx):
        # to hinder set attribute recursion
        self.__dict__['dx'] = dx
        self.inv_dx = 1.0 / dx

        self.half_dx = 0.5 * dx
        self.half_inv_dx = 0.5 * self.inv_dx

    @SetterProperty
    def bool_save(self, save):
        self.__dict__['bool_save'] = save

        if save:
            self.save_frame_length = self.cfg.save_frame_length
            self.video_manager = self.cfg.video_manager