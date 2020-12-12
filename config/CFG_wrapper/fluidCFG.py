import taichi as ti
from abc import ABCMeta, abstractmethod
import os
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

        self.half_dt = None
        self.dt = cfg.dt
        # pixel length in physical world
        self.inv_dx = None
        self.half_dx = None
        self.half_inv_dx = None
        self.dx = cfg.dx

        self.CFL = cfg.CFL

        self.Colliders = cfg.Colliders
        self.Emitters = cfg.Emitters

        self.profile_name = cfg.profile_name
        # save to png(gif, mp4)
        self.save_frame_length = None
        self.video_managers = []
        self.save_what = None
        self.bool_save = cfg.bool_save


    @SetterProperty
    def dt(self, dt):
        self.__dict__['dt'] = dt
        # self.__dict__['half_dt'] = 0.5 * dt
        self.half_dt = 0.5 * self.dt

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
            self.save_what = self.cfg.save_what
            self.save_frame_length = self.cfg.save_frame_length
            for save_thing in self.save_what:
                self.video_managers.append(ti.VideoManager(
                        output_dir=os.path.join(self.cfg.save_path, str(save_thing)),
                        framerate=self.cfg.frame_rate,
                        automatic_build=False
                    )
                )
