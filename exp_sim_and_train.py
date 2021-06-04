import taichi as ti
import argparse
from config.CFG_wrapper.eulerCFG import EulerCFG
import os
from importlib import import_module
from config.class_cfg import SchemeType
from Engine import AdvectionProjectionEulerScheme, AdvectionReflectionEulerScheme, Bimocq_Scheme

import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Main file ")

    parser.add_argument("-L", "--LowCfg", help="low resolution cfg path", type=str)
    parser.add_argument("-H", "--HighCfg", help="high resolution cfg path", type=str)
    args = parser.parse_args()

    low_cfg = import_module(args.LowCfg)
    high_cfg = import_module(args.HighCfg)

    return [EulerCFG(low_cfg), EulerCFG(high_cfg)]


def get_scheme(cfg):
    scheme = None
    if cfg.run_scheme == SchemeType.Advection_Projection:
        scheme = AdvectionProjectionEulerScheme(cfg)
    elif cfg.run_scheme == SchemeType.Advection_Reflection:
        scheme = AdvectionReflectionEulerScheme(cfg)
    elif cfg.run_scheme == SchemeType.Bimocq:
        scheme = Bimocq_Scheme(cfg)
    return scheme

class NN_0(torch.nn):
    """
    first lstm in experiment
    """
    def __init__(self, dim, hidden_size):
        self.hidden_size = hidden_size
        if dim == 2:
            self.blk = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
                nn.Tanh()
            )
            self.blkend = nn.Sequential(
                self.nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1))
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        for _ in self.hidden_size:
            x = self.blk()

if __name__ == '__main__':
    ti.init(arch=ti.gpu,
            device_memory_GB=1.0,
            kernel_profiler=True
            )
    m_cfgs = parse_args()
    assert(m_cfgs[0])
    assert(m_cfgs[1])

    solvers = [get_scheme(cfg) for cfg in m_cfgs]

    GUIs = [ti.GUI(cfg.profile_name, tuple(cfg.screen_res), fast_gui=True) for cfg in m_cfgs]

    print("gui_res", [GUI.res for GUI in GUIs])
    print("screen_res", [cfg.screen_res for cfg in m_cfgs])
    for i, solver in enumerate(solvers):
        solver.materialize()

    frame_count = 0
    while all([GUI.running for GUI in GUIs]):
        for i, solver in enumerate(solvers):
            mouse_data = None
            solver.step(mouse_data)
            GUIs[i].set_image(solver.renderer.clr_bffr)
            GUIs[i].show()
            # where the network module insert

            if m_cfgs[i].bool_save:
                for save_what, video_manager in zip(m_cfgs[i].save_what, m_cfgs[i].video_managers):

                    if frame_count < m_cfgs[i].save_frame_length:
                        # blit the corresponding frame
                        solvers[i].renderer.render_frame(save_what)

                        img = solvers[i].renderer.clr_bffr.to_numpy()
                        video_manager.write_frame(img)
                    else:
                        video_manager.make_video(gif=True, mp4=False)
                        # m_cfg.video_manager.get_output_filename(".mp4")
                        video_manager.get_output_filename(".gif")

            if m_cfgs[i].bool_save_grid and frame_count % m_cfgs[i].grid_save_frequency == 0:
                os.makedirs(m_cfgs[i].grid_save_dir, exist_ok=True)
                file_name = os.path.join(m_cfgs[i].grid_save_dir, f"{solvers[i].curFrame:06d}.npz")
                solvers[i].write_grid(file_name)

            m_cfgs[i].frame_count += 1
        if frame_count >= 324:
            break
        frame_count += 1
