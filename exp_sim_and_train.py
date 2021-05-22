import taichi as ti
import argparse
import importlib.util
from config.CFG_wrapper.eulerCFG import EulerCFG
import os
from importlib import import_module
from config.class_cfg import SchemeType
from Engine import AdvectionProjectionEulerScheme, AdvectionReflectionEulerScheme, Bimocq_Scheme


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


            m_cfgs[i].frame_count += 1
        if frame_count >= 324:
            break
        frame_count += 1
