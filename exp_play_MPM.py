import taichi as ti
import taichi_glsl as ts
import numpy as np
import argparse
from config import mpmCFG, MaType


def parse_args():
    parser = argparse.ArgumentParser(description="Run which config")

    parser.add_argument("--cfg", help="configure file", type=str)
    args = parser.parse_args()
    cfg = None

    if args.cfg == "Jello-Fall-2D":
        import config.config2D.Jello_Fall2D
        cfg = config.config2D.Jello_Fall2D

    return mpmCFG(cfg)


if __name__ == '__main__':
    m_cfg = parse_args()

    from Engine.MPM_solver import MPMSolver

    scheme = MPMSolver(m_cfg)
    scheme.materialize()

    # scheme.add_cube(l_b=ts.vec2(0.2),
    #                 cube_size=ts.vec2(0.2),
    #                 mat=MaType.elastic
    #                 )

    gui = ti.GUI(m_cfg.profile_name, tuple(m_cfg.screen_res), fast_gui=False)
    paused = False

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == 'p':
                paused = not paused
            elif e.key == 'r':
                scheme.reset()

        if not paused:
            scheme.step()

        colors = np.array([0xED553B, 0x068587, 0xEEEEF0], dtype=np.uint32)
        gui.circles(scheme.Layout.p_x.to_numpy(), radius=1.5, color=colors[scheme.Layout.p_material_id.to_numpy()])
        gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk

    ti.kernel_profiler_print()
