import taichi as ti
import argparse
from config import mpmCFG


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

    from Engine.MPM_solver import mpmScheme

    scheme = mpmScheme(m_cfg)

    gui = ti.GUI(m_cfg.profile_name, tuple(m_cfg.screen_res), fast_gui=True)
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

        gui.show()
