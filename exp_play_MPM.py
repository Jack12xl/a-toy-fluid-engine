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
    elif args.cfg == "Jello-Fall-3D":
        import config.config3D.Jello_Fall3D
        cfg = config.config3D.Jello_Fall3D
    return mpmCFG(cfg)


if __name__ == '__main__':
    m_cfg = parse_args()

    from Engine.MPM_solver import MPMSolver

    colors = np.array([0xED553B, 0x068587, 0xEEEEF0, 0xFFFF00], dtype=np.int32)
    scheme = MPMSolver(m_cfg)
    dim = m_cfg.dim
    scheme.materialize()


    def init_fall_cube():
        # scheme.Layout.add_cube(l_b=ts.vec(0.05, 0.4),
        #                        cube_size=ts.vecND(dim, 0.15),
        #                        mat=MaType.liquid,
        #                        n_p=m_cfg.max_n_particle // 4,
        #                        velocity=ts.vecND(dim, 0.0),
        #                        color=colors[MaType.liquid]
        #                        )

        # scheme.Layout.add_cube(l_b=ts.vecND(dim, 0.3),
        #                        cube_size=ts.vecND(dim, 0.15),
        #                        mat=MaType.elastic,
        #                        n_p=m_cfg.max_n_particle // 4,
        #                        velocity=ts.vecND(dim, 0.0),
        #                        color=colors[MaType.elastic]
        #                        )

        # scheme.Layout.add_cube(l_b=ts.vecND(dim, 0.5),
        #                        cube_size=ts.vecND(dim, 0.15),
        #                        mat=MaType.sand,
        #                        n_p=m_cfg.max_n_particle // 4,
        #                        velocity=ts.vecND(dim, 0.0),
        #                        color=colors[MaType.sand]
        #                        )

        scheme.Layout.add_cube(l_b=ts.vecND(dim, 0.4),
                               cube_size=ts.vecND(dim, 0.15),
                               mat=MaType.snow,
                               n_p=m_cfg.max_n_particle // 1,
                               velocity=ts.vecND(dim, 0.0),
                               color=colors[MaType.snow]
                               )


    init_fall_cube()

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
                init_fall_cube()

        if not paused:
            scheme.step()

        np_x = scheme.Layout.p_x.to_numpy()
        if m_cfg.dim == 2:
            screen_pos = np_x
        elif m_cfg.dim == 3:
            screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2 ** 0.5) - 0.2
            screen_y = (np_x[:, 1])
            screen_pos = np.stack([screen_x, screen_y], axis=-1)

        gui.circles(screen_pos, radius=1.5, color=scheme.Layout.p_color.to_numpy())

        gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk

    ti.kernel_profiler_print()
