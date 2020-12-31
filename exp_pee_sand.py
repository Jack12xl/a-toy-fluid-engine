import taichi as ti
import taichi_glsl as ts
import numpy as np
import argparse
from config import TwinGridmpmCFG, MaType


def parse_args():
    parser = argparse.ArgumentParser(description="Run which config")

    parser.add_argument("--cfg", help="configure file", type=str)
    args = parser.parse_args()
    cfg = None

    if args.cfg == "pee2D":
        import config.config2D.pee_2D
        cfg = config.config2D.pee_2D
    elif args.cfg == "pee3D":
        import config.config3D.Jello_Fall3D
        cfg = config.config3D.Jello_Fall3D
    return TwinGridmpmCFG(cfg)


if __name__ == '__main__':
    m_cfg = parse_args()

    from Engine.MPM_solver import MPMSolver

    colors = np.array([0xED553B, 0x068587, 0xEEEEF0, 0x8a6844], dtype=np.int32)
    solver = MPMSolver(m_cfg)
    dim = m_cfg.dim
    solver.materialize()


    def init_fall_cube():
        solver.Layout.add_liquid_cube(l_b=ts.vecND(dim, 0.3),
                                      cube_size=ts.vecND(dim, 0.15),
                                      n_p=m_cfg.max_n_w_particle // 4,
                                      velocity=ts.vecND(dim, 0.0),
                                      color=colors[MaType.liquid]
                                      )

        solver.Layout.add_sand_cube(l_b=ts.vecND(dim, 0.5),
                                    cube_size=ts.vecND(dim, 0.15),
                                    n_p=m_cfg.max_n_particle // 4,
                                    velocity=ts.vecND(dim, 0.0),
                                    color=colors[MaType.sand]
                                    )


    def update_jet(total_frame_jet: int, n_jet_p: int):
        """

        :param total_frame_jet:
        :param n_jet_p:
        :return:
        """
        tmp_pos = ts.vecND(m_cfg.dim, 0.15)
        tmp_pos[0] = 0.0

        tmp_vel = ts.vecND(m_cfg.dim, 0.0)
        tmp_vel[0] = 1.5
        solver.Layout.add_liquid_cube(l_b=tmp_pos,
                                      cube_size=ts.vecND(dim, 0.03),
                                      n_p=n_jet_p // total_frame_jet,
                                      velocity=tmp_vel,
                                      color=colors[MaType.liquid]
                                      )


    init_fall_cube()

    gui = ti.GUI(m_cfg.profile_name, tuple(m_cfg.screen_res), fast_gui=False)
    paused = False

    # jet_frame = 64
    while gui.running:
        # if scheme.curFrame < jet_frame:
        #     update_jet(jet_frame, m_cfg.max_n_particle // 4)

        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == 'p':
                paused = not paused
            elif e.key == 'r':
                solver.reset()
                init_fall_cube()

        if not paused:
            solver.step()

        np_x = solver.Layout.p_x.to_numpy()
        if m_cfg.dim == 2:
            screen_pos = np_x
        elif m_cfg.dim == 3:
            # screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2 ** 0.5) - 0.2
            # screen_y = (np_x[:, 1])
            # screen_pos = np.stack([screen_x, screen_y], axis=-1)
            np_x -= 0.5

            phi, theta = np.radians(28), np.radians(32)
            x, y, z = np_x[:, 0], np_x[:, 1], np_x[:, 2]

            c, s = np.cos(phi), np.sin(phi)
            C, S = np.cos(theta), np.sin(theta)

            x, z = x * c + z * s, z * c - x * s
            u, v = x, y * C + z * S

            screen_pos = np.array([u, v]).swapaxes(0, 1) + 0.5

        solver.Layout.update_liquid_color()
        gui.circles(screen_pos, radius=1.5, color=solver.Layout.p_color.to_numpy())
        if not m_cfg.bool_save:
            gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
        else:
            if solver.curFrame <= m_cfg.save_frame_length:
                import os

                os.makedirs(m_cfg.save_path, exist_ok=True)
                gui.show(os.path.join(m_cfg.save_path, f'{solver.curFrame:06d}.png'))

            else:
                break

        if m_cfg.bool_save_particle:
            if solver.curFrame < m_cfg.save_frame_length:
                import os

                os.makedirs(m_cfg.particle_path, exist_ok=True)
                file_name = os.path.join(m_cfg.particle_path, f'{solver.curFrame:06d}.npz')
                # print(file_name)
                solver.write_particles(file_name)

    ti.kernel_profiler_print()
