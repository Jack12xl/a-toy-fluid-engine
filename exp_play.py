import utils
# import config.config2D.stable_fluid_mouse as m_cfg
from config import VisualizeEnum
import taichi as ti
from config.class_cfg import SchemeType
from Scheme import AdvectionProjectionEulerScheme, AdvectionReflectionEulerScheme
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run which config")

    parser.add_argument("--cfg", help="configure file", type=str)
    args = parser.parse_args()
    if args.cfg == "mouse2d":
        import config.config2D.stable_fluid_mouse
        cfg = config.config2D.stable_fluid_mouse
    elif args.cfg == "jit2d":
        import config.config2D.stable_fluid_fixed
        cfg = config.config2D.stable_fluid_fixed
    elif args.cfg == "jit3d":
        import config.config3D.jit3D
        cfg = config.config3D.jit3D

    return cfg
if __name__ == '__main__':
    m_cfg = parse_args()

    if m_cfg.run_scheme == SchemeType.Advection_Projection:
        s = AdvectionProjectionEulerScheme(m_cfg)
    elif m_cfg.run_scheme == SchemeType.Advection_Reflection:
        s = AdvectionReflectionEulerScheme(m_cfg)

    gui = ti.GUI(m_cfg.profile_name, tuple(m_cfg.screen_res), fast_gui=True)
    md_gen = utils.MouseDataGen(m_cfg)
    paused = False

    frame_count = 0

    s.materialize()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'p':
                paused = not paused
            elif e.key == 'r':
                s.reset()
            # change visualize type
            elif e.key == ',':
                # TODO
                print(e.key)
            elif e.key == ',':
                # TODO
                print(e.key)
            elif e.key == '1':
                m_cfg.VisualType = VisualizeEnum.Density
            elif e.key == '2':
                m_cfg.VisualType = VisualizeEnum.Velocity
            elif e.key == '3':
                m_cfg.VisualType = VisualizeEnum.Divergence
            elif e.key == '4':
                m_cfg.VisualType = VisualizeEnum.Vorticity
            elif e.key == '5':
                m_cfg.VisualType = VisualizeEnum.VelocityMagnitude

        if not paused:
            mouse_data = md_gen(gui)
            s.step(mouse_data)

        # gui.set_image()
        # too slow
        if m_cfg.screen_res[0] != m_cfg.res[0]:
            import skimage
            import skimage.transform

            img = s.renderer.clr_bffr.to_numpy()
            img = skimage.transform.resize(img, m_cfg.screen_res)
            gui.set_image(img)
        else:
            gui.set_image(s.renderer.clr_bffr)
        gui.show()

        if m_cfg.bool_save:
            if frame_count <= m_cfg.save_frame_length:
                # not sure this would work
                img = s.renderer.clr_bffr.to_numpy()
                m_cfg.video_manager.write_frame(img)
            else:
                m_cfg.video_manager.make_video(gif=True, mp4=False)
                # m_cfg.video_manager.get_output_filename(".mp4")
                m_cfg.video_manager.get_output_filename(".gif")
                break
        frame_count += 1

    ti.kernel_profiler_print()
