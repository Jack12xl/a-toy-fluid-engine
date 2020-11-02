import utils
import config.stable_fluid_fixed as m_cfg
from Scheme.Euler_Scheme import EulerScheme
import taichi as ti


if __name__ == '__main__':
    s = EulerScheme(m_cfg)

    gui = ti.GUI(m_cfg.profile_name, tuple(m_cfg.screen_res), fast_gui=False)
    md_gen = utils.MouseDataGen(m_cfg)
    paused = False

    frame_count = 0

    s.materialize_collider()

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
                #TODO
                print(e.key)
            elif e.key == ',':
                #TODO
                print(e.key)

        if not paused:
            mouse_data = md_gen(gui)
            s.step(mouse_data)

        # gui.set_image()
        # too slow
        if (m_cfg.screen_res[0] != m_cfg.res[0]):
            import skimage
            import skimage.transform
            img = s.clr_bffr.to_numpy()
            img = skimage.transform.resize(img, m_cfg.screen_res)
            gui.set_image(img)
        else:
            gui.set_image(s.clr_bffr)
        gui.show()

        if (m_cfg.bool_save):
            if (frame_count <= m_cfg.save_frame_length):
                # not sure this would work
                img = s.clr_bffr.to_numpy()
                m_cfg.video_manager.write_frame(img)
            else:
                m_cfg.video_manager.make_video(gif=True, mp4=False)
                # m_cfg.video_manager.get_output_filename(".mp4")
                m_cfg.video_manager.get_output_filename(".gif")
                break
        frame_count += 1

    ti.kernel_profiler_print()