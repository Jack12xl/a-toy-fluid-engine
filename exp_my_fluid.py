import utils
import config.stable_fluid_fixed as m_cfg
from src.Scheme import EulerScheme
import taichi as ti


if __name__ == '__main__':
    # m_cfg = read_cfg(YAML_PATH)
    # ti.init(arch=ti.gpu, debug=m_cfg.debug,kernel_profiler=True)

    # cfg_dict = m_cfg.scheme_setting
    s = EulerScheme(m_cfg)

    gui = ti.GUI('Stable-Fluid', tuple(m_cfg.res))
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

        if not paused:
            mouse_data = md_gen(gui)
            s.step(mouse_data)

        img = s.clr_bffr.to_numpy()
        gui.set_image(img)
        gui.show()


        if (m_cfg.bool_save):
            if (frame_count <= m_cfg.save_frame_length):
                m_cfg.video_manager.write_frame(img)
            else:
                m_cfg.video_manager.make_video(gif=True, mp4=False)
                # m_cfg.video_manager.get_output_filename(".mp4")
                m_cfg.video_manager.get_output_filename(".gif")
        frame_count += 1

    ti.kernel_profiler_print()