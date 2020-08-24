import yaml
import os
import utils
import config.stable_fluid_cfg as m_cfg
from src.Scheme import EulerScheme
import taichi as ti


if __name__ == '__main__':
    # m_cfg = read_cfg(YAML_PATH)
    ti.init(arch=ti.gpu, debug=m_cfg.debug)

    # cfg_dict = m_cfg.scheme_setting
    s = EulerScheme(m_cfg)

    gui = ti.GUI('Stable-Fluid', tuple(m_cfg.res))
    md_gen = utils.MouseDataGen(m_cfg)
    paused = False
    while True:
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
