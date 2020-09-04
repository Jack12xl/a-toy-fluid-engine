import taichi as ti
from .class_cfg import SceneEnum, VisualizeEnum, SchemeType, SimulateType
import os
from utils import set_attribute_from_cfg
import sys
import config.scene_config.shot_from_bottom_config as scene_cfg
import config.default_config

debug = True

# simulate_type = SimulateType.Gas

FILTER_TYPE = 'm_'
set_attribute_from_cfg(config.default_config, sys.modules[__name__], FILTER_TYPE, _if_print=True)
set_attribute_from_cfg(scene_cfg, sys.modules[__name__], FILTER_TYPE, _if_print=True)
SceneType = SceneEnum.ShotFromBottom
VisualType = VisualizeEnum.Density
## run Scheme
run_scheme = SchemeType.Advection_Reflection

from advection import MacCormackSolver
advection_solver = MacCormackSolver
dye_decay = 0.99

# save to video(gif)
bool_save = False
save_frame_length = 240
save_root = './tmp_result'
file_name = 'Projection-MacCormack-GuassSedial-RK2'
save_path = os.path.join(save_root, file_name)
video_manager = ti.VideoManager(output_dir=save_path,
                                framerate=24,
                                automatic_build=False)



# if __name__ == '__main__':
#     import sys
#     thismodule = sys.modules[__name__]
#     print(thismodule.__dict__.items())
#     # print(get_variable_from_module('projection_config'))
#     # print(config.default_config.__dict__.items())
#     for k, v in config.default_config.__dict__.items():
#         if (k.startswith('m_')):
#             print(k, v)
#             vars()[k[2:]] = v
#     print(projection_solver)