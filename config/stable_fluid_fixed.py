import taichi as ti
from .class_cfg import SceneEnum, VisualizeEnum, SchemeType
import numpy as np
from advection import SemiLagrangeOrder, SemiLagrangeSolver, MacCormackSolver
from projection import JacobiProjectionSolver, RedBlackGaussSedialProjectionSolver
import os
from utils import get_variable_from_module

# dim = 2
# res = [600, 600]
# dx = 1.0
# inv_dx = 1.0 / dx
# half_inv_dx = 0.5 * inv_dx
# dt = 0.03
# half_dt = dt / 2
p_jacobi_iters = 30
f_strength = 10000.0
dye_decay = 0.99
debug = True

import config.default_config

FILTER_TYPE = 'm_'

for k, v in config.default_config.__dict__.items():
    if (k.startswith(FILTER_TYPE)):
        print(k, v)
        vars()[k[len(FILTER_TYPE):]] = v

force_radius = res[0] / 3.0
inv_force_radius = 1.0 / force_radius
inv_dye_denom = 4.0 / (res[0] / 15.0)**2
f_strength_dt = f_strength * dt


SceneType = SceneEnum.ShotFromBottom
fluid_color = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3))

f_gravity = ti.static(9.8)
fluid_shot_direction = ti.Vector([0.0, 1.0])
direct_X_force = f_strength * fluid_shot_direction
source_x = ti.static(res[0] / 2)
source_y = ti.static(0)

VisualType = VisualizeEnum.Dye



#Projection
# projection_solver = RedBlackGaussSedialProjectionSolver





# save to video(gif)
bool_save = False
save_frame_length = 240
save_root = './tmp_result'
file_name = 'Projection-MacCormack-GuassSedial-RK2'
save_path = os.path.join(save_root, file_name)
video_manager = ti.VideoManager(output_dir=save_path,
                                framerate=24,
                                automatic_build=False)

## run Scheme
run_scheme = SchemeType.Advection_Reflection

# if __name__ == '__main__':
#     # print(get_variable_from_module('projection_config'))
#     # print(config.default_config.__dict__.items())
#     for k, v in config.default_config.__dict__.items():
#         if (k.startswith('m_')):
#             print(k, v)
#             vars()[k[2:]] = v
#     print(projection_solver)