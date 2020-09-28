import taichi as ti
from .class_cfg import SceneEnum, VisualizeEnum, SchemeType, SimulateType
import os
from utils import set_attribute_from_cfg
import sys
import config.scene_config.shot_from_bottom_config as scene_cfg
import config.default_config
from geometry import RigidBodyCollider, Ball

debug = False

# simulate_type = SimulateType.Gas

FILTER_TYPE = 'm_'
set_attribute_from_cfg(config.default_config, sys.modules[__name__], FILTER_TYPE, _if_print=False)
set_attribute_from_cfg(scene_cfg, sys.modules[__name__], FILTER_TYPE, _if_print=False)
SceneType = SceneEnum.ShotFromBottom
VisualType = VisualizeEnum.Density
## run Scheme
run_scheme = SchemeType.Advection_Reflection

from advection import MacCormackSolver
# advection_solver = MacCormackSolver

from projection import RedBlackGaussSedialProjectionSolver
projection_solver = RedBlackGaussSedialProjectionSolver
p_jacobi_iters = 30
dye_decay = 0.99

# collider
from geometry import Transform2, Velocity2
Colliders = []
Colliders.append(RigidBodyCollider(Ball(
    transform=Transform2(translation=ti.Vector([300, 150]), localscale=16),
    velocity=Velocity2(velocity_to_world=ti.Vector([0.0, 0.0]), angular_velocity_to_centroid=0.0))))
Colliders.append(RigidBodyCollider(Ball(
    transform=Transform2(translation=ti.Vector([150, 150]), localscale=8),
    velocity=Velocity2(velocity_to_world=ti.Vector([0.0, 0.0]), angular_velocity_to_centroid=0.0))))

# save to video(gif)
bool_save = False
save_frame_length = 240
save_root = './tmp_result'
file_name = 'Projection-MacCormack-GuassSedial-RK2'
save_path = os.path.join(save_root, file_name)
video_manager = ti.VideoManager(output_dir=save_path,
                                framerate=24,
                                automatic_build=False)


# print
# import copy
# tmp = sys.modules[__name__].__dict__.copy()
# for k, v in tmp.items():
#     if (~k.startswith("__")):
#         print(k, v)
print(run_scheme)
print(advection_solver)
print(projection_solver)

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