import taichi as ti
from .class_cfg import SceneEnum, VisualizeEnum, SchemeType, SimulateType
import os
from utils import set_attribute_from_cfg, filterUpCase
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
run_scheme = SchemeType.Advection_Projection

from advection import MacCormackSolver, SemiLagrangeSolver, SemiLagrangeOrder
advection_solver = MacCormackSolver

from projection import RedBlackGaussSedialProjectionSolver, JacobiProjectionSolver, ConjugateGradientProjectionSolver
projection_solver = RedBlackGaussSedialProjectionSolver
p_jacobi_iters = 30
dye_decay = 0.99
semi_order = SemiLagrangeOrder.RK_3

# vorticity enhancement
curl_strength = 7.0

# collider
from geometry import Transform2, Velocity2
ti.init(arch=ti.gpu, debug=debug, kernel_profiler=True)
# init should put before init ti.field

Colliders = []
# Colliders.append(RigidBodyCollider(Ball(
#     transform=Transform2(translation=ti.Vector([300, 250]), localscale=16),
#     velocity=Velocity2(velocity_to_world=ti.Vector([0.0, -10.0]),angular_velocity_to_centroid=15.0))))
# Colliders.append(RigidBodyCollider(Ball(
#     transform=Transform2(translation=ti.Vector([150, 150]), localscale=8),
#     velocity=Velocity2(velocity_to_world=ti.Vector([0.0, 0.0]), angular_velocity_to_centroid=-5.0))))

profile_name = str(res[0]) + 'x' + str(res[1]) + '-' \
               + str(run_scheme) + '-' \
               + filterUpCase(advection_solver.__name__) + '-' \
               + filterUpCase(projection_solver.__name__) + '-' \
               + str(p_jacobi_iters) + 'it-' \
               + 'RK' + str(int(semi_order)) + '-' \
               + 'curl' + str(curl_strength)
if (Colliders):
    profile_name += '-Collider'
print(profile_name)

# save to video(gif)
bool_save = False

save_frame_length = 240
save_root = './tmp_result'
save_path = os.path.join(save_root, profile_name)
video_manager = ti.VideoManager(output_dir=save_path,
                                framerate=24,
                                automatic_build=False)