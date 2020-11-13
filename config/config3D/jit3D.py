import taichi as ti
import taichi_glsl as ts
import sys
import math
import os
import config.euler_config
from config.class_cfg import SceneEnum, VisualizeEnum, SchemeType
import config.config3D.scene_config3D.scene_jit3D as scene_cfg
from utils import set_attribute_from_cfg, filterUpCase


FILTER_TYPE = 'm_'
set_attribute_from_cfg(config.euler_config, sys.modules[__name__], FILTER_TYPE, _if_print=False)
set_attribute_from_cfg(scene_cfg, sys.modules[__name__], FILTER_TYPE, _if_print=False)
set_attribute_from_cfg(config.config3D.basic_config3D, sys.modules[__name__], FILTER_TYPE, _if_print=False)

SceneType = SceneEnum.Jit
VisualType = VisualizeEnum.Density

# run scheme
run_scheme = SchemeType.Advection_Projection
Colliders = []

from advection import MacCormackSolver, SemiLagrangeOrder, SemiLagrangeSolver

advection_solver = SemiLagrangeSolver

from projection import RedBlackGaussSedialProjectionSolver, JacobiProjectionSolver
projection_solver = JacobiProjectionSolver
p_jacobi_iters = 30
dye_decay = 0.99
semi_order = SemiLagrangeOrder.RK_2

# vorticity enhancement
curl_strength = 0.0

DEBUG = False
ti.init(arch=ti.gpu, debug=DEBUG, kernel_profiler=True)
# init should put before init ti.field

from geometry import Transform3, Velocity3
from Emitter import ForceEmitter3
Emitters = []
Emitters.append(ForceEmitter3(
    sys.modules[__name__],
    t=Transform3(
        translation=ts.vec3(res[0] // 2, 0, res[0] // 2),
        localscale=ts.vec3(1000.0),
        orientation=ts.vec2(math.pi / 2.0, math.pi / 2.0) # Up along Y axis
    ),
    v=Velocity3(),
    # force_radius=res[0] / 3.0
    force_radius = 256.0
)
)

dt = 0.0001

profile_name = '3D' + '-'\
               + 'x'.join(map(str, res)) + '-' \
               + str(VisualType) + '-' \
               + str(run_scheme) + '-' \
               + filterUpCase(advection_solver.__name__) + '-' \
               + filterUpCase(projection_solver.__name__) + '-' \
               + str(p_jacobi_iters) + 'it-' \
               + 'RK' + str(int(semi_order)) + '-' \
               + 'curl' + str(curl_strength) + '-' \
               + 'dt-' + str(dt)
if (Colliders):
    profile_name += '-Collider'
print(profile_name)

# save to video(gif)
bool_save = False

save_frame_length = 120
save_root = './tmp_result'
save_path = os.path.join(save_root, profile_name)
video_manager = ti.VideoManager(output_dir=save_path,
                                framerate=24,
                                automatic_build=False)