import taichi as ti
import taichi_glsl as ts
import sys
import math
import os
import config.euler_config
from config.class_cfg import SceneEnum, VisualizeEnum, SchemeType, SimulateType
import config.config3D.scene_config3D.scene_jit3D as scene_cfg
from utils import set_attribute_from_cfg, filterUpCase
from Grid import GRIDTYPE
from DataLayout.Euler import collocatedGridData, MacGridData

FILTER_TYPE = 'm_'
set_attribute_from_cfg(config.euler_config, sys.modules[__name__], FILTER_TYPE, _if_print=False)
set_attribute_from_cfg(scene_cfg, sys.modules[__name__], FILTER_TYPE, _if_print=False)
# set_attribute_from_cfg(config.config3D.basic_config3D, sys.modules[__name__], FILTER_TYPE, _if_print=False)
dim = 3
res = [256, 256, 256]
screen_res = [256, 256]

dx = 1.0 / res[0]
dt = 0.03

v_grid_type = GRIDTYPE.FACE_GRID

SceneType = SceneEnum.Jet
VisualType = VisualizeEnum.Density

SimType = SimulateType.Gas
GasAlpha = 2.0
GasBeta = 2.0
GasInitAmbientT = 23.33
GasMaxT = 85.0


# run scheme
run_scheme = SchemeType.Advection_Projection
CFL = None
Colliders = []

from advection import MacCormackSolver, RK_Order, SemiLagrangeSolver

advection_solver = MacCormackSolver

from projection import RedBlackGaussSedialProjectionSolver, JacobiProjectionSolver

projection_solver = RedBlackGaussSedialProjectionSolver
p_jacobi_iters = 64
dye_decay = 1.0
semi_order = RK_Order.RK_3

# vorticity enhancement
curl_strength = 0.0

DEBUG = False
ti.init(arch=ti.gpu, debug=DEBUG, kernel_profiler=True, device_memory_GB=4.0)
# init should put before init ti.field

from geometry import Transform3, Velocity3
from Emitter import ForceEmitter3, SquareEmitter

fluid_color = ts.vec3(1.0, 1.0, 1.0)

Emitters = []
Emitters.append(SquareEmitter(
    t=Transform3(
        translation=ts.vec3(res[0] // 2, res[2] // 8, res[2] // 2),
        localscale=ts.vec3(res[0] / 32, res[1] / 32, res[2] / 32),
        orientation=ts.vec2(math.pi / 2.0, math.pi / 2.0)  # Up along Y axis
    ),
    v=Velocity3(),
    jet_v=ts.vec3(0.0, 1.0, 0.0),
    jet_t=GasMaxT,
    fluid_color=fluid_color,
    v_grid_type=v_grid_type
)
)

# Emitters.append(SquareEmitter(
#     t=Transform3(
#         translation=ts.vec3(res[0] // 2, res[2] // 8 * 7, res[2] // 2),
#         localscale=ts.vec3(8.0, 8.0, 8.0),
#         orientation=ts.vec2(math.pi / 2.0, math.pi / 2.0)  # Up along Y axis
#     ),
#     v=Velocity3(),
#     jet_v=ts.vec3(0.0, 1.0, 0.0),
#     jet_t=GasMaxT,
#     fluid_color=fluid_color,
#     v_grid_type=v_grid_type
# )
# )

profile_name = '3D' + '-' \
               + 'x'.join(map(str, res)) + '-' \
               + str(v_grid_type) + '-' \
               + str(run_scheme) + '-' \
               + filterUpCase(advection_solver.__name__) + '-' \
               + filterUpCase(projection_solver.__name__) + '-' \
               + str(p_jacobi_iters) + 'it-' \
               + 'RK' + str(int(semi_order)) + '-' \
               + 'curl' + str(curl_strength) + '-' \
               + 'dt-' + str(dt)
if Colliders:
    profile_name += '-Collider'
print(profile_name)

# save to video(gif)
bool_save = False
save_what = [
    VisualizeEnum.Density,
    # VisualizeEnum.Velocity,
    VisualizeEnum.Vorticity
    # VisualizeEnum.Divergence,
    # VisualizeEnum.VelocityMagnitude
]

save_frame_length = 180
save_root = './tmp_result'
save_path = os.path.join(save_root, profile_name)
frame_rate = int(1.0 / dt)

bool_save_ply = False
ply_frequency = 4

