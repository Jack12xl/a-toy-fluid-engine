import taichi as ti
import config.euler_config
from config.class_cfg import SceneEnum, VisualizeEnum, SchemeType
import config.config3D.scene_config3D.scene_jit3D as scene_cfg
from utils import set_attribute_from_cfg, filterUpCase
import sys

FILTER_TYPE = 'm_'
set_attribute_from_cfg(config.euler_config, sys.modules[__name__], FILTER_TYPE, _if_print=False)
set_attribute_from_cfg(scene_cfg, sys.modules[__name__], FILTER_TYPE, _if_print=False)
set_attribute_from_cfg(config.config3D.basic_config3D, sys.modules[__name__], FILTER_TYPE, _if_print=False)

Colliders = []
Emitters = []