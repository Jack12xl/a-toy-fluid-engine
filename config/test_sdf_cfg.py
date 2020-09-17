from utils import set_attribute_from_cfg
import config.default_config as default_cfg
import sys


FILTER_TYPE = 'm_'
set_attribute_from_cfg(default_cfg, sys.modules[__name__], FILTER_TYPE, False)
res = [512, 512]