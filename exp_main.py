import yaml
import os
from utils import read_cfg
from .config import stable_fluid_cfg as m_cfg


# CONFIG_DIR = './config'
# YAML_BASENAME = 'stable_fluid.yaml'
#
# YAML_PATH = os.path.join(CONFIG_DIR, YAML_BASENAME)

if __name__ == '__main__':
    # m_cfg = read_cfg(YAML_PATH)

    print(m_cfg.debug)
    # for section in m_cfg:
    #     print(section)
    # print(m_cfg["mysql"])
    # print(m_cfg["other"])
    # print(m_cfg["test_num"]["a"] - m_cfg["test_num"]["b"])
