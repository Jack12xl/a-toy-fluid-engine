import yaml
import os
from utils import read_cfg
import config.stable_fluid_cfg as m_cfg
from src.Scheme import EulerScheme

# CONFIG_DIR = './config'
# YAML_BASENAME = 'stable_fluid.yaml'
#
# YAML_PATH = os.path.join(CONFIG_DIR, YAML_BASENAME)

if __name__ == '__main__':
    # m_cfg = read_cfg(YAML_PATH)

    s = EulerScheme(m_cfg.scheme_setting)
    print(m_cfg.debug)
    # for section in m_cfg:
    #     print(section)
    # print(m_cfg["mysql"])
    # print(m_cfg["other"])
    # print(m_cfg["test_num"]["a"] - m_cfg["test_num"]["b"])
