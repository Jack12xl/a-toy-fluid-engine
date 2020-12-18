import argparse
from config import mpmCFG


def parse_args():
    parser = argparse.ArgumentParser(description="Run which config")

    parser.add_argument("--cfg", help="configure file", type=str)
    args = parser.parse_args()
    cfg = None

    if args.cfg == "Jello-Fall-2D":
        import config.config2D.Jello_Fall2D
        cfg = config.config2D.Jello_Fall2D

    return mpmCFG(cfg)


if __name__ == '__main__':
    m_cfg = parse_args()

    scheme = None
