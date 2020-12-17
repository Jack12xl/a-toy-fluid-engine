import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run which config")

    parser.add_argument("--cfg", help="configure file", type=str)
    args = parser.parse_args()
    cfg = None