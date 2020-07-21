import yaml

def read_cfg( cfg_dir : str ):
    with open(cfg_dir, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    return cfg