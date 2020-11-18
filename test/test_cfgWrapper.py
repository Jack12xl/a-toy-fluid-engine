import config.config3D.jet3D as cfg
from config import FluidCFG

if __name__ == "__main__":
    f = FluidCFG(cfg)
    print(f.dx)
    print(f.half_dx)
    print(f.half_inv_dx)