import config.config3D.jet3D as cfg
from config import FluidCFG, EulerCFG

if __name__ == "__main__":
    f = EulerCFG(cfg)
    print(f.dx)
    print(f.half_dx)
    print(f.half_inv_dx)
    print(f.viscosity_coefficient)
    print(f.poisson_pressure_beta)
    print(f.poisson_viscosity_beta)
    print(f.GasInitAmbientT)