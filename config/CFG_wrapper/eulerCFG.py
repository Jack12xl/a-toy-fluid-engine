from .fluidCFG import FluidCFG


class EulerCFG(FluidCFG):
    """
    Hold property especially for Euler-based simulation
    """
    def __init__(self, cfg):
        super(EulerCFG, self).__init__(cfg)

