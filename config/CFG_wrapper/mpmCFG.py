from .fluidCFG import FluidCFG
import taichi as ti


class mpmCFG(FluidCFG):
    """
    Property for mpm-based simulation
    """

    def __init__(self, cfg):
        """

        :param cfg: A module,
                    contains the config for each each simulated scene
        """
        super(mpmCFG, self).__init__(cfg)

        pass
