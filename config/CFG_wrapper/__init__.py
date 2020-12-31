# this directory holds the wrapper for cfg
# The reason I set this is because I thought transmitting cfg parameter
# from python flle is somehow not elegant and editter unfriendly

# the basic idea is letting these py file do the dirty work
# let the wrapper to serve as the interface
# sry to be this complicated

from .fluidCFG import FluidCFG
from .eulerCFG import EulerCFG
from .mpmCFG import *
from .TwinGrid_mpmCFG import *


