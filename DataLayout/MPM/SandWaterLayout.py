import numpy as np
import taichi as ti
import taichi_glsl as ts
from abc import ABCMeta, abstractmethod
from config.CFG_wrapper import mpmCFG, DLYmethod, MaType, BC
from utils import Int, Float, Matrix, Vector
from Grid import CellGrid
from .dataLayout import mpmLayout

@ti.data_oriented
class doubleGridLayout(mpmLayout)
    """
    Especially designed for coupling of Sand 
    and water
    """

    def __init__(self, cfg: ):