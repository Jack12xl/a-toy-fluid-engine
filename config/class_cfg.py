from enum import Enum, IntEnum

class PreConditioner(Enum):
    Nothing = 0
    Jacobian = 1
    MultiGrid = 2

class SceneEnum(Enum):
    MouseDragDye = 0
    ShotFromBottom = 1
    pass

class VisualizeEnum(Enum):
    Density = 0
    Velocity = 1
    pass

class SchemeType(Enum):
    Advection_Projection = 0
    Advection_Reflection = 1

class SimulateType(Enum):
    Liquid = 0
    Gas = 1

class PixelType(IntEnum):
    Liquid = 0
    Collider = 1
    Air = 2

class SurfaceShapeType(IntEnum):
    Base = 0
    Ball = 1
    Square = 2