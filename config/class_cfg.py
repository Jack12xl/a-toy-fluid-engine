from enum import Enum, IntEnum


class PreConditioner(Enum):
    Nothing = 0
    Jacobian = 1
    MultiGrid = 2


class SceneEnum(Enum):
    MouseDragDye = 0
    Jit = 1
    pass


class VisualizeEnum(Enum):
    Density = 0
    Velocity = 1
    Divergence = 2
    Vorticity = 3
    VelocityMagnitude = 4

    def __init__(self, *args):
        super().__init__()
        self.map = ['Density', 'Velocity', 'Div', 'Curl', 'Vel-Norm']

    def __str__(self):
        return self.map[self.value]


class SchemeType(Enum):
    Advection_Projection = 0
    Advection_Reflection = 1

    def __init__(self, *args):
        super().__init__()
        self.map = ['AP', 'AR']

    def __str__(self):
        return self.map[self.value]


class SimulateType(Enum):
    Liquid = 0
    Gas = 1


class PixelType(IntEnum):
    Liquid = 0
    Collider = 1
    Air = 2
    Emitter = 16


class SurfaceShapeType(IntEnum):
    Base = 0
    Ball = 1
    Square = 2
