from enum import Enum, IntEnum


class PreConditioner(Enum):
    Nothing = 0
    Jacobian = 1
    MultiGrid = 2


class SceneEnum(Enum):
    MouseDragDye = 0
    Jet = 1
    pass


class VisualizeEnum(Enum):
    Density = 0
    Velocity = 1
    Divergence = 2
    Vorticity = 3
    VelocityMagnitude = 4
    Temperature = 5

    def __init__(self, *args):
        super().__init__()
        self.map = ['Density', 'Velocity', 'Div', 'Curl', 'Vel-Norm']

    def __str__(self):
        return self.map[self.value]


class SchemeType(Enum):
    Advection_Projection = 0
    Advection_Reflection = 1
    Bimocq = 2

    def __init__(self, *args):
        super().__init__()
        self.map = ['AP', 'AR', 'BMcq']

    def __str__(self):
        return self.map[self.value]


class SimulateType(IntEnum):
    Test = 0
    Liquid = 1
    Gas = 2
    Flame = 3


class PixelType(IntEnum):
    Liquid = 0
    Collider = 1
    Air = 2
    Emitter = 16


class SurfaceShapeType(IntEnum):
    Base = 0
    Ball = 1
    Square = 2
