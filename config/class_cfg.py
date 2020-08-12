from enum import Enum

class PreConditioner(Enum):
    Nothing = 0
    Jacobian = 1
    MultiGrid = 2

class SceneEnum(Enum):
    MouseDragDye = 0
    ShotFromBottom = 1
    pass

class VisualizeEnum(Enum):
    Dye = 0
    Velocity = 1
    pass

