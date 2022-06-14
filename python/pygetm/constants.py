import enum

RHO0 = 1025.0
GRAVITY = 9.81  #: gravitational acceleration

CENTERS = 1
INTERFACES = 2

ZERO_GRADIENT = 1
SOMMERFELD = 2
CLAMPED = 3
FLATHER_ELEV = 4
SPONGE = 5

FILL_VALUE = -2e20

BAROTROPIC = BAROTROPIC_2D = 1
BAROTROPIC_3D = 2
FROZEN_DENSITY = 3
BAROCLINIC = 4


class TimeVarying(enum.Enum):
    MACRO = enum.auto()
    MICRO = enum.auto()
