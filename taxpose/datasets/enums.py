from enum import Enum

class ObjectClass(str, Enum):
    MUG = "mug"
    RACK = "rack"
    GRIPPER = "gripper"
    BOTTLE = "bottle"
    BOWL = "bowl"
    SLAB = "slab"


class Phase(str, Enum):
    GRASP = "grasp"
    PLACE = "place"