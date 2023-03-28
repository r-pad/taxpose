from typing import Dict, List, Set, Tuple

microwave = {
    # In
    "7236_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7128_0": {"x": 0.0, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "door"},
    "7349_0": {"x": 0.0, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "7310_0": {"x": 0.0, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "door"},
    "7366_0": {"x": 0.0, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "7167_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "door"},
    "7263_0": {"x": 0.0, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "door"},
    "7304_0": {"x": 0.0, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "door"},
    "7265_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "door"},
    # Left
    "7236_1": {"x": -1.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7128_1": {"x": -1.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7349_1": {"x": -1.0, "y": 1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7310_1": {"x": -1.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7366_1": {"x": -1.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7167_1": {"x": -1.0, "y": 1.1, "z": 0.1, "ind": 0, "partsem": "door"},
    "7263_1": {"x": -1.0, "y": 1.1, "z": 0.1, "ind": 0, "partsem": "door"},
    "7304_1": {"x": -1.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7265_1": {"x": -1.0, "y": 1.1, "z": 0.1, "ind": 0, "partsem": "door"},
    # Above
    "7236_2": {"x": 0.0, "y": 0.0, "z": 1.05, "ind": 0, "partsem": "door"},
    "7128_2": {"x": 0.0, "y": 0.0, "z": 0.90, "ind": 0, "partsem": "door"},
    "7349_2": {"x": 0.0, "y": 0.0, "z": 1.10, "ind": 0, "partsem": "door"},
    "7310_2": {"x": 0.0, "y": 0.0, "z": 1.00, "ind": 0, "partsem": "door"},
    "7366_2": {"x": 0.0, "y": 0.0, "z": 1.10, "ind": 0, "partsem": "door"},
    "7167_2": {"x": 0.0, "y": 0.0, "z": 1.05, "ind": 0, "partsem": "door"},
    "7263_2": {"x": 0.0, "y": 0.0, "z": 1.05, "ind": 0, "partsem": "door"},
    "7304_2": {"x": 0.0, "y": 0.0, "z": 0.90, "ind": 0, "partsem": "door"},
    "7265_2": {"x": 0.0, "y": 0.0, "z": 0.90, "ind": 0, "partsem": "door"},
    # Right
    "7236_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7128_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7349_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7310_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7366_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7167_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7263_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7304_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7265_3": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
}

# Dishwasher looks for rotation_door
dishwasher = {
    # In
    "11700_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12092_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12428_0": {"x": 0.0, "y": 0.0, "z": 0.8, "ind": 0, "partsem": "rotation_door"},
    "12480_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12530_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12531_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12540_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12553_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    "12592_0": {"x": 0.0, "y": 0.0, "z": 0.3, "ind": 0, "partsem": "rotation_door"},
    # Left
    "11700_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12092_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12428_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12480_1": {"x": 0.0, "y": 1.1, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12530_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12531_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12540_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12553_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12592_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    # Right
    "11700_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12092_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12428_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12480_2": {"x": 0.0, "y": -1.1, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12530_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12531_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12540_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12553_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    "12592_2": {"x": 0.0, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "rotation_door"},
    # Top
    "11700_3": {"x": 0.0, "y": 0.0, "z": 1.3, "ind": 0, "partsem": "rotation_door"},
    "12092_3": {"x": 0.0, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "rotation_door"},
    "12428_3": {"x": 0.0, "y": 0.0, "z": 1.4, "ind": 0, "partsem": "rotation_door"},
    "12480_3": {"x": 0.0, "y": 0.0, "z": 1.0, "ind": 0, "partsem": "rotation_door"},
    "12530_3": {"x": 0.0, "y": 0.0, "z": 1.3, "ind": 0, "partsem": "rotation_door"},
    "12531_3": {"x": 0.0, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "rotation_door"},
    "12540_3": {"x": 0.0, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "rotation_door"},
    "12553_3": {"x": 0.0, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "rotation_door"},
    "12592_3": {"x": 0.0, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "rotation_door"},
}

# Folding chair doesn't look for specific part
foldingchair = {
    # Under
    "100520_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100521_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100523_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100526_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100531_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100532_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100586_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100600_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    "100611_0": {"x": 0.0, "y": 0.0, "z": 0.1, "ind": 0, "partsem": "none"},
    # On
    "100520_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    "100521_1": {"x": -0.2, "y": 0.0, "z": 1.0, "ind": 0, "partsem": "none"},
    "100523_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    "100526_1": {"x": -0.2, "y": 0.0, "z": 1.1, "ind": 0, "partsem": "none"},
    "100531_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    "100532_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    "100586_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    "100600_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    "100611_1": {"x": -0.2, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "none"},
    # Left
    "100520_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100521_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100523_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100526_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100531_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100532_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100586_2": {"x": -0.1, "y": 0.7, "z": 0.1, "ind": 0, "partsem": "none"},
    "100600_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100611_2": {"x": -0.1, "y": 0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    # Right
    "100520_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100521_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100523_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100526_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100531_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100532_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100586_3": {"x": -0.1, "y": -0.7, "z": 0.1, "ind": 0, "partsem": "none"},
    "100600_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
    "100611_3": {"x": -0.1, "y": -0.6, "z": 0.1, "ind": 0, "partsem": "none"},
}

# Oven looks for door
oven = {
    # In
    "7201_0": {"x": -0.1, "y": 0.0, "z": 0.65, "ind": 0, "partsem": "door"},
    "102018_0": {"x": -0.1, "y": 0.0, "z": 0.35, "ind": 0, "partsem": "door"},
    "7332_0": {"x": -0.1, "y": 0.0, "z": 0.65, "ind": 0, "partsem": "door"},
    "7290_0": {"x": -0.1, "y": 0.0, "z": 0.25, "ind": 0, "partsem": "door"},
    "7179_0": {"x": -0.1, "y": 0.0, "z": 0.65, "ind": 0, "partsem": "door"},
    "101917_0": {"x": -0.1, "y": 0.0, "z": 0.5, "ind": 0, "partsem": "door"},
    "101773_0": {"x": -0.1, "y": 0.0, "z": 0.55, "ind": 0, "partsem": "door"},
    "101909_0": {"x": -0.1, "y": 0.0, "z": 0.75, "ind": 0, "partsem": "door"},
    "101940_0": {"x": -0.1, "y": 0.0, "z": 0.5, "ind": 0, "partsem": "door"},
    "101943_0": {"x": -0.1, "y": 0.0, "z": 0.75, "ind": 0, "partsem": "door"},
    "7220_0": {"x": -0.1, "y": 0.0, "z": 0.15, "ind": 0, "partsem": "door"},
    # On
    "7201_1": {"x": -0.1, "y": 0.0, "z": 1.7, "ind": 0, "partsem": "door"},
    "102018_1": {"x": -0.1, "y": 0.0, "z": 1.4, "ind": 0, "partsem": "door"},
    "7332_1": {"x": -0.1, "y": 0.0, "z": 1.7, "ind": 0, "partsem": "door"},
    "7290_1": {"x": -0.1, "y": 0.0, "z": 1.1, "ind": 0, "partsem": "door"},
    "7179_1": {"x": -0.1, "y": 0.0, "z": 1.7, "ind": 0, "partsem": "door"},
    "101917_1": {"x": -0.1, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "door"},
    "101773_1": {"x": -0.1, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "door"},
    "101909_1": {"x": -0.1, "y": 0.0, "z": 1.3, "ind": 0, "partsem": "door"},
    "101940_1": {"x": -0.1, "y": 0.0, "z": 1.4, "ind": 0, "partsem": "door"},
    "101943_1": {"x": -0.1, "y": 0.0, "z": 1.4, "ind": 0, "partsem": "door"},
    "7220_1": {"x": -0.1, "y": 0.0, "z": 1.7, "ind": 0, "partsem": "door"},
    # Left
    "7201_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "102018_2": {"x": -0.1, "y": 1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7332_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7290_2": {"x": -0.1, "y": 1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7179_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101917_2": {"x": -0.1, "y": 1.1, "z": 0.1, "ind": 0, "partsem": "door"},
    "101773_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101909_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101940_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101943_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7220_2": {"x": -0.1, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    # Right
    "7201_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "102018_3": {"x": -0.1, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7332_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7290_3": {"x": -0.1, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "door"},
    "7179_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101917_3": {"x": -0.1, "y": -1.1, "z": 0.1, "ind": 0, "partsem": "door"},
    "101773_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101909_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101940_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "101943_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "7220_3": {"x": -0.1, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "door"},
}

# Fridge looks for door
fridge = {
    # In
    "10036_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    "10068_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    "10620_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    "10655_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    "10685_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    "11231_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    "11299_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 1, "partsem": "door"},
    "10347_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 1, "partsem": "door"},
    "10586_0": {"x": 0.0, "y": -0.2, "z": 1.2, "ind": 0, "partsem": "door"},
    # Left
    "10036_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "10068_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "10620_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "10655_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "10685_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "11231_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
    "11299_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 1, "partsem": "door"},
    "10347_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 1, "partsem": "door"},
    "10586_1": {"x": 0.0, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "door"},
}

# Safe looks for door
safe = {
    # In
    "101363_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101564_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101579_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101583_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101591_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101593_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101594_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "101611_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    "102301_0": {"x": -0.2, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "door"},
    # Top
    "101363_1": {"x": -0.2, "y": 0.0, "z": 1.4, "ind": 0, "partsem": "door"},
    "101564_1": {"x": -0.2, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "door"},
    "101579_1": {"x": -0.2, "y": 0.0, "z": 1.4, "ind": 0, "partsem": "door"},
    "101583_1": {"x": -0.2, "y": 0.0, "z": 1.1, "ind": 0, "partsem": "door"},
    "101591_1": {"x": -0.2, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "door"},
    "101593_1": {"x": -0.2, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "door"},
    "101594_1": {"x": -0.2, "y": 0.0, "z": 1.2, "ind": 0, "partsem": "door"},
    "101611_1": {"x": -0.2, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "door"},
    "102301_1": {"x": -0.2, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "door"},
}

# Drawer looks for drawer
drawer = {
    # In
    "45841_0": {"x": -0.8, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "drawer"},
    "45261_0": {"x": -0.8, "y": 0.0, "z": 0.2, "ind": 4, "partsem": "drawer"},
    "46014_0": {"x": -0.8, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "drawer"},
    "48169_0": {"x": -0.7, "y": 0.0, "z": 0.4, "ind": 0, "partsem": "drawer"},
    "49140_0": {"x": -0.8, "y": 0.0, "z": 0.2, "ind": 0, "partsem": "drawer"},
    # Top
    "45841_1": {"x": -0.5, "y": 0.0, "z": 1.3, "ind": 0, "partsem": "drawer"},
    "45261_1": {"x": -0.5, "y": 0.0, "z": 1.3, "ind": 4, "partsem": "drawer"},
    "46014_1": {"x": -0.5, "y": 0.0, "z": 1.3, "ind": 0, "partsem": "drawer"},
    "48169_1": {"x": 0.0, "y": 0.0, "z": 1.9, "ind": 0, "partsem": "drawer"},
    "49140_1": {"x": -0.5, "y": 0.0, "z": 1.5, "ind": 0, "partsem": "drawer"},
    # Left
    "45841_2": {"x": -0.6, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "45261_2": {"x": -0.6, "y": 0.8, "z": 0.1, "ind": 4, "partsem": "drawer"},
    "46014_2": {"x": -0.6, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "48169_2": {"x": -0.6, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "49140_2": {"x": -0.6, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    # Right
    "45841_3": {"x": -0.6, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "45261_3": {"x": -0.6, "y": -0.8, "z": 0.1, "ind": 4, "partsem": "drawer"},
    "46014_3": {"x": -0.6, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "48169_3": {"x": -0.6, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "49140_3": {"x": -0.6, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
}

# Table looks for drawer
table = {
    # In
    "20279_0": {"x": -0.8, "y": -0.3, "z": 0.8, "ind": 0, "partsem": "drawer"},
    "20985_0": {"x": -0.7, "y": 0.0, "z": 0.5, "ind": 0, "partsem": "drawer"},
    "22367_0": {"x": -0.7, "y": 0.0, "z": 0.7, "ind": 7, "partsem": "drawer"},
    "24644_0": {"x": -0.7, "y": 0.0, "z": 1.0, "ind": 0, "partsem": "drawer"},
    "26503_0": {"x": -0.5, "y": 0.0, "z": 1.2, "ind": 0, "partsem": "drawer"},
    "26525_0": {"x": -0.7, "y": 0.0, "z": 1.3, "ind": 0, "partsem": "drawer"},
    "27044_0": {"x": -0.8, "y": 0.0, "z": 1.1, "ind": 0, "partsem": "drawer"},
    "27189_0": {"x": -1.0, "y": 0.0, "z": 1.1, "ind": 0, "partsem": "drawer"},
    "23807_0": {"x": -0.4, "y": -0.3, "z": 0.7, "ind": 0, "partsem": "drawer"},
    "32601_0": {"x": -0.7, "y": -0.2, "z": 0.9, "ind": 3, "partsem": "drawer"},
    # Left
    "20279_1": {"x": -0.2, "y": 1.0, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "20985_1": {"x": -0.2, "y": 1.0, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "22367_1": {"x": -0.2, "y": 0.8, "z": 0.1, "ind": 7, "partsem": "drawer"},
    "24644_1": {"x": -0.2, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "26503_1": {"x": -0.2, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "26525_1": {"x": -0.2, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "27044_1": {"x": -0.2, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "27189_1": {"x": -0.2, "y": 0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "23807_1": {"x": -0.2, "y": 1.0, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "32601_1": {"x": -0.2, "y": 1.0, "z": 0.1, "ind": 3, "partsem": "drawer"},
    # Right
    "20279_2": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "20985_2": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "22367_2": {"x": -0.2, "y": -0.8, "z": 0.1, "ind": 7, "partsem": "drawer"},
    "24644_2": {"x": -0.2, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "26503_2": {"x": -0.2, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "26525_2": {"x": -0.2, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "27044_2": {"x": -0.2, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "27189_2": {"x": -0.2, "y": -0.8, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "23807_2": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 0, "partsem": "drawer"},
    "32601_2": {"x": -0.2, "y": -1.0, "z": 0.1, "ind": 3, "partsem": "drawer"},
}

# Washer looks for door
washer = {
    # In
    "100282_0": {"x": -0.4, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "door"},
    "100283_0": {"x": -0.4, "y": 0.0, "z": 0.8, "ind": 0, "partsem": "door"},
    "103361_0": {"x": -0.4, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "door"},
    "103369_0": {"x": -0.4, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "door"},
    "103425_0": {"x": -0.4, "y": 0.0, "z": 0.9, "ind": 0, "partsem": "door"},
    "103778_0": {"x": -0.8, "y": 0.0, "z": 0.5, "ind": 0, "partsem": "door"},
    # Top
    "100282_1": {"x": -0.2, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "door"},
    "100283_1": {"x": -0.2, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "door"},
    "103361_1": {"x": -0.2, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "door"},
    "103369_1": {"x": -0.2, "y": 0.0, "z": 1.7, "ind": 0, "partsem": "door"},
    "103425_1": {"x": -0.2, "y": 0.0, "z": 1.6, "ind": 0, "partsem": "door"},
    "103778_1": {"x": -0.2, "y": 0.0, "z": 1.2, "ind": 0, "partsem": "door"},
}

TOPS = {
    "microwave": "2",
    "dishwasher": "3",
    "foldingchair": "1",
    "oven": "1",
    "safe": "1",
    "drawer": "1",
    "washingmachine": "1",
}
INSIDES = {
    "microwave": "0",
    "dishwasher": "0",
    "oven": "0",
    "fridge": "0",
    "safe": "0",
    "drawer": "1",
    "table": "0",
    "washingmachine": "0",
}
LEFTS = {
    "microwave": "1",
    "dishwasher": "1",
    "foldingchair": "2",
    "oven": "2",
    "fridge": "1",
    "drawer": "2",
    "table": "1",
}
RIGHTS = {
    "microwave": "3",
    "dishwasher": "2",
    "foldingchair": "3",
    "oven": "3",
    "drawer": "3",
    "table": "2",
}
UNDERS = {"foldingchair": "0"}

all_objs = {
    "microwave": microwave,
    "dishwasher": dishwasher,
    "chair": foldingchair,
    "oven": oven,
    "fridge": fridge,
    "washingmachine": washer,
    "table": table,
    "drawer": drawer,
    "safe": safe,
}


split_data: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "train": {
        "Fridge": {
            "train": [
                "10036",
                "10068",
                "10655",
                "10685",
                "11299",
                "10347",
                "10586",
            ],
            "test": ["11231", "10620"],
        },
        "Dishwasher": {
            "train": [
                "11700",
                "12092",
                "12428",
                "12480",
                "12530",
                "12531",
                "12592",
            ],
            "test": [
                "12540",
                "12553",
            ],
        },
        "Chair": {
            "train": [
                "100520",
                "100521",
                "100523",
                "100526",
                "100531",
                "100532",
                "100586",
            ],
            "test": [
                "100600",
                "100611",
            ],
        },
        "Washingmachine": {
            "train": [
                "100282",
                "100283",
                "103361",
                "103369",
                "103425",
            ],
            "test": [
                "103778",
            ],
        },
        "Oven": {
            "train": [
                "7201",
                "102018",
                "7332",
                "7290",
                "7179",
                "101917",
                "101773",
                "101909",
                "101940",
            ],
            "test": [
                "101943",
                "7220",
            ],
        },
        "Microwave": {
            "train": [
                "7236",
                "7128",
                "7349",
                "7310",
                "7366",
                "7167",
                "7263",
            ],
            "test": [
                "7304",
                "7265",
            ],
        },
        "Table": {
            "train": [
                "20279",
                "20985",
                "22367",
                "24644",
                "26503",
                "26525",
                "27044",
            ],
            "test": ["27189", "23807", "32601"],
        },
        "Drawer": {
            "train": [
                "45841",
                "45261",
                "46014",
                "48169",
            ],
            "test": [
                "49140",
            ],
        },
        "Safe": {
            "train": [
                "101363",
                "101564",
                "101579",
                "101583",
                "101591",
            ],
            "test": [
                "101593",
                "101594",
                "101611",
                "102301",
            ],
        },
    },
    "test": {},
}

# These are scenes where, for some various, we have a degenerate condition
# where the object is not visible no matter what the camera position is.
BAD_GOAL_SCENES = {
    ("10347", "block", "0"),
    ("10347", "block", "0"),
    ("10347", "disk0", "0"),
    ("10347", "disk1", "0"),
    ("10347", "disk3", "0"),
    ("10347", "slimdisk", "0"),
    ("10347", "suctiontip", "0"),
    ("10347", "bowl", "0"),
    ("10347", "disk2", "0"),
}

BAD_OBS_SCENES: Set[Tuple[str, str, str]] = set()
