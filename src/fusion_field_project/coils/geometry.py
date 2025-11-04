from dataclasses import dataclass
import numpy as np

@dataclass
class Loop:
    center: np.ndarray  # (x,y,z)
    axis: np.ndarray    # unit vector
    radius: float
    current: float
    turns: int = 1
