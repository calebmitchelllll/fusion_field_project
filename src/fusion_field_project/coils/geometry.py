from dataclasses import dataclass
import numpy as np

@dataclass
class Loop:
    center: np.ndarray  # (x,y,z)
    axis: np.ndarray    # unit vector
    radius: float
    current: float
    turns: int = 1

def helmholtz_pair(radius: float, current: float, turns: int, separation: float):
    """Two coaxial loops separated by 'separation' along +y/-y."""
    c = separation / 2.0
    loop1 = Loop(center=np.array([0, -c, 0]), axis=np.array([0,1,0]), radius=radius, current=current, turns=turns)
    loop2 = Loop(center=np.array([0,  c, 0]), axis=np.array([0,1,0]), radius=radius, current=current, turns=turns)
    return [loop1, loop2]