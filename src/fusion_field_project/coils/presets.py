import numpy as np
from .geometry import Loop, helmholtz_pair

def single_loop(radius=0.2, current=100.0, turns=50):
    return [Loop(center=np.array([0,0,0]), axis=np.array([0,1,0]),
                 radius=radius, current=current, turns=turns)]

def maxwell_pair(radius=0.2, current=100.0, turns=50):
    # Maxwell pair uses separation = sqrt(3)*radius for 3rd-derivative cancelation
    sep = np.sqrt(3.0) * radius
    return helmholtz_pair(radius, current, turns, sep)

def tokamak_like(R_major=0.35, r_minor=0.08, current=100.0, turns=20, ncoils=24):
    """
    Very rough torus: ncoils circular loops arranged around a circle (major radius).
    Each loop’s axis is tangential around the torus; centers lie in x–z plane.
    """
    loops = []
    for k in range(ncoils):
        phi = 2*np.pi * k / ncoils
        cx, cz = R_major*np.cos(phi), R_major*np.sin(phi)
        # tangent direction around torus in x–z plane:
        axis = np.array([-np.sin(phi), 0.0, np.cos(phi)])  # unit length
        loops.append(Loop(center=np.array([cx, 0.0, cz]),
                          axis=axis, radius=r_minor,
                          current=current, turns=turns))
    return loops