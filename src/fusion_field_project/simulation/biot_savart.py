import numpy as np
from scipy.constants import mu_0
from fusion_field_project.coils.geometry import Loop

def _rotation_matrix_from_y(axis: np.ndarray) -> np.ndarray:
    """Rotation that maps +y to given unit axis (Rodrigues)."""
    a = axis / (np.linalg.norm(axis) + 1e-30)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, a)
    c = float(np.dot(y, a))
    s = np.linalg.norm(v)
    if s < 1e-12:  # already aligned or opposite
        return np.eye(3) if c > 0 else np.diag([1,-1,1])
    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R

def _loop_segments(loop: Loop, nseg: int = 200):
    """Discretize a circular loop into points P[k] and segment vectors dL[k]."""
    theta = np.linspace(0, 2*np.pi, nseg, endpoint=False)
    # Local frame: loop lies in x–z plane, normal +y
    x = loop.radius * np.cos(theta)
    z = loop.radius * np.sin(theta)
    y = np.zeros_like(x)
    P_local = np.stack([x, y, z], axis=1)
    # segment vectors between consecutive points
    P_next = np.roll(P_local, -1, axis=0)
    dL_local = P_next - P_local
    # rotate to world frame so +y maps to axis
    R = _rotation_matrix_from_y(loop.axis)
    P_world = (P_local @ R.T) + loop.center
    dL_world = dL_local @ R.T
    return P_world, dL_world

def _biot_savart_segments(points: np.ndarray, P: np.ndarray, dL: np.ndarray, I: float):
    """
    Sum B from polyline segments for all observer points.
    points: (N,3), P: (M,3) segment start points, dL: (M,3) segment vectors
    Returns B: (N,3)
    """
    N = points.shape[0]
    M = P.shape[0]
    B = np.zeros((N, 3), dtype=float)
    # vectorized over segments in manageable chunks to save memory
    chunk = 256
    for i0 in range(0, M, chunk):
        i1 = min(M, i0 + chunk)
        r_vec = points[:, None, :] - P[None, i0:i1, :]           # (N, m, 3)
        r3 = np.linalg.norm(r_vec, axis=2)**3 + 1e-30            # (N, m)
        dLxR = np.cross(dL[i0:i1][None, :, :], r_vec, axis=2)    # (N, m, 3)
        contrib = (mu_0 / (4*np.pi)) * I * (dLxR / r3[..., None]) # (N, m, 3)
        B += contrib.sum(axis=1)
    return B

def field_of_loops(loops, points: np.ndarray, nseg: int = 200) -> np.ndarray:
    """
    Compute B-field [Tesla] at 'points' (N,3) from a list of Loop objects
    using a discretized Biot–Savart integral. Multiplies by 'turns'.
    """
    points = np.asarray(points, dtype=float)
    B = np.zeros((points.shape[0], 3), dtype=float)
    for L in loops:
        P, dL = _loop_segments(L, nseg=nseg)
        B += _biot_savart_segments(points, P, dL, I=L.current) * max(1, int(L.turns))
    return B