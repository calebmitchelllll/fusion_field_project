import numpy as np

def field_magnitude(B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(B, axis=1)

def uniformity_score(Bmag_1d: np.ndarray) -> float:
    m = Bmag_1d.mean()
    return 0.0 if m == 0 else max(0.0, 1.0 - Bmag_1d.std()/m)