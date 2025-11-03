import numpy as np
from fusion_field_project.simulation.metrics import field_magnitude, uniformity_score

def test_metrics():
    B = np.array([[1.0,0,0],[0,2.0,0]])
    m = field_magnitude(B)
    assert m.shape == (2,)
    s = uniformity_score(m)
    assert 0.0 <= s <= 1.0