import numpy as np
from fusion_field_project.coils.presets import tokamak_like

def test_tokamak_like():
    loops = tokamak_like(0.35, 0.08, 100.0, 20, 24)
    assert len(loops) == 24
    assert np.isclose(np.linalg.norm(loops[0].axis), 1.0, atol=1e-6)