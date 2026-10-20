import numpy as np
from tests.triangle import lp, b
def test_H4():
    eps = 0.1; rhs = b.copy(); rhs[0] += eps
    r = lp(rhs); assert abs(r.fun - (9 + 4 * eps)) < 1e-6
    assert np.allclose(r.x, [(1 + eps) / 2, (1 + eps) / 2, (1 - eps) / 2], atol=1e-6)
    assert abs((lp(b + np.array([1e-3, 0, 0])).fun - 9) / 1e-3 - 4) < 1e-6
