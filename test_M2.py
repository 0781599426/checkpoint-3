import numpy as np
from tests.triangle import c, A, b, lp
def test_H2():
    h = np.full(3, .5); assert (A @ h >= b - 1e-12).all() and abs(c @ h - 9) < 1e-9
    assert abs(lp().fun - 9) < 1e-6
    up = np.ones(3); down = np.zeros(3)
    assert (A @ up >= b).all() and c @ up == 18 and not (A @ down >= b).all()
    assert abs((11 - 9) / 11 - 0.181818) < 1e-5
