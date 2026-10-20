import itertools, numpy as np
from tests.triangle import c, A, b, lp
def test_H1():
    assert A.shape == (3, 3) and len(c) == 3                  # 3 variables, 3 rows
    r = lp(); assert abs(r.fun - 9) < 1e-6
    r.ineqlin.marginals                                       # rows written as <=: prices are -pi
    assert np.allclose(-r.ineqlin.marginals, [4, 3, 2], atol=1e-6)
    feas = [y for y in itertools.product([0, 1], repeat=3) if (A @ np.array(y) >= b).all()]
    assert len(feas) == 4 and min(np.dot(c, y) for y in feas) == 11
