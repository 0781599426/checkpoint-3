import numpy as np
from src.kkt.checker import check_kkt
from tests.triangle import c, A, b, pi, pi2, ub
def run(p):
    x = np.full(3, .5)
    return check_kkt(c, np.zeros((0, 3)), np.zeros(0), -A, -b, ub, x, np.zeros(0), -p)
def test_H3():
    r = run(pi); assert r['ok'] and max(r[k] for k in 'abcde') < 1e-9
    r2 = run(pi2)
    assert r2['a'] < 1e-9 and r2['b'] < 1e-9 and r2['c'] < 1e-9
    assert abs(r2['d'] - 2) < 1e-6 and abs(r2['e'] - 1) < 1e-6
    assert abs(b @ pi2 - 8) < 1e-9          # still a lower bound (dual feasible), 1 below zLP
