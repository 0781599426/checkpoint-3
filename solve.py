"""M2: solve the LP relaxation and read the solution."""
import time
import numpy as np
from scipy.optimize import linprog


def solve_lp(m):
    t0 = time.perf_counter()
    res = linprog(m.c, A_ub=m.Aub, b_ub=m.bub, A_eq=m.Aeq, b_eq=m.beq,
                  bounds=list(zip(np.zeros(len(m.c)), m.ub)), method='highs')
    el = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(res.message)
    # scipy convention: marginals = d(optimal value)/d(rhs); eq free, <= rows <= 0
    return dict(z=res.fun, x=res.x, pi_eq=res.eqlin.marginals, pi_ub=res.ineqlin.marginals,
                time=el, solver='scipy.optimize.linprog (HiGHS)')


def fractional(m, x, tol=1e-6):
    Y = x[:m.M * m.S].reshape(m.M, m.S)
    f = (Y > tol) & (Y < 1 - tol)
    return int(f.sum()), int(f.any(axis=1).sum())   # fractional entries, VMs split
