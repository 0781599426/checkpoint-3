"""M4: perturbation runs for the designated constraint (B1 of VM k*)."""
from src.model.model import build
from src.lp.solve import solve_lp


def sensitivity(d, k, h=0.01, tol=1e-6):
    z = {}
    for name, v in (('minus', 1 - h), ('base', 1.0), ('plus', 1 + h)):
        z[name] = solve_lp(build(d, {k: v}))
    base = z['base']
    left = (base['z'] - z['minus']['z']) / h
    right = (z['plus']['z'] - base['z']) / h
    pi = base['pi_eq'][k]
    return dict(left=left, pi=pi, right=right,
                interval_ok=(left - tol <= pi <= right + tol),
                degenerate=abs(right - left) > tol)
