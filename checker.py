"""M3: checker of conditions (a)-(e). Works for ANY solver output: pass x and the dual prices
in the convention pi = d z / d rhs  (eq rows free, '<=' rows pi <= 0)."""
import numpy as np


def check_kkt(c, Aeq, beq, Aub, bub, ub, x, pi_eq, pi_ub, tol=1e-6):
    x = np.asarray(x, float)
    # (a) primal feasibility
    viol_a = max(np.abs(Aeq @ x - beq).max() if len(beq) else 0,
                 max(0.0, (Aub @ x - bub).max()) if len(bub) else 0,
                 max(0.0, (-x).max()), max(0.0, (x - ub).max()))
    # (b) dual feasibility (signs)
    viol_b = max(0.0, np.max(pi_ub)) if len(pi_ub) else 0.0
    # (c) complementary slackness, rows
    viol_c = np.abs(pi_ub * (bub - Aub @ x)).max() if len(bub) else 0.0
    # (d) reduced costs
    r = c - Aeq.T @ pi_eq - Aub.T @ pi_ub
    viol_d = 0.0
    for j in range(len(x)):
        if x[j] <= tol: viol_d = max(viol_d, -r[j])
        elif np.isfinite(ub[j]) and x[j] >= ub[j] - tol: viol_d = max(viol_d, r[j])
        else: viol_d = max(viol_d, abs(r[j]))
    # (e) strong duality (variables with finite upper bound contribute min(0, r_j))
    fin = np.isfinite(ub)
    dual = beq @ pi_eq + bub @ pi_ub + np.minimum(0.0, r)[fin].sum()
    diff_e = abs(c @ x - dual)
    out = dict(a=viol_a, b=viol_b, c=viol_c, d=viol_d, e=diff_e)
    out['ok'] = all(v <= tol for v in (viol_a, viol_b, viol_c, viol_d, diff_e))
    return out
