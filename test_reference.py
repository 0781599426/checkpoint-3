from checkpoint3_generator_PUBLIC import cp34_instance
from src.model.model import build, designated_vm
from src.lp.solve import solve_lp
from src.kkt.checker import check_kkt
def test_instance():
    d = cp34_instance(10, 'B', 2); m = build(d)
    assert (m.n_vars, m.n_rows, m.n_fixed) == (561, 116, 51) and designated_vm(d) == 2
    s = solve_lp(m); assert abs(s['z'] - 8.339506) < 1e-6
    assert check_kkt(m.c, m.Aeq, m.beq, m.Aub, m.bub, m.ub, s['x'], s['pi_eq'], s['pi_ub'])['ok']
