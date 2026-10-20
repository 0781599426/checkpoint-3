"""python -m experiments.run_all  -> regenerates Tables 1-3 and the active-row table (results/*.csv)"""
import csv, os
import numpy as np
from checkpoint3_generator_PUBLIC import cp34_instance
from src.model.model import build, designated_vm
from src.lp.solve import solve_lp, fractional
from src.kkt.checker import check_kkt
from src.sens.sens import sensitivity

SEED, STORY, VARIANT = 10, 'B', 2


def write(name, header, rows):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{name}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    d = cp34_instance(SEED, STORY, VARIANT)
    print('check values:', d['S'], d['M'], d['cpu_cap'].sum(), d['mem_cap'].sum(),
          d['cpu_req'].sum(), d['mem_req'].sum())
    m = build(d)
    sol = solve_lp(m)
    nfrac, nsplit = fractional(m, sol['x'])
    write('table1', ['S', 'M', 'fixed_to_0', 'variables', 'rows', 'zLP', 'solver', 'time_s',
                     'fractional_entries', 'VMs_split'],
          [[m.S, m.M, m.n_fixed, m.n_vars, m.n_rows, f"{sol['z']:.6f}", sol['solver'],
            f"{sol['time']:.4f}", nfrac, nsplit]])
    kk = check_kkt(m.c, m.Aeq, m.beq, m.Aub, m.bub, m.ub, sol['x'], sol['pi_eq'], sol['pi_ub'])
    write('table2', ['a_primal', 'b_dual_sign', 'c_comp_slack', 'd_reduced_cost', 'e_strong_duality'],
          [[f"{kk[k]:.2e}" for k in 'abcde']])
    k = designated_vm(d)
    s = sensitivity(d, k)
    write('table3', ['vm_k', 'left_slope', 'pi', 'right_slope', 'interval_ok', 'degenerate'],
          [[k, f"{s['left']:.6f}", f"{s['pi']:.6f}", f"{s['right']:.6f}", s['interval_ok'], s['degenerate']]])
    act = []
    slack = m.bub - m.Aub @ sol['x']
    act.append(['B1', m.M, m.M, int((np.abs(sol['pi_eq']) > 1e-9).sum())])
    for fam in ('B2', 'B3', 'B4'):
        idx = [i for i, f in enumerate(m.fam_ub) if f == fam]
        act.append([fam, len(idx), int((np.abs(slack[idx]) < 1e-6).sum()),
                    int((np.abs(sol['pi_ub'][idx]) > 1e-9).sum())])
    write('active_rows', ['family', 'rows', 'tight', 'nonzero_dual'], act)
    print(f"zLP={sol['z']:.6f} frac={nfrac} split={nsplit} KKT ok={kk['ok']} {kk}")
    print('sens', s)


if __name__ == '__main__':
    main()
