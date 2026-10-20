"""M1: reference model B2 (cloud, zones) in matrix form.
min c^T x  s.t.  Aeq x = beq,  Aub x <= bub,  0 <= x <= ub  (t_l has ub = inf)
x = [y_00..y_(M-1)(S-1) (row-major k,l), t_0..t_(S-1)]"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from checkpoint3_generator_PUBLIC import TANGENT_POINTS


class Model:
    pass


def build(d, rhs_b1=None):
    """rhs_b1: optional dict {k: value} replacing the right-hand side of row (B1) of VM k."""
    S, M = d['S'], d['M']
    C = d['cpu_cap'].astype(float); R = d['mem_cap'].astype(float)
    cpu = d['cpu_req'].astype(float); mem = d['mem_req'].astype(float)
    nv = M * S + S
    y = lambda k, l: k * S + l
    t = lambda l: M * S + l
    c = np.zeros(nv); c[M * S:] = 1.0
    ub = np.ones(nv); ub[M * S:] = np.inf
    fixed = 0
    for k in range(M):
        zk = d['vm_required_zone'][k]
        for l in range(S):
            if zk != -1 and d['zone'][l] != zk:
                ub[y(k, l)] = 0.0; fixed += 1          # variable kept, upper bound 0
    Aeq = lil_matrix((M, nv)); beq = np.ones(M)
    for k in range(M):
        for l in range(S): Aeq[k, y(k, l)] = 1.0
    if rhs_b1:
        for k, v in rhs_b1.items(): beq[k] = v
    A = lil_matrix((2 * S + 4 * S, nv)); b = []; fam = []; r = 0
    for l in range(S):                                  # (B2)
        for k in range(M): A[r, y(k, l)] = cpu[k]
        b.append(C[l]); fam.append('B2'); r += 1
    for l in range(S):                                  # (B3)
        for k in range(M): A[r, y(k, l)] = mem[k]
        b.append(R[l]); fam.append('B3'); r += 1
    for l in range(S):                                  # (B4)  a_m*load/C_l - t_l <= -b_m
        for u in TANGENT_POINTS:
            a = 1 / (1 - u) ** 2; bm = -u ** 2 / (1 - u) ** 2
            for k in range(M): A[r, y(k, l)] = a * cpu[k] / C[l]
            A[r, t(l)] = -1.0
            b.append(-bm); fam.append('B4'); r += 1
    m = Model()
    m.c, m.Aeq, m.beq = c, csr_matrix(Aeq), beq
    m.Aub, m.bub, m.ub = csr_matrix(A), np.array(b), ub
    m.fam_ub = fam; m.n_fixed = fixed
    m.n_vars = nv; m.n_rows = M + len(b)
    m.S, m.M = S, M
    return m


def designated_vm(d):
    """VM with the largest cpu_req (first index on ties)."""
    return int(np.argmax(d['cpu_req']))
