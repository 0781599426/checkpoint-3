"""Triangle problem of team T10 (data, not part of the reference model)."""
import numpy as np
from scipy.optimize import linprog
c = np.array([6., 7., 5.])
A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], float)   # A y >= 1
b = np.ones(3)
pi = np.array([4., 3., 2.]); pi2 = np.array([5., 2., 1.])
ub = np.ones(3)


def lp(rhs=b):
    return linprog(c, A_ub=-A, b_ub=-rhs, bounds=[(0, 1)] * 3, method='highs')
