from tests.triangle import lp
def test_H5():
    zl = lp().fun; assert zl <= 11 + 1e-9
    assert abs((11 - zl) / 11 - 2 / 11) < 1e-9 and abs((18 - zl) / 18 - .5) < 1e-9
