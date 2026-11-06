import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blindpd.harden import lam_desc, dual_class, valley_threshold_offset, principal_cos
from blindpd.thresholds import valley_threshold


def test():
    # FFK 2602.04958 eq. 3.2 and its special cases
    D0, L0 = 5.02, 4
    assert abs(lam_desc(D0, 1, L0 - 1, L0) - 2 * (D0 - L0 - 1)) < 1e-12          # divergence of a spin-L0 current
    assert abs(lam_desc(0.518, 2, 0, 0) - 2 * (2 * 0.518 - 1)) < 1e-12              # box of an almost-free scalar
    assert abs(lam_desc(0.518, 1, 1, 0) - 2 * 0.518) < 1e-12                        # d_mu phi: |K|^2 = 2 Delta
    assert lam_desc(3.0, 1, 1, 2) == 0.0                                            # conserved spin-2 current: null
    # dual-pass logic: both gates -> conflict; unguarded high w_P is not a P pass
    assert dual_class(True, 0.95, 0.97, 0.9) == (True, True, True)
    assert dual_class(True, 0.95, 0.50, 0.9) == (True, False, False)
    assert dual_class(False, 0.99, 0.10, 0.9) == (False, False, False)
    assert dual_class(False, 0.99, 0.95, 0.9) == (False, True, False)
    # histogram rule: offset 0 reproduces the frozen rule exactly
    rng = np.random.default_rng(1)
    k = np.concatenate([10 ** rng.uniform(-6, -2.5, 300), 10 ** rng.uniform(-1, 0.5, 300)])
    assert valley_threshold_offset(k, (1e-3, 10 ** -0.5), 0.25, 0.0)[0] == valley_threshold(k)[0]
    # principal cosines: identical spans -> 1, orthogonal -> 0
    A = np.eye(4)[:, :2]; B = np.eye(4)[:, 2:]
    assert np.allclose(principal_cos(A, A @ np.array([[1, 2], [3, 4.0]])), 1) and np.allclose(principal_cos(A, B), 0)
    print("HARDEN UNIT TESTS PASS")


if __name__ == "__main__":
    test()
