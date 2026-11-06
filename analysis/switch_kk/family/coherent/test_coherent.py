"""Synthetic tests for coherent.py (run: python -B test_coherent.py)."""

import numpy as np
import pandas as pd

import coherent as co


def test_align_signs():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(5, 4))           # rows a-states, columns T-states
    sigma = np.array([1, -1, -1, 1])
    c = -np.sqrt(3 / 5)
    rows = np.array([2, -1, 0, 3, 1])     # guarded row j -> block position t
    A = np.zeros((5, 5))
    for j, t in enumerate(rows):
        if t >= 0:
            A[j] = sigma[t] * c * B[:, t]
    found, worst = co.align_signs(A, B, rows, c, np.abs(A).max())
    assert worst < 1e-14
    assert {t: s for t, (s, _) in found.items()} == {t: float(sigma[t]) for t in range(4)}


def test_lowdin():
    Z = np.eye(4)[:, :3]
    Zt, lam, dropped = co.lowdin(Z)
    assert np.allclose(Zt, Z) and abs(lam - 1) < 1e-12 and dropped == 0
    v = np.array([1.0, 1, 0, 0]) / np.sqrt(2)
    Z = np.column_stack([np.eye(4)[:, 0], v, v])
    Zt, lam, dropped = co.lowdin(Z)
    P = Zt @ Zt.T
    assert dropped == 1 and np.allclose(P @ P, P) and abs(np.trace(P) - 2) < 1e-10


def test_slot_basis():
    a, b = np.array([1.0, 0, 0]), np.array([0, 2.0, 0])
    U, s = co.slot_basis(np.column_stack([a, b, a + b]), 2)
    assert U.shape == (3, 2) and s[2] / s[1] < 1e-12


def test_labels():
    assert co.label({("F", 0): 0.95})[0] == "primary"
    assert co.label({("F", 1): 0.5})[0] == "uncovered"
    assert co.label({("F", 1): 0.92, ("G", 0): 0.05})[0] == "descendant"
    assert co.label({("F", 1): 0.5, ("F", 2): 0.45})[0] == "family_mixed_level"
    assert co.label({("F", 1): 0.5, ("G", 0): 0.45})[0] == "mixed"


def toy(theta):
    """Scalar primary F (1.0) in L0; L1 has 2.0 and 2.5; L0 has a level-2 state at 3.0.

    P F = a (cos e1 + sin e2); the second L1 primary G is the orthogonal combination."""
    a, b, c, s = 1.7, 1.3, np.cos(theta), np.sin(theta)
    info = {"s0_s0_L0": dict(delta=np.array([1.0, 3.0]), c2=np.zeros(2), x=np.ones(2), p=np.ones(2, int), L=0),
            "s0_s0_L1": dict(delta=np.array([2.0, 2.5]), c2=np.zeros(2), x=np.ones(2), p=np.ones(2, int), L=1)}
    up = np.array([[a * c, 0.0], [a * s, 0.0]])          # L1 <- L0 raising part
    up2 = np.array([[0.0, 0.0], [b * c, b * s]])         # L0 <- L1: level-2 scalar from the P F direction
    phi = {("s0_s0_L1", "s0_s0_L0"): up - up2.T, ("s0_s0_L0", "s0_s0_L1"): up2 - up.T}
    base = dict(L=0, P=1, s_B=0, so5_c2=0, X=1, conserved=False, kind="concentrated", irrep="(0,0)")
    fam = {"F": dict(base, block="s0_s0_L0", delta=1.0),
           "G": dict(base, block="s0_s0_L1", L=1, delta=2.0 * s * s + 2.5 * c * c)}
    vec = {"F": np.array([1.0, 0]), "G": np.array([-s, c])}
    states = pd.DataFrame(dict(block=["s0_s0_L0", "s0_s0_L0", "s0_s0_L1", "s0_s0_L1"], raw_index=[1, 2, 1, 2],
                               status="eligible", so5_c2=0, X=1, P=1, delta=[1.0, 3.0, 2.0, 2.5]))
    cover = {"s0_s0_L0": np.inf, "s0_s0_L1": np.inf}
    praise = co.raising_maps(phi, info, 0.0, 1e-9)
    Q, slots = co.build(fam, vec, praise, info, cover, "hard")
    rho, sectors = co.decompose(fam, vec, Q, slots, info, states, cover, "hard")
    return rho, slots, sectors


def test_toy_coherent():
    theta = 0.3
    rho, slots, sectors = toy(theta)
    assert co.label(rho[("s0_s0_L0", 1)])[0] == "primary"
    e1 = rho[("s0_s0_L1", 1)]
    assert abs(e1[("F", 1)] - np.cos(theta) ** 2) < 1e-12 and abs(e1[("G", 0)] - np.sin(theta) ** 2) < 1e-12
    assert abs(rho[("s0_s0_L0", 2)][("F", 2)] - 1) < 1e-12
    one = slots[(slots.family == "F") & (slots.level == 1) & (slots.j == 1) & (slots.P == 1)].iloc[0]
    assert one["rank"] == 1 and one.deficit == 0 and one.complete
    assert sectors.residual_max_eig.max() < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
