"""Synthetic tests for the family check (run: python test_family.py)."""

import math

import family as fm


def test_multiplet_dimensions():
    # untruncated level-n content of a generic spin-ell primary: (2 ell + 1) * C(n + 2, 2) states
    for ell in range(4):
        for n in range(5):
            total = sum((2 * j + 1) * fm.mult(n, ell, j, p) for j in range(ell + n + 1) for p in (1, -1))
            assert total == (2 * ell + 1) * math.comb(n + 2, 2), (ell, n, total)


def test_scalar_and_currents():
    assert [fm.mult(1, 0, j, 1) for j in range(3)] == [0, 1, 0]
    assert fm.mult(1, 0, 1, -1) == 0
    assert (fm.mult(2, 0, 0, 1), fm.mult(2, 0, 2, 1), fm.mult(2, 0, 1, 1), fm.mult(2, 0, 1, -1)) == (1, 1, 0, 0)
    # level 1 of a spin-1 primary: j = 0 (+), 1 (-), 2 (+); the conserved current loses its divergence
    assert [fm.mult(1, 1, j, (-1) ** j) for j in range(3)] == [1, 1, 1]
    assert [fm.mult(1, 1, j, -(-1) ** j) for j in range(3)] == [0, 0, 0]
    assert fm.expected(1, 1, 0, 1, True) == 0 and fm.expected(1, 1, 1, -1, True) == 1
    # stress tensor: the spin-1 level-1 state is null
    assert fm.expected(1, 2, 1, 1, True) == 0 and fm.expected(1, 2, 2, -1, True) == 1
    assert fm.expected(1, 2, 3, 1, True) == 1
    # conserved current content is never negative
    for ell in (1, 2):
        for n in range(1, 5):
            for j in range(4):
                for p in (1, -1):
                    assert fm.expected(n, ell, j, p, True) >= 0


def test_channel_sum():
    for ell in range(4):
        for delta in (0.7, 2.3, 5.1):
            js = [j for j in (ell - 1, ell, ell + 1) if j >= 0 and not (ell == 0 and j == 0)]
            assert abs(sum(fm.channel_cft(delta, ell, j) for j in js) - 6 * delta) < 1e-12


def test_propagation():
    order = ["p", "q", "d1", "d2", "m", "u"]
    prim = {"p": {"A": 1.0}, "q": {"B": 0.95}, "m": {"B": 0.3}}
    parents = {"p": ({}, 0.0), "q": ({"p": 1.0}, 0.0), "d1": ({"p": 4.0}, 0.0), "d2": ({"d1": 1.0, "p": 1.0}, 0.0),
               "m": ({"q": 1.0, "d1": 1.0}, 0.0), "u": ({"d1": 1.0}, 3.0)}
    rho, unres = fm.propagate(order, prim, parents)
    assert rho["d1"] == {("A", 1): 1.0}
    assert abs(rho["d2"][("A", 2)] - 0.5) < 1e-12 and abs(rho["d2"][("A", 1)] - 0.5) < 1e-12
    assert abs(rho["q"][("B", 0)] - 0.95) < 1e-12 and abs(rho["q"][("A", 1)] - 0.05) < 1e-12
    assert abs(rho["m"][("B", 0)] - 0.3) < 1e-12 and abs(rho["m"][("B", 1)] - 0.7 * 0.5 * 0.95) < 1e-12
    assert abs(unres["u"] - 0.75) < 1e-12 and abs(rho["u"][("A", 2)] - 0.25) < 1e-12
    for t in order:
        assert abs(sum(rho[t].values()) + unres[t] - 1) < 1e-12
    assert fm.label(prim["p"], rho["p"], unres["p"])[0] == "primary"
    assert fm.label({}, rho["d1"], unres["d1"])[0] == "descendant"
    assert fm.label({}, rho["d2"], unres["d2"])[0] == "family_mixed_level"
    assert fm.label({}, rho["u"], unres["u"])[0] == "unresolved"
    rho2, unres2 = fm.propagate(["p", "q", "x"], prim, {**parents, "x": ({"p": 1.0, "q": 1.0}, 0.0)})
    assert fm.label({}, rho2["x"], unres2["x"])[0] == "mixed"
    try:
        fm.propagate(["d1", "p"], prim, parents)
    except ValueError:
        pass
    else:
        raise AssertionError("parent order not enforced")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
