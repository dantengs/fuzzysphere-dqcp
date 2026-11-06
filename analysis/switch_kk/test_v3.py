"""Synthetic checks of run.py (no real data). Run: python test_v3.py"""

import numpy as np

import run

ATOL = 1e-8


def test_lowering_weight():
    gap = np.array([-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.7, 1.0, 3.0])
    f, active = run.lowering_weight(gap, 0.5, ATOL)
    np.testing.assert_allclose(f, [1, 1, 1, .75, .6, 0, .4, .25, 0, 0, 0, 0])
    assert not active[5]
    f, _ = run.lowering_weight(gap, 0.0, ATOL)
    np.testing.assert_array_equal(f, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    f, _ = run.lowering_weight(gap, None, ATOL)
    np.testing.assert_allclose(f, np.where(gap == 0, 0, (1 - gap) / 2))
    # bounded weights: every transition above +w is pure P, so unsaved targets never enter K
    for w in (0.25, 0.5):
        f, _ = run.lowering_weight(np.linspace(w + 1e-9, 50, 100), w, ATOL)
        assert np.all(f == 0)
        f, _ = run.lowering_weight(np.array([-1e-6, 1e-6]), w, ATOL)
        np.testing.assert_allclose(f, [0.5, 0.5], atol=1e-5)   # continuous through omega = 0


def test_cut_grid():
    cuts = run.cut_grid({"threshold": 0.042169650342858224, "rule": {"bin_left": -1.5}})
    np.testing.assert_allclose([cuts["band_lo"], cuts["band_hi"]], [10**-1.5, 10**-1.25])
    try:
        run.cut_grid({"threshold": 0.05, "rule": {"bin_left": -1.5}})
    except ValueError:
        return
    raise AssertionError("off-centre threshold accepted")


def toy():
    """L0: vacuum, scalar primary phi (1.5).  L1: vector primary v (2.0), descendant d = P phi at a
    distorted level 2.2 (omega = +0.7 from phi).  Normalised so that [K, P] = 2D holds exactly."""
    blocks = {
        "s0_s0_L0": dict(energy=[0.0, 1.5], c2=[0, 0], x=[1, 1]),
        "s0_s0_L1": dict(energy=[2.0, 2.2], c2=[0, 0], x=[1, 1]),
    }
    parity = {k: [dict(index=str(i + 1), energy=repr(e), C2="0", P="1") for i, e in enumerate(b["energy"])]
              for k, b in blocks.items()}

    def item(target_delta, matrix, frontier, extra_norm=0.0):
        matrix = np.asarray(matrix, float)
        return dict(edge={"target_C2": [0] * len(target_delta)}, matrix=matrix,
                    norm=np.sum(matrix**2, axis=0) + extra_norm, delta=np.asarray(target_delta, float),
                    frontier=frontier, atol=ATOL)

    # L0 -> L1 (multiplicity 3): ||P phi||^2 = 3 * 3 = 9 = 6 * 1.5
    l0 = {1: item([2.0, 2.2], [[0, 0], [0, np.sqrt(3)]], 10.0)}
    # L1 -> L0 (multiplicity 1/3): ||K d||^2 = 9 / 3 = 3 = 2 * 1.5
    # L1 -> L1: rotation element on v (omega = 0) must be ignored; d -> v at omega = -0.2 is tiny
    # L1 -> L2 (multiplicity 5/3): ||P v||^2 = 5/3 * (6 + 1.2) = 12 = 6 * 2.0, of which 1.2 is unsaved
    l1 = {
        0: item([0.0, 1.5], [[0, 0], [0, 3]], 10.0),
        1: item([2.0, 2.2], [[0.7, 0.0], [0.0, 0.0]], 10.0),
        2: item([2.8, 3.2], [[np.sqrt(6), 0], [0, 0]], 3.4, extra_norm=np.array([1.2, 0.0])),
    }
    return blocks, {"s0_s0_L0": l0, "s0_s0_L1": l1}, parity


def test_toy_analysis():
    blocks, indexed, parity = toy()
    cuts = run.cut_grid({"threshold": 0.042169650342858224, "rule": {"bin_left": -1.5}})
    states, modes, sectors = run.analyse(blocks, indexed, parity, cuts, 1.0, 0.0)
    s = {(r["block"], r["raw_index"]): r for r in states}
    assert s["s0_s0_L0", 1]["status"] == "vacuum"
    phi, v, d = s["s0_s0_L0", 2], s["s0_s0_L1", 1], s["s0_s0_L1", 2]
    for m in run.CLASSIFY:
        assert abs(phi[f"{m}_k2"]) < 1e-14 and abs(v[f"{m}_k2"]) < 1e-14       # no P leakage, rotation ignored
        np.testing.assert_allclose(d[f"{m}_k2"], 3.0)
        np.testing.assert_allclose([phi[f"{m}_comm_minus_delta"], v[f"{m}_comm_minus_delta"]], 0, atol=1e-12)
    # paper eq. (4.2) leaks ((1 - 0.7)/2)^2 of phi's descendant into K and fails the central cut
    np.testing.assert_allclose(phi["paper_window_k2"], 3 * 0.15**2 * 3)
    assert phi["paper_window_weight_central"] < run.GATE <= phi["switch_0.50_weight_central"]
    # its window-limited commutator is biased, while the bounded methods are exact
    np.testing.assert_allclose(phi["paper_window_comm_minus_delta_saved"], 3 * 0.7 * 3 / 6 - 1.5)
    assert v["paper_window_comm_minus_delta_saved"] < v["paper_window_comm_minus_delta_lower_bound"] <= 1e-12
    assert phi["support"] == v["support"] == "robust" and d["support"] == "low"
    np.testing.assert_allclose(v["lambda_missing_norm2"], 5 / 3 * 1.2)
    np.testing.assert_allclose(v["equal_energy_lambda_norm2"], 1 / 1 * 0.49)
    assert len([m for m in modes if m["method"] == "switch_0.50"]) == 2 and sectors


def test_frontier_exclusion():
    blocks, indexed, parity = toy()
    indexed["s0_s0_L1"][2]["frontier"] = 2.2 + 0.5            # d needs frontier > 2.7 + atol
    cuts = run.cut_grid({"threshold": 0.042169650342858224, "rule": {"bin_left": -1.5}})
    states, _, _ = run.analyse(blocks, indexed, parity, cuts, 1.0, 0.0)
    s = {(r["block"], r["raw_index"]): r["status"] for r in states}
    assert s["s0_s0_L1", 2] == "frontier_excluded" and s["s0_s0_L1", 1] == "eligible"


def test_cross_casimir_gate():
    cuts = run.cut_grid({"threshold": 0.042169650342858224, "rule": {"bin_left": -1.5}})
    for size, fails in ((1e-7, False), (1e-3, True)):
        blocks, indexed, parity = toy()
        edge = indexed["s0_s0_L1"][2]
        edge["edge"]["target_C2"] = [0, 0, 5]
        edge["delta"] = np.append(edge["delta"], 3.3)
        edge["matrix"] = np.vstack([edge["matrix"], [size * np.sqrt(6), 0]])
        edge["norm"] = edge["norm"] + np.array([(size * np.sqrt(6))**2, 0])
        try:
            run.analyse(blocks, indexed, parity, cuts, 1.0, 0.0)
        except ValueError as error:
            assert fails and "Casimir" in str(error)
        else:
            assert not fails


def test_support_labels():
    def grid(value):
        return {m: {c: value for c in run.BAND} for m in run.CLASSIFY}
    assert run.support(grid(0.95))["support"] == "robust"
    assert run.support(grid(0.5))["support"] == "low"
    w = grid(0.95)
    w["switch_0.50"]["band_lo"] = 0.5
    out = run.support(w)
    assert out["support"] == "sensitive" and out["cut_flip"] and out["method_flip"]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
