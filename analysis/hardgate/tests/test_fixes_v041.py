"""v0.4.1 failure-path tests for review findings R1-R6 of the v0.4.0 code review.
Synthetic data only; no input files are read."""
import sys, os
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blindpd import ope_exact as ox
from blindpd.tracking import assignment_ambiguous, hungarian
from blindpd.harden import primary_regression_ok, shrunk_sector_states, t3_mode_check, worst_window_status


def stageb_row(eps, norm):
    """Toy Stage B contraction (as in the review): x = e3, saved targets T = (e1, e3), z = sqrt(1-eps^2) e1 + eps e2,
    O x = e1 + 1000 e2; the omitted image norm is sqrt(G11 - C11) = 1000."""
    x = np.array([0., 0., 1.]); z = np.array([np.sqrt(1 - eps ** 2), eps, 0.]); T = np.array([[1., 0.], [0., 0.], [0., 1.]])
    edge = dict(name="half_gamma2_L0", insertion="gamma2", rank=0, Ms=[np.array([[1.], [0.]])], scols=[1])
    src = dict(label="x", c2=0, P=1, Delta=1., vec=x)
    prim = [dict(L=0, C2=0, P=1, Delta=1., phys_id="x", vec=x), dict(L=0, C2=2, P=-1, Delta=2., phys_id="z", vec=z)]
    norms = None
    if norm is not None:
        G = np.diag([1000001., 0.])
        if norm == "nan":
            G[0, 0] = np.nan
        norms = {("half", 5, "gamma2", 0, 1): (G, np.diag([1., 0.]), 1)}
    with patch.object(ox, "stageb_edges", return_value=[edge]), \
         patch.object(ox, "stageb_source_order", return_value=({0: "x"}, [src])), \
         patch.object(ox, "stageb_target_basis", return_value=(T, ["z0", "x"])):
        rows, _, _ = ox.recontract_stageb(5, SimpleNamespace(), SimpleNamespace(beta=[1., 0.], vac=0.), prim, pd.DataFrame(),
                                          {("gamma2", "half", 0, 0, 2): [1.], "__signed__": {}}, {("gamma2", 0): 1.},
                                          {(5, "gamma2"): 1.}, 6., "half", imagenorm=norms, imagenorm_sector="half")
    r = next(r for r in rows if r["Z_phys_id"] == "z")
    return r, abs(float(z @ np.array([1., 1000., 0.])) - r["me"])


def test():
    # ---- R3: a tiny omitted weight keeps its rigorous bound; the 5% error is bounded and not certified
    r, err = stageb_row(5e-5, "ok")
    assert r["contraction_status"] == "truncated_uncertified", r["contraction_status"]
    assert r["truncation_bound_me"] >= err - 1e-12 and r["truncation_rel_bound"] > 0.01
    # exactly in the saved span: zero bound, exact label
    r, err = stageb_row(0.0, "ok")
    assert r["contraction_status"] == "exact_in_saved_span" and r["truncation_bound_me"] == 0.0 and err < 1e-12
    # a small error under the 1% limit is certified with a finite non-zero bound
    r, err = stageb_row(2e-6, "ok")
    assert r["contraction_status"] == "truncated_certified" and 0 < r["truncation_rel_bound"] <= 0.01
    assert r["truncation_bound_me"] >= err - 1e-12
    # ---- R4: missing or NaN image Gram is no evidence: never certified, relative bound NaN
    for norm in (None, "nan"):
        r, _ = stageb_row(0.2, norm)
        assert r["contraction_status"] == "truncated_uncertified" and np.isnan(r["truncation_rel_bound"]), norm
        r, _ = stageb_row(5e-5, norm)
        assert r["contraction_status"] == "truncated_uncertified", norm

    # ---- R5: a mode missing from the recomputation fails the regression gate
    b0 = pd.DataFrame(dict(pid=[0, 1, 2], Delta=[1.0, 2.0, 3.0]))
    assert primary_regression_ok(pd.DataFrame(dict(pid=[2, 0, 1], Delta=[3.0, 1.0, 2.0])), b0, 1e-9)
    assert not primary_regression_ok(pd.DataFrame(dict(pid=[0, 1], Delta=[1.0, 2.0])), b0, 1e-9)       # missing pid 2
    assert not primary_regression_ok(pd.DataFrame(dict(pid=[0, 1, 2, 3], Delta=[1, 2, 3, 4.0])), b0, 1e-9)  # extra pid
    assert not primary_regression_ok(pd.DataFrame(dict(pid=[0, 1, 1], Delta=[1, 2, 2.0])), b0, 1e-9)    # duplicate
    assert not primary_regression_ok(pd.DataFrame(dict(pid=[0, 1, 2], Delta=[1, 2, 3.1])), b0, 1e-9)    # Delta moved
    assert not primary_regression_ok(pd.DataFrame(), b0, 1e-9)
    assert primary_regression_ok(pd.DataFrame(), b0.iloc[:0], 1e-9)

    # ---- R1: a surviving guarded sector without a selected primary is a failure, not an absent sector
    blk = dict(X=np.array([1, 1, -1]), P=np.array([1, 1, 1]),
               channels=[dict(C2=14, sectors=[dict(X=1, P=1)], excluded=np.array([2], int))])
    ks = shrunk_sector_states({(0, 0, 2): dict(blk=blk)})
    assert ks[("s0_s0", 2, 14, 1, 1)] == "guarded_sector_present"
    assert ks[("s0_s0", 2, 14, -1, 1)] == "sources_unguarded"
    m = dict(vec=np.array([1.0, 0.0, 0.0]), Delta=5.76)
    c = t3_mode_check(m, [], ks[("s0_s0", 2, 14, 1, 1)], 0.9)
    assert c["sector_present"] and not c["stable"] and c["best_overlap"] == 0.0
    c = t3_mode_check(m, [dict(vec=np.array([1.0, 0.0]), Delta=5.765)], "guarded_sector_present", 0.9)
    assert c["stable"]
    c = t3_mode_check(m, [], "sector_absent", 0.9)
    assert not c["sector_present"] and not c["stable"]
    c = t3_mode_check(m, [], "sources_unguarded", 0.9)
    assert not c["sector_present"]

    # ---- R2: worst status over copies
    assert worst_window_status(pd.Series(["window_stable", "window_unstable"])) == "window_unstable"
    assert worst_window_status(pd.Series(["sector_absent", "window_stable"])) == "window_stable"
    assert worst_window_status(pd.Series(["not_tested", "sector_absent"])) == "sector_absent"

    # ---- R6: full-assignment ambiguity
    C = np.array([[0.0, 0.04], [0.04, 0.0]])          # local gap 0.04 < 0.05 but the swap costs 0.08 in total
    a = hungarian(C)
    assert not any(assignment_ambiguous(C, a, i, j, 0.3, 0.05) for i, j in a)
    C = np.array([[0.0, 0.02], [0.02, 0.0]])          # swap costs 0.04 < 0.05
    a = hungarian(C)
    assert all(assignment_ambiguous(C, a, i, j, 0.3, 0.05) for i, j in a)
    C = np.array([[0.0, 0.01], [0.5, 0.2]])           # the alternative needs an edge above max_jump -> loses a match
    a = hungarian(C)
    assert not any(assignment_ambiguous(C, a, i, j, 0.3, 0.05) for i, j in a)
    C = np.array([[0.1, 0.12, 0.9]])                  # one track, two nearby candidates: genuinely ambiguous
    a = hungarian(C)
    assert a == [(0, 0)] and assignment_ambiguous(C, a, 0, 0, 0.3, 0.05)
    C = np.array([[0.1]])                             # no alternative at all
    assert not assignment_ambiguous(C, hungarian(C), 0, 0, 0.3, 0.05)
    print("V0.4.1 FIX TESTS PASS")


if __name__ == "__main__":
    test()
