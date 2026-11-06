"""Unit tests of descendant_span_v1 (dry_run_acceptance.unit_tests of config/descendant_span_v1.json), synthetic data."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blindpd.descspan import (span_svd, span_greedy, family_order, conflict_split, classify_s2, evidence_b, lam_desc,
                              orth_cols, w)
from blindpd.spectrum import _orth

G = 0.9
proj = lambda M: M @ M.T


def test():
    # 1. near-duplicate directions (1,0) and (1,0.01) merge into one direction (v0.3.5 kept two)
    u1 = np.array([1.0, 0.0]); u2 = np.array([1.0, 0.01]); u2 /= np.linalg.norm(u2)
    H = np.column_stack([u1, u2])
    assert _orth(H).shape[1] == 2                                   # the v0.3.5 defect: second coordinate gets weight 1
    D, s, dropped = span_svd(H, G)
    assert D.shape[1] == 1 and len(dropped) == 1 and dropped[0][1] < 1e-4
    assert w(D, 1) < 1e-4 and abs(w(D, 0) - 1) < 1e-4               # e2 is not a descendant any more
    Dg, rej = span_greedy([u1, u2], G)
    assert Dg.shape[1] == 1 and len(rej) == 1 and rej[0][0] == 1

    # 2. two orthogonal families: D unchanged with respect to v0.3.5
    e = np.eye(3)
    H = np.column_stack([e[:, 0], e[:, 1]])
    D, _, dropped = span_svd(H, G)
    assert not dropped and np.allclose(proj(D), proj(_orth(H)))
    Dg, rej = span_greedy([e[:, 0], e[:, 1]], G)
    assert not rej and np.allclose(proj(Dg), proj(D))
    # a partially new direction (residual^2 = 0.5 >= 0.1) is kept by both rules
    v = (e[:, 0] + e[:, 1]) / np.sqrt(2)
    assert span_svd(np.column_stack([e[:, 0], v]), G)[0].shape[1] == 2
    assert span_greedy([e[:, 0], v], G)[0].shape[1] == 2

    # 3. d_P + d_D > n_s forces a conflict (cos^2 = 1); the state is ambiguous, not P
    Q = e[:2, :1]                                                    # n_s = 2, primary e1
    H = np.column_stack([np.array([0.0, 1.0]), np.array([1.0, 1.0]) / np.sqrt(2)])
    D, _, dropped = span_svd(H, G)
    assert D.shape[1] == 2 and not dropped                           # 1 + 2 > 2
    cs = conflict_split(Q, D, G)
    assert cs["p"].shape[1] == 1 and abs(cs["cos2"][0] - 1) < 1e-12 and cs["Q_clean"] is None
    assert cs["D_rest"].shape[1] == 1 and abs(abs(cs["D_rest"][1, 0]) - 1) < 1e-12
    C = orth_cols(cs["p"], cs["d"])
    r0 = classify_s2(True, w(Q, 0), w(cs["Q_clean"], 0), w(D, 0), w(cs["D_rest"], 0), G)
    r1 = classify_s2(True, w(Q, 1), w(cs["Q_clean"], 1), w(D, 1), w(cs["D_rest"], 1), G)
    assert r0 == "ambiguous_primary_descendant_conflict" and r1 == "descendant" and abs(w(C, 0) - 1) < 1e-12
    # no conflict when the principal cos^2 < g; Q and D untouched
    Q2 = np.array([[1.0], [0.0]]); D2 = np.array([[np.sqrt(0.5)], [np.sqrt(0.5)]])
    cs2 = conflict_split(Q2, D2, G)
    assert cs2["p"] is None and cs2["Q_clean"] is Q2 and cs2["D_rest"] is D2

    # 4. S3(b): a direction with weight outside the guarded sources (missing K edge) gives no evidence
    assert evidence_b(0.8, 0.0, 10.0, (0.5, 2.0), 1.0) == "no_evidence_not_guarded"
    assert evidence_b(1.0, 0.0, 0.5, (0.5, 2.0), 1.0) == "no_evidence_near_short"
    assert evidence_b(1.0, 1.0, 10.0, (0.5, 2.0), 1.0) == "supports_P"
    assert evidence_b(1.0, 9.0, 10.0, (0.5, 2.0), 1.0) == "supports_D"
    assert evidence_b(1.0, 25.0, 10.0, (0.5, 2.0), 1.0) == "no_evidence_above_band"
    assert evidence_b(np.nan, 1.0, 10.0, (0.5, 2.0), 1.0) == "no_evidence_not_guarded"

    # 5. S2 class order (symmetric)
    assert classify_s2(True, 1.0, 0.95, 1.0, 0.95, G) == "ambiguous_primary_descendant_overlap"
    assert classify_s2(True, 0.95, 0.95, 0.2, 0.2, G) == "primary"
    assert classify_s2(False, 0.95, 0.95, 0.95, 0.95, G) == "descendant"           # unguarded: never P
    assert classify_s2(True, 0.95, 0.1, 0.95, 0.1, G) == "ambiguous_primary_descendant_conflict"
    assert classify_s2(True, 0.5, 0.5, 0.5, 0.5, G) is None
    assert classify_s2(False, 0.95, 0.95, 0.2, 0.2, G) is None                      # falls through to unclassifiable

    # 6. lambda (same formula as the frozen T6)
    assert abs(lam_desc(0.518, 1, 1, 0) - 2 * 0.518) < 1e-12 and lam_desc(3.0, 1, 1, 2) == 0.0

    # 7. S1' order: level, then parent Delta, then pid
    fams = [dict(level=2, parent_Delta=1.0, pid=5), dict(level=1, parent_Delta=3.0, pid=1), dict(level=1, parent_Delta=2.0, pid=9)]
    assert family_order(fams) == [2, 1, 0]

    # 8. S3 decision and S4 merges
    from blindpd.resolve_s3 import decide, phys_level_class, phys_mode_identity
    assert decide(True, False, False) == "P" and decide(False, True, False) == "D"
    assert decide(True, True, False) is None and decide(True, False, True) is None and decide(False, False, False) is None
    assert phys_level_class(["primary", "ambiguous_primary_descendant_conflict"]) == "ambiguous_primary_descendant_conflict"
    assert phys_level_class(["primary_resolved", "descendant"]) == "ambiguous_cross_copy_split"
    assert phys_level_class(["primary", "mixed_primary_descendant"]) == "primary"
    assert phys_mode_identity(["clean", "conflict_unresolved"]) == "conflict"
    assert phys_mode_identity(["clean", "descendant_resolved"]) == "conflict"
    assert phys_mode_identity(["primary_resolved", "clean"]) == "primary_resolved"
    assert phys_mode_identity(["clean"]) == "clean"
    print("DESCSPAN UNIT TESTS PASS")


if __name__ == "__main__":
    test()
