"""v0.5.0-dev: descendant-span revision, rule descendant_span_v1 (config/descendant_span_v1.json, frozen 2026-10-25).

  S1   descendant span D of one (pair, L): SVD of H = [B_1 | B_2 | ...] (all family blocks), keep left singular
       vectors with s^2 >= 1 - g.  H H^T = sum_f Pi_f, so a near-duplicate of an accepted direction contributes an
       eigenvalue ~ 1 - |c| -> 0 and is dropped; a genuinely new direction contributes ~ 1.
  S1'  comparator: greedy acceptance in the order (level, parent Delta, singular value), accept u iff
       ||(1 - Pi_A) u||^2 >= 1 - g.
  S2   per (C2, X, P) sector: principal pairs (p_i, d_i) of the primary subspace Q and D with cos^2 >= g are
       conflicts; d_i leaves D, p_i leaves Q (Q = Q_clean + span{p_i}).  No Gram-Schmidt in either direction.
  S3b  K^2 prediction on a conflicting descendant direction (FFK 2602.04958 eq. 3.2).
g is the raw_class_weight of the run configuration; no other numerical parameter enters the rule.  NUM_ZERO is a
numerical-zero tolerance (floating-point leakage), not a physical parameter.
"""
import numpy as np

NUM_ZERO = 1e-10


def span_svd(H, g):
    """S1.  Returns (D, s, dropped) with D the kept left singular vectors (None if none) and dropped a list of
    (index, s^2, right singular vector, left singular vector) of the discarded directions."""
    if H is None or H.size == 0:
        return None, np.array([]), []
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    keep = s ** 2 >= 1 - g
    dropped = [(i, float(s[i] ** 2), Vt[i], U[:, i]) for i in np.where(~keep)[0]]
    return (U[:, keep] if keep.any() else None), s, dropped


def span_greedy(cols, g):
    """S1'.  cols: list of unit vectors already in acceptance order.  Returns (D, rejected) with rejected a list
    of (position, residual^2)."""
    A = []; rejected = []
    for k, u in enumerate(cols):
        r = u - (np.column_stack(A) @ (np.column_stack(A).T @ u) if A else 0.0)
        r2 = float(r @ r)
        if r2 >= 1 - g:
            A.append(r / np.sqrt(r2))
        else:
            rejected.append((k, r2))
    return (np.column_stack(A) if A else None), rejected


def family_order(fams):
    """S1' order of family blocks: level n ascending, parent Delta ascending, pid as tie-break; within a block the
    columns are already in decreasing singular value (_orth_top)."""
    return sorted(range(len(fams)), key=lambda k: (fams[k]["level"], fams[k]["parent_Delta"], fams[k]["pid"]))


def conflict_split(Q, D, g):
    """S2 for one sector.  Q: orthonormal primary subspace (n x dQ, zero outside the sector), D: orthonormal
    descendant span (n x dD).  Returns dict(cos2, p, d, Q_clean, D_rest) where p, d hold the conflicting principal
    directions (cos^2 >= g) and D_rest = D without d."""
    if Q is None or D is None or Q.shape[1] == 0 or D.shape[1] == 0:
        return dict(cos2=np.array([]), p=None, d=None, Q_clean=Q, D_rest=D)
    U, s, Vt = np.linalg.svd(Q.T @ D, full_matrices=True)
    c2 = np.clip(s, 0, 1) ** 2
    k = int(np.sum(c2 >= g))
    if k == 0:
        return dict(cos2=c2, p=None, d=None, Q_clean=Q, D_rest=D)
    V = Vt.T
    p = Q @ U[:, :k]; d = D @ V[:, :k]
    Qc = Q @ U[:, k:] if U.shape[1] > k else None
    Dr = D @ V[:, k:] if V.shape[1] > k else None
    return dict(cos2=c2, p=p, d=d, Q_clean=Qc, D_rest=Dr)


def orth_cols(*mats):
    """Orthonormal basis of the joint column span (numerical-zero cut)."""
    A = np.hstack([m for m in mats if m is not None and m.size])
    if A.size == 0:
        return None
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    k = int(np.sum(s > NUM_ZERO))
    return U[:, :k] if k else None


def w(M, r):
    """Weight of basis state r in the column span of the orthonormal M."""
    return float(np.sum(M[r, :] ** 2)) if M is not None and M.size else 0.0


def classify_s2(guarded, wQ, wQc, wD, wDp, g):
    """S2 raw-level order.  Returns the class, or None when the v0.3.5 fall-through rules apply."""
    pPp = bool(guarded and wQc >= g); pDp = bool(wDp >= g)
    if pPp and pDp:
        return "ambiguous_primary_descendant_overlap"
    if pPp:
        return "primary"
    if pDp:
        return "descendant"
    if (guarded and wQ >= g) or wD >= g:
        return "ambiguous_primary_descendant_conflict"
    return None


_RULE_CACHE = {}


def load_rule(spec):
    """The frozen rule file named in cfg['descendant_span'], verified against the sha256 recorded there."""
    import json, hashlib
    path, sha = spec["rule_config"], spec["rule_sha256"]
    if path not in _RULE_CACHE:
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"descendant_span rule file {path} does not match its frozen sha256")
        _RULE_CACHE[path] = json.loads(raw.decode("utf-8"))
    if spec["option"] not in ("B", "A", "S1prime"):
        raise ValueError(f"unknown descendant_span option {spec['option']}")
    return _RULE_CACHE[path]


def lam_desc(D0, n, L, L0):
    """|K|^2 of a level-n, spin-L descendant of a (D0, L0) primary in an exact 3d CFT (FFK 2602.04958 eq. 3.2)."""
    return (2 * D0 + n - 3) * n + L * (L + 1) - L0 * (L0 + 1)


def evidence_b(guarded_weight, K2d, lam, band, lam_min):
    """S3(b) on a conflicting descendant direction d: guarded_weight = weight of d on the guarded KK sources of its
    channel (precondition: 1 - guarded_weight <= NUM_ZERO), K2d = <d|K^2|d>, lam = predicted descendant |K|^2."""
    if not (1.0 - guarded_weight <= NUM_ZERO):
        return "no_evidence_not_guarded"
    if not np.isfinite(lam) or lam < lam_min:
        return "no_evidence_near_short"
    if K2d < band[0] * lam:
        return "supports_P"
    if K2d <= band[1] * lam:
        return "supports_D"
    return "no_evidence_above_band"
