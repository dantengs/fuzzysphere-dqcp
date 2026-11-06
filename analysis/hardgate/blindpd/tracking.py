"""Step 4: blind cross-N identification of physical primaries.

No energy-rank labels. Within a quantum-number sector (C2, L, P) the points of successive N are
matched by minimum-cost assignment (Hungarian) with cost |Delta_N - prediction|, prediction = linear
extrapolation in 1/N from the track's last two points (else its last point). A match costing more
than max_jump is refused (track ends / new track starts). A match is 'ambiguous' when an alternative
assignment is within ambiguity_margin of the chosen cost (v0.4.1: complete alternative assignment, see
assignment_ambiguous).
"""
import numpy as np


def hungarian(C):
    """Minimum-cost assignment for a rectangular cost matrix (rows <= cols handled by padding)."""
    C = np.asarray(C, float)
    n, m = C.shape
    k = max(n, m)
    big = (np.max(C) if C.size else 0) * 10 + 1e6
    A = np.full((k, k), big); A[:n, :m] = C
    u = np.zeros(k + 1); v = np.zeros(k + 1); p = np.zeros(k + 1, int); way = np.zeros(k + 1, int)
    for i in range(1, k + 1):
        p[0] = i; j0 = 0; minv = np.full(k + 1, np.inf); used = np.zeros(k + 1, bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, k + 1):
                if not used[j]:
                    cur = A[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    res = []
    for j in range(1, k + 1):
        i = p[j] - 1; jj = j - 1
        if i < n and jj < m:
            res.append((i, jj))
    return res


ILLEGAL = 1e6


def assignment_ambiguous(C, assign, i, j, max_jump, margin):
    """v0.4.1 (review R6): full-assignment ambiguity of the accepted match (i, j).  The alternative is the
    minimum-cost assignment that uses only edges <= max_jump and avoids (i, j); it counts only if it keeps at
    least as many accepted matches as the chosen assignment.  Ambiguous iff its total cost exceeds the chosen
    accepted total by less than margin (v0.3.5 compared a single row/column entry instead)."""
    acc = [(a, b) for a, b in assign if C[a, b] <= max_jump]
    base = float(sum(C[a, b] for a, b in acc))
    A = np.where(C <= max_jump, C, ILLEGAL); A[i, j] = ILLEGAL
    alt = [(a, b) for a, b in hungarian(A) if A[a, b] < ILLEGAL]
    if len(alt) < len(acc):
        return False
    return bool(float(sum(C[a, b] for a, b in alt)) - base < margin)


def build_tracks(points, sizes, max_jump=0.3, ambiguity_margin=0.05):
    """points: list of dicts with N, sector (hashable), Delta, phys_id. Returns list of tracks."""
    tracks = []
    by = {}
    for p in points:
        by.setdefault(p["sector"], {}).setdefault(p["N"], []).append(p)
    for sec, perN in sorted(by.items(), key=lambda x: str(x[0])):
        sec_tracks = []
        for N in sizes:
            new = sorted(perN.get(N, []), key=lambda x: x["Delta"])
            open_tr = [t for t in sec_tracks if t["active"]]
            if open_tr and new:
                def pred(t):
                    pts = t["points"]
                    if len(pts) >= 2:
                        (n1, d1), (n2, d2) = (pts[-2]["N"], pts[-2]["Delta"]), (pts[-1]["N"], pts[-1]["Delta"])
                        b = (d2 - d1) / (1 / n2 - 1 / n1)
                        return d2 + b * (1 / N - 1 / n2)
                    return pts[-1]["Delta"]
                C = np.array([[abs(q["Delta"] - pred(t)) for q in new] for t in open_tr])
                assign = hungarian(C)
                taken = set()
                for i, j in assign:
                    if C[i, j] > max_jump:
                        continue
                    q = dict(new[j]); q["match_cost"] = float(C[i, j])
                    q["ambiguous"] = assignment_ambiguous(C, assign, i, j, max_jump, ambiguity_margin)
                    open_tr[i]["points"].append(q); taken.add(j); open_tr[i]["_hit"] = True
                for t in open_tr:
                    if not t.pop("_hit", False):
                        t["active"] = False
                new = [q for j, q in enumerate(new) if j not in taken]
            else:
                for t in open_tr:
                    t["active"] = False
            for q in new:
                q = dict(q); q["match_cost"] = np.nan; q["ambiguous"] = False
                sec_tracks.append(dict(sector=sec, points=[q], active=True))
        tracks += sec_tracks
    for k, t in enumerate(tracks):
        t["track_id"] = f"T{k:04d}"
    return tracks
