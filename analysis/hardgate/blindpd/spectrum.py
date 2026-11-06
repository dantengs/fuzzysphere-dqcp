"""Steps 2-3 for one N: primary modes, descendant families (P images), raw-level classes.

Per (pair, L) the classification basis is the block eigen-window, or - when the KK stage solved a
larger "guarded_lowest_window" target window for that (pair, L) - that extended window.  Extended
windows are separate eigensolves; their eigenvector signs are fixed relative to the block window with
the exact reduced Hermitian conjugacy  M_rev^T = (-1)^(L'-L) sqrt((2L'+1)/(2L+1)) M_fwd  (verified to
1e-14 on all reused pairs).  Only block states are KK sources, so states that exist only in the
extended window can be classified as descendants but never certified as primaries.
v0.3.5: edges with status "block_top_frontier" land in the block window itself (like "reused_run_window").  When
the KK sources of a (pair, L) are a saved target window (block entry source_window == "target"), its extended
window is that same eigensolve: the gauge is identical (column SHAs verified in io.SizeData) and no sign fixing is done.
"""
import numpy as np
from .io import BLOCK_TARGET_STATUS
from .kk import block_kk, primary_modes
from .so5 import rep_name
from . import descspan as dsp

LMAX = 4


def _orth(A, rtol=1e-6, atol=1e-6):
    """Orthonormal basis of the column span; singular values <= max(atol, rtol*s0) dropped
    (atol removes null images, e.g. P on the vacuum or the divergence of a conserved current)."""
    if A is None or A.size == 0:
        return None
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= atol:
        return None
    k = int(np.sum(s > max(atol, rtol * s[0])))
    return U[:, :k] if k else None


def multiplicity(n, l, Lp):
    """Number of spin-Lp states in P^n O_l for generic Delta (3d): P^n -> spins n, n-2, ..., (0|1)."""
    return sum(1 for j in range(n, -1, -2) if abs(j - l) <= Lp <= j + l)


NULL_SV = 0.3   # a genuine P step has singular value ~ sqrt(2(Delta+..)) >~ 1; below 0.3 = null/leakage


def _orth_top(A, m, atol=NULL_SV):
    """Top-m left singular directions of A with singular value > atol; also the next (dropped) value."""
    if A is None or A.size == 0 or m <= 0:
        return None, None, None
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    k = int(min(m, np.sum(s > atol)))
    nxt = float(s[k]) if s.size > k else 0.0
    return (U[:, :k], s[:k], nxt) if k else (None, None, nxt)


def conj_factor(L, Lp):
    return (-1) ** (Lp - L) * np.sqrt((2 * Lp + 1) / (2 * L + 1))


class Basis:
    """Classification basis of one (pair, L)."""

    def __init__(self, sd, a, b, L, parity_df):
        key = f"s{a}_s{b}_L{L}"
        blk = sd.blocks[key]
        self.blockE = np.array(blk["energy"]); nb = len(self.blockE)
        ext = [sd.edges[k] for k in sd.edges if k[0] == a and k[1] == b and k[3] == L
               and sd.edges[k]["target_status"] not in BLOCK_TARGET_STATUS]
        self.extended = bool(ext)
        if not ext:
            self.E = self.blockE; self.C2 = np.array(blk["c2"]); self.X = np.array(blk["x"])
            self.P = sd.parity[key]; self.to_blk = np.arange(nb); self.sign = np.ones(nb)
            full = nb >= blk["dim"]
            self.complete_to = np.inf if full else float(sd.delta(self.blockE.max()))
            self.target_coords = {}
        else:
            e0 = ext[0]
            for e in ext:
                if list(e["target_E"]) != list(e0["target_E"]):
                    raise ValueError(f"inconsistent extended windows for {key}")
            self.E = np.array(e0["target_E"]); self.C2 = np.array(e0["target_C2"])
            if hasattr(sd, "window_basis"):                      # stageB sector: full window carries its own labels
                Ew, Cw, Xw, Pw, frw = sd.window_basis(L)
                if len(Ew) != len(self.E) or np.max(np.abs(Ew - self.E)) > 1e-10:
                    raise ValueError(f"stageB window mismatch {key}")
                self.P = Pw; self.X = Xw
                self.same_gauge = True                       # sources are a sub-window of the same eigensolve
            elif blk.get("source_window") == "target":       # v0.3.5: sources are this very window
                if list(self.E) != list(blk["energy"]):
                    raise ValueError(f"target source window differs from its extended window {key}")
                self.P = sd.parity[key]; self.X = np.array(blk["x"])
                self.same_gauge = True
            else:
                pt = parity_df[(parity_df.sA == a) & (parity_df.sB == b) & (parity_df.L == L) &
                               (parity_df.window == "target")].sort_values("index")
                if len(pt) != len(self.E) or np.max(np.abs(pt.energy.values - self.E)) > 1e-9:
                    raise ValueError(f"target parity rows do not match {key}")
                self.P = pt.P.values.astype(int); self.X = pt.X.values.astype(int)
            self.to_blk = np.full(len(self.E), -1)
            for j, x in enumerate(self.E):
                i = np.argmin(np.abs(self.blockE - x))
                if abs(self.blockE[i] - x) < 1e-8 and abs(blk["c2"][i] - self.C2[j]) < 1e-5:
                    self.to_blk[j] = i
            fr = e0["target_frontier"]
            self.complete_to = np.inf if fr is None else float(sd.delta(fr))
            self.sign = np.ones(len(self.E)) if getattr(self, "same_gauge", False) else np.full(len(self.E), np.nan)
            self.target_coords = {(e["a"], e["b"], e["li"], e["lf"]): True for e in ext}
        self.key = key; self.n = len(self.E)

    def fix_signs(self, sd, a, b, L, bases=None):
        """Signs of block states inside the extended window from the exact reduced Hermitian conjugacy
        M_rev^T = c(L_x,L) M_fwd with every available partner L_x.  The reverse edge may land in a block
        window (block gauge) or in another extended window whose signs are already fixed (chained)."""
        if not self.extended:
            return dict(status="block_window")
        if getattr(self, "same_gauge", False):
            return dict(status="same_eigensolve_gauge", unresolved=0)
        if not hasattr(self, "_stats"):
            self._stats = dict(worst=0.0, used=set(), checks=0)
        st = self._stats
        for (aa, bb, Lx, Lp) in sorted(sd.edges):
            if (aa, bb, Lp) != (a, b, L) or Lx > LMAX or Lx == L or (a, b, L, Lx) not in sd.edges:
                continue
            fwd = sd.edge_arrays(a, b, Lx, L)            # rows: extended window of L, cols: block L_x
            rev = sd.edge_arrays(a, b, L, Lx)            # cols: block L,  rows: window at L_x
            if fwd["status"] in BLOCK_TARGET_STATUS:
                continue
            if rev["status"] in BLOCK_TARGET_STATUS:
                revb = rev["M"]
            elif bases is not None and Lx in bases and bases[Lx].extended and not np.all(np.isnan(bases[Lx].sign)):
                bx = bases[Lx]; revb = np.zeros((len(bx.blockE), rev["M"].shape[1]))
                ok = (bx.to_blk >= 0) & ~np.isnan(bx.sign)
                revb[bx.to_blk[ok]] = bx.sign[ok][:, None] * rev["M"][ok]
            else:
                continue
            c = conj_factor(Lx, L)
            for j, i in enumerate(self.to_blk):
                if i < 0:
                    continue
                row = fwd["M"][j, :]
                u = int(np.argmax(np.abs(row)))
                if abs(row[u]) < 1e-6 or abs(revb[u, i]) < 1e-6:
                    continue
                r = revb[u, i] / (c * row[u])
                if not np.isnan(self.sign[j]) and np.sign(r) != self.sign[j]:
                    raise ValueError(f"inconsistent conjugacy signs in {self.key}")
                self.sign[j] = np.sign(r); st["worst"] = max(st["worst"], abs(abs(r) - 1.0)); st["used"].add(Lx); st["checks"] += 1
        unresolved = int(np.sum(np.isnan(self.sign[self.to_blk >= 0])))
        # an unresolved sign can only matter for composition through this window; weights are sign-invariant
        return dict(status="fixed_by_conjugacy" if not unresolved else "partially_fixed",
                    partners=sorted(st["used"]), conjugacy_checks=st["checks"],
                    max_magnitude_mismatch=st["worst"], unresolved=unresolved)

    # coordinate maps
    def from_block(self, F):
        """block-coordinate rows -> basis coordinates (extras zero)."""
        out = np.zeros((self.n,) + F.shape[1:])
        m = self.to_blk >= 0
        s = np.where(np.isnan(self.sign[m]), 1.0, self.sign[m])   # weights are sign-invariant
        out[m] = (s[:, None] * F[self.to_blk[m]]) if F.ndim == 2 else s * F[self.to_blk[m]]
        return out

    def to_block(self, V, nblock):
        """basis-coordinate vectors -> block coordinates (extras dropped; unknown signs dropped)."""
        out = np.zeros((nblock,) + V.shape[1:])
        m = self.to_blk >= 0
        s = np.where(np.isnan(self.sign[m]), 1.0, self.sign[m])   # consistent with from_block (round trip = id)
        out[self.to_blk[m]] = s[:, None] * V[m] if V.ndim == 2 else s * V[m]
        return out


def analyse_size(sd, cfg, kappa_thr, parity_df, extra=None):
    """v0.4.0: when `extra` is a dict it receives, per (a, b, L), the classification basis, the KK channels, the
    primary subspaces at the three thresholds, the orthonormal descendant span and the families (read by
    harden.py).  The returned rows are unchanged."""
    Dmax = cfg["Delta_max"]; wcut = cfg["raw_class_weight"]
    # v0.5.0-dev: descendant_span_v1 is active only when the configuration names it; otherwise v0.4.1 exactly
    dspec = cfg.get("descendant_span"); opt = dspec["option"] if dspec else None
    rule = dsp.load_rule(dspec) if dspec else None
    pairs = sorted(set((int(k[1]), int(k[4])) for k in sd.blocks))
    prim_rows, raw_rows, fam_rows, mode_vecs = [], [], [], []
    gates = dict(max_sector_leak=0.0, windows=[])
    if opt:
        gates["ds_conflicts"], gates["ds_dropped"] = [], []
    pid = 0
    for (a, b) in pairs:
        blks, bas = {}, {}
        for L in range(LMAX + 1):
            if f"s{a}_s{b}_L{L}" in sd.blocks:
                blks[L] = block_kk(sd, a, b, L)
                bas[L] = Basis(sd, a, b, L, parity_df)
        for _ in range(3):                                   # chained conjugacy sign fixing
            for L in bas:
                bas[L].fix_signs(sd, a, b, L, bas)
        for L in bas:
            st = bas[L].fix_signs(sd, a, b, L, bas)
            gates["windows"].append(dict(N=sd.N, key=bas[L].key, extended=bas[L].extended,
                                         complete_to_Delta=bas[L].complete_to, **st))
        # ---- step 2: primary subspaces (block coords) at thr, thr/3, thr*3
        prim = {L: [] for L in blks}
        Qs = {t: {L: {} for L in blks} for t in ("mid", "lo", "hi")}
        for L, blk in blks.items():
            nL = len(blk["E"])
            for ch in blk["channels"]:
                for sec in ch["sectors"]:
                    key = (ch["C2"], sec["X"], sec["P"])
                    alt = {}
                    for tag, thr in (("lo", kappa_thr / 3), ("hi", kappa_thr * 3), ("mid", kappa_thr)):
                        modes, Q = primary_modes(sd, blk, sec, thr)
                        F = np.zeros((nL, Q.shape[1])); F[sec["ids"], :] = Q
                        Qs[tag][L][key] = bas[L].from_block(F)
                        alt[tag] = np.array([x["Delta"] for x in modes])
                    for m in modes:                       # last loop iteration = "mid"
                        mode_vecs.append(None)
                        m["in_lo"] = bool(alt["lo"].size and np.min(np.abs(alt["lo"] - m["Delta"])) < 0.01)
                        m["in_hi"] = bool(alt["hi"].size and np.min(np.abs(alt["hi"] - m["Delta"])) < 0.01)
                        v = np.zeros(nL); v[sec["ids"]] = m["vec"]
                        m["full"] = v; m["pid"] = pid; pid += 1
                        mode_vecs[-1] = dict(pid=m["pid"], block=f"s{a}_s{b}", L=L, C2=key[0], X=key[1], P=key[2],
                                             Delta=m["Delta"], vec=v)
                        prim[L].append((m, key))
        # ---- raising maps in basis coordinates:  R[(L,Lp)] : basis(L) -> basis(Lp)
        R = {}
        for L in blks:
            for Lp in (L - 1, L, L + 1):
                if (a, b, L, Lp) not in sd.edges:
                    continue
                e = sd.edge_arrays(a, b, L, Lp)
                Rr = e["M"] * (e["tE"][:, None] > e["sE"][None, :] + e["atol"])   # rows: edge targets, cols: block L
                if Lp in bas and not (bas[Lp].extended and e["status"] not in BLOCK_TARGET_STATUS):
                    Rr = bas[Lp].from_block(Rr) if Lp <= LMAX else Rr
                R[(L, Lp)] = Rr
        famB = {L: [] for L in blks}
        pinfo, prow = {}, {}                                 # v0.5.0-dev: parent (Delta, L) and row of each mode
        for L0 in blks:
            for (m, key) in prim[L0]:
                Dp = m["Delta"]
                if Dp < 0.1:          # identity: no descendants (P|0> = 0); finite-N leakage is not a family
                    m["Pnorm_ratio"] = np.nan; m["level1_deviation"] = np.nan
                    prim_rows.append(dict(N=sd.N, pid=m["pid"], block=f"s{a}_s{b}", L=L0, C2=key[0], rep=rep_name(key[0]),
                                          X=key[1], P=key[2], Delta=Dp, sigma_Delta=m["sigma_Delta"], K2=m["K2"],
                                          kappa=m["kappa"], dominant_index=m["dominant_index"] + 1,
                                          dominant_weight=m["dominant_weight"], Pnorm_ratio=np.nan,
                                          level1_deviation=np.nan, in_window=True, stable_kappa_x3=True))
                    pinfo[m["pid"]] = (Dp, L0); prow[m["pid"]] = prim_rows[-1]
                    continue
                pn = 0.0
                for Lp in (L0 - 1, L0, L0 + 1):
                    if (L0, Lp) in R:
                        pn += (2 * Lp + 1) / (2 * L0 + 1) * float(np.sum((R[(L0, Lp)] @ m["full"]) ** 2))
                m["Pnorm_ratio"] = pn / (6 * Dp) if Dp > 1e-9 else np.nan
                spans = {0: {L0: bas[L0].from_block(m["full"][:, None])}}
                lvl1 = []
                for n in range(1, cfg["family_max_level"] + 1):
                    if Dp + n > Dmax + cfg["family_Delta_margin"]:
                        break
                    spans[n] = {}
                    for Lp in blks:
                        imgs = []
                        for L in (Lp - 1, Lp, Lp + 1):
                            if L in spans[n - 1] and (L, Lp) in R:
                                Vb = bas[L].to_block(spans[n - 1][L], len(blks[L]["E"]))
                                imgs.append(R[(L, Lp)] @ Vb)
                        mult = multiplicity(n, L0, Lp)
                        B, sv, nxt = _orth_top(np.hstack(imgs), mult) if imgs else (None, None, None)
                        if B is None:
                            continue
                        bs = bas[Lp]
                        Pexp = key[2] * (-1) ** (L0 + n + Lp)   # P_mu flips eta = P(-1)^L
                        ins = (np.abs(bs.C2 - key[0]) < 1e-5) & (bs.X == key[1]) & (bs.P == Pexp)
                        leak = float(np.max(np.sum(B[~ins, :] ** 2, axis=0), initial=0.0))
                        gates["max_sector_leak"] = max(gates["max_sector_leak"], leak)
                        spans[n][Lp] = B
                        Dbar = sd.delta(np.einsum("ij,i,ij->j", B, bs.E, B))
                        fam_rows.append(dict(N=sd.N, block=f"s{a}_s{b}", parent_pid=m["pid"], parent_L=L0, level=n, L=Lp,
                                             C2=key[0], X=key[1], P=int(Pexp), rank=B.shape[1], expected_rank=mult,
                                             min_singular=float(sv[-1]), next_singular=nxt,
                                             rank_ambiguous=bool(nxt > 0.5 * float(sv[-1])), sector_leak=leak,
                                             Delta_parent=Dp, Delta_mean=float(np.mean(Dbar)),
                                             Delta_spread=float(np.max(Dbar) - np.min(Dbar)),
                                             level_deviation=float(np.mean(Dbar) - Dp - n)))
                        famB[Lp].append((m["pid"], n, B))
                        if n == 1:
                            lvl1.append(float(np.mean(Dbar) - Dp - 1))
                m["level1_deviation"] = float(np.max(np.abs(lvl1))) if lvl1 else np.nan
                prim_rows.append(dict(N=sd.N, pid=m["pid"], block=f"s{a}_s{b}", L=L0, C2=key[0], rep=rep_name(key[0]),
                                      X=key[1], P=key[2], Delta=Dp, sigma_Delta=m["sigma_Delta"], K2=m["K2"],
                                      kappa=m["kappa"], dominant_index=m["dominant_index"] + 1,
                                      dominant_weight=m["dominant_weight"], Pnorm_ratio=m["Pnorm_ratio"],
                                      level1_deviation=m["level1_deviation"], in_window=bool(Dp <= Dmax),
                                      stable_kappa_x3=bool(m["in_lo"] and m["in_hi"])))
                pinfo[m["pid"]] = (Dp, L0); prow[m["pid"]] = prim_rows[-1]
        if opt:
            for r_ in prow.values():
                r_.update(conflict_pair=False, conflict_weight=0.0, conflict_id=None)
        # ---- raw-level classification over the basis of each (pair, L)
        for L, blk in blks.items():
            bs = bas[L]; Dl = sd.delta(bs.E)
            guarded_blk = set()
            for ch in blk["channels"]:
                guarded_blk |= set(ch["guarded"].tolist())
            allD = _orth(np.hstack([B for _, _, B in famB[L]])) if famB[L] else None
            ds = descendant_span_v1(sd, a, b, L, bs, blk, prim[L], famB[L], pinfo, prow, Qs["mid"][L], allD, opt, rule,
                                    wcut, Dmax, gates) if opt else None
            if extra is not None:
                extra[(a, b, L)] = dict(basis=bs, blk=blk, Q={t: Qs[t][L] for t in Qs}, allD=ds["D"] if ds else allD,
                                        allD_v035=allD, fam=famB[L], guarded_blk=guarded_blk)
            for r in range(bs.n):
                if Dl[r] > Dmax:
                    continue
                key = (int(round(bs.C2[r])), int(bs.X[r]), int(bs.P[r]))
                def wsub(tag):
                    F = Qs[tag][L].get(key)
                    return float(np.sum(F[r, :] ** 2)) if F is not None and F.size else 0.0
                wP, wPlo, wPhi = wsub("mid"), wsub("lo"), wsub("hi")
                wD = float(np.sum(allD[r, :] ** 2)) if allD is not None else 0.0
                best = (None, None, 0.0)
                for (p, n, B) in famB[L]:
                    w = float(np.sum(B[r, :] ** 2))
                    if w > best[2]:
                        best = (p, n, w)
                ib = bs.to_blk[r]
                g = bool(ib >= 0 and ib in guarded_blk)
                # classes (convention v1):
                #   primary                      guarded KK source, weight in primary subspace >= wcut
                #   descendant                   weight in P-image span of identified primaries >= wcut
                #   nonprimary_no_family_coverage       guarded, primary weight <= 1-wcut, descendant weight <= 0.3
                #   nonprimary_partial_family_coverage  guarded, primary weight <= 1-wcut, 0.3 < descendant weight < wcut
                #   mixed_primary_descendant     finite-N admixture of a primary mode and descendants
                #   unclassifiable_*             not a guarded KK source and not inside a descendant span
                if g and wP >= wcut:
                    cls = "primary"
                elif wD >= wcut:
                    cls = "descendant"
                elif not g:
                    cls = "unclassifiable_not_KK_source" if ib < 0 else "unclassifiable_unguarded"
                elif wP <= 1 - wcut:
                    # non-primary by KK (K image nonzero) but not covered by any family generated in the window;
                    # possible causes (not decided here): parent with L >= 5, parent above the window, family
                    # truncation by the saved windows, finite-N spreading of descendant states
                    cls = "nonprimary_no_family_coverage" if wD <= 0.3 else "nonprimary_partial_family_coverage"
                else:
                    cls = "mixed_primary_descendant"
                row = dict(N=sd.N, block=f"s{a}_s{b}", L=L, basis="extended_target" if bs.extended else "block",
                           basis_index=r + 1, block_index=(int(ib) + 1) if ib >= 0 else None,
                           energy=float(bs.E[r]), Delta=float(Dl[r]), C2=key[0], rep=rep_name(key[0]),
                           X=key[1], P=key[2], guarded=g, w_primary=wP, w_primary_thr_lo=wPlo,
                           w_primary_thr_hi=wPhi, w_descendant=wD, cls=cls, parent_pid=best[0],
                           parent_level=best[1], parent_weight=best[2])
                if ds:
                    row.update(ds_raw_row(ds, key, r, g, ib, wP, wD, cls, wcut))
                raw_rows.append(row)
    return prim_rows, raw_rows, fam_rows, gates, mode_vecs


# ------------------------------------------------------------------ v0.5.0-dev: descendant_span_v1 (S0-S2, S3b)
def descendant_span_v1(sd, a, b, L, bs, blk, primL, famL, pinfo, prow, QmidL, allD, opt, rule, g, Dmax, gates):
    """Descendant span and P/D conflicts of one (pair, L) under option opt (B official; A, S1prime comparators)."""
    N = sd.N; blkname = f"s{a}_s{b}"
    keys = sorted(set(zip(np.round(bs.C2).astype(int), bs.X.astype(int), bs.P.astype(int))))
    rows = {k: np.where((np.abs(bs.C2 - k[0]) < 1e-5) & (bs.X == k[1]) & (bs.P == k[2]))[0] for k in keys}
    # R0 (harden_v1 T1 span_triage): d_P (modes with Delta <= Delta_max) + effective d_D (sv > 0.5) > n_s
    oc = {}
    for k in keys:
        dP = sum(1 for m, kk in primL if kk == k and m["Delta"] <= Dmax)
        dD = int(np.sum(np.linalg.svd(allD[rows[k], :], compute_uv=False) > 0.5)) if allD is not None and rows[k].size else 0
        oc[k] = bool(dP + dD > rows[k].size)
    # S1 / S1'
    H = np.hstack([B for _, _, B in famL]) if famL else None
    tags = [(p, n) for p, n, B in famL for _ in range(B.shape[1])]
    sector_of = lambda u: max(keys, key=lambda k: float(np.sum(u[rows[k]] ** 2))) if keys else None
    if opt == "S1prime":
        order = dsp.family_order([dict(level=n, parent_Delta=pinfo[p][0], pid=p) for p, n, _ in famL])
        cols, ctag = [], []
        for k in order:
            p, n, B = famL[k]
            for c in range(B.shape[1]):
                cols.append(B[:, c]); ctag.append((p, n))
        D1, rej = dsp.span_greedy(cols, g) if cols else (None, [])
        for j, r2 in rej:
            sk = sector_of(cols[j])
            gates["ds_dropped"].append(dict(N=N, block=blkname, L=L, C2=sk[0], X=sk[1], P=sk[2], rule="S1prime_greedy",
                                            s2_or_residual2=r2, parent_pid=ctag[j][0], level=ctag[j][1], loadings=""))
    else:
        D1, _, dropped = dsp.span_svd(H, g) if H is not None else (None, None, [])
        for i, s2, v, u in dropped:
            sk = sector_of(u); lo = {}
            for t, x in zip(tags, v):
                lo[t] = lo.get(t, 0.0) + float(x * x)
            top = sorted(lo.items(), key=lambda kv: -kv[1])[:3]
            gates["ds_dropped"].append(dict(N=N, block=blkname, L=L, C2=sk[0], X=sk[1], P=sk[2], rule="S1_svd",
                                            s2_or_residual2=s2, parent_pid=top[0][0][0], level=top[0][0][1],
                                            loadings=";".join(f"pid{p}/n{n}:{x:.3f}" for (p, n), x in top),
                                            sector_overcomplete=oc[sk]))
    # S2 conflicts, sector by sector (option A: overcomplete sectors only)
    Dcur, info = D1, {}
    band, lmin = rule["parameters"]["lambda_band"], rule["parameters"]["almost_primary_lambda_max"]
    nL = len(blk["E"])
    for k in keys:
        if opt == "A" and not oc[k]:
            continue
        Q = QmidL.get(k)
        cs = dsp.conflict_split(Q if Q is not None and Q.size else None, Dcur, g)
        Dcur = cs["D_rest"]
        info[k] = dict(Qc=cs["Q_clean"], p=cs["p"], d=cs["d"], cids=[], C=None, Cs=[])
        if cs["p"] is None:
            continue
        ch = next((c for c in blk["channels"] if c["C2"] == k[0] and "K2" in c), None)
        for i in range(cs["p"].shape[1]):
            cid = f"N{N}|{blkname}|L{L}|C{k[0]}|X{k[1]:+d}|P{k[2]:+d}|{i}"
            d = cs["d"][:, i]; db = bs.to_block(d, nL)
            if ch is not None and ch["guarded"].size:
                dg = db[ch["guarded"]]; gw = float(dg @ dg); K2d = float(dg @ ch["K2"] @ dg)
            else:
                gw, K2d = 0.0, np.nan
            fw = [(float(np.sum((B.T @ d) ** 2)), p, n) for p, n, B in famL]
            fwb = max(fw) if fw else (0.0, None, None)
            if fwb[1] is not None:
                D0, L0 = pinfo[fwb[1]]; lam = dsp.lam_desc(D0, fwb[2], L, L0)
            else:
                D0 = L0 = lam = np.nan
            ev = dsp.evidence_b(gw, K2d, lam, band, lmin)
            info[k]["cids"].append(cid); info[k]["Cs"].append(dsp.orth_cols(cs["p"][:, i:i + 1], d[:, None]))
            gates["ds_conflicts"].append(dict(conflict_id=cid, N=N, block=blkname, L=L, C2=k[0], rep=rep_name(k[0]), X=k[1], P=k[2],
                                              cos2=float(cs["cos2"][i]), sector_dim=int(rows[k].size),
                                              sector_overcomplete=oc[k], d_guarded_weight=gw, K2_d=K2d,
                                              parent_pid=fwb[1], parent_level=fwb[2], parent_family_weight=fwb[0],
                                              parent_Delta=D0, parent_L=L0, lambda_pred=lam, evidence_b=ev))
        info[k]["C"] = dsp.orth_cols(cs["p"], cs["d"])
        # conflict_pair modes: weight on span{p_i} >= 1 - g
        for m, kk in primL:
            if kk != k:
                continue
            vb = bs.from_block(m["full"])
            pw = (cs["p"].T @ vb) ** 2
            prow[m["pid"]].update(conflict_weight=float(pw.sum()), conflict_pair=bool(pw.sum() >= 1 - g),
                                  conflict_id=info[k]["cids"][int(np.argmax(pw))])
    return dict(opt=opt, D1=D1, D=Dcur, oc=oc, info=info, nconf={k: len(v["cids"]) for k, v in info.items()})


def ds_raw_row(ds, key, r, guarded, ib, wP, wD035, cls035, g):
    """S2 class of one raw state; returns the extra columns (cls overwritten when the rule applies)."""
    applied = ds["opt"] != "A" or ds["oc"].get(key, False)
    out = dict(cls_v035=cls035, w_descendant_v035=wD035, sector_overcomplete=ds["oc"].get(key, False), ds_applied=applied,
               sector_n_conflicts=ds["nconf"].get(key, 0), conflict_id=None, w_conflict=0.0)
    if not applied:
        out.update(w_primary_clean=wP, w_descendant_S1=wD035)
        return out
    inf = ds["info"].get(key)
    Qc = inf["Qc"] if inf else None
    wQc = dsp.w(Qc, r) if inf else wP
    wD1 = dsp.w(ds["D1"], r); wDp = dsp.w(ds["D"], r)
    cls = dsp.classify_s2(guarded, wP, wQc, wD1, wDp, g)
    if cls is None:                      # v0.3.5 fall-through with w_P = w(Q), w_D = w(D')
        if not guarded:
            cls = "unclassifiable_not_KK_source" if ib < 0 else "unclassifiable_unguarded"
        elif wP <= 1 - g:
            cls = "nonprimary_no_family_coverage" if wDp <= 0.3 else "nonprimary_partial_family_coverage"
        else:
            cls = "mixed_primary_descendant"
    if inf and inf["cids"]:
        cw = [dsp.w(C, r) for C in inf["Cs"]]
        j = int(np.argmax(cw)); out.update(conflict_id=inf["cids"][j], w_conflict=dsp.w(inf["C"], r))
    out.update(cls=cls, w_descendant=wDp, w_descendant_S1=wD1, w_primary_clean=wQc)
    return out
