"""Post-freeze hardening layer (v0.4.0).  python -m blindpd.harden <harden_config.json> <out_dir>

Reads a frozen v0.3.5 baseline (config + run directory), recomputes the integer-sector analysis with the same
inputs and thresholds (regression gate: per-copy classes identical), and adds the checks T1-T7 of
the harden_v1 rules (config/harden_v1.json).  Nothing of the baseline is rewritten: original labels are kept and the
revised labels, evidence and reasons are written to new files.  No published table is read.

  T1 dual pass P/D, cross-copy and cross-sector conflicts, primary/descendant span overlap triage
  T2 per-state scalar algebra residual  sum_A ||P_A psi||^2 - ||K_A psi||^2 - 6 Delta  (informational)
  T3 stability under shrinking every source/target window by dDelta
  T4 kappa: local gaps per sector, histogram-rule variants, vector-level mode correspondence, absolute K2 cut-offs
  T5 per-track aggregation of the diagnostics across N
  T6 near-short descendants (FFK 2602.04958 eq. 3.2) and a normalisation check on descendant levels
  T7 revised_class with reasons, mode statuses and OPE identity status
"""
import json, os, sys, platform, datetime
import numpy as np
import pandas as pd
from . import __version__
from .io import SizeData, sha256_file, BLOCK_TARGET_STATUS
from .kk import block_kk, primary_modes, required_lf
from .spectrum import analyse_size, multiplicity
from .so5 import rep_name


# ------------------------------------------------------------------------------------------------ helpers
def lam_desc(D0, n, L, L0):
    """|K|^2 eigenvalue of a level-n, spin-L descendant of a (D0, L0) primary (exact 3d CFT, FFK eq. 3.2)."""
    return (2 * D0 + n - 3) * n + L * (L + 1) - L0 * (L0 + 1)


def dual_class(guarded, wP, wD, gate):
    pP = bool(guarded and wP >= gate); pD = bool(wD >= gate)
    return pP, pD, (pP and pD)


def valley_threshold_offset(kappas, search_range, width, offset):
    """thresholds.valley_threshold with the bin grid shifted by offset*width (offset 0 reproduces it exactly)."""
    k = np.asarray(kappas); k = k[k > 0]; lk = np.log10(k)
    edges = np.arange(-12.0 + offset * width, 1.0 + 1e-9, width)
    h, _ = np.histogram(lk, edges)
    lo, hi = np.log10(search_range[0]), np.log10(search_range[1])
    cand = [(h[i], edges[i]) for i in range(len(h)) if edges[i] >= lo - 1e-9 and edges[i + 1] <= hi + 1e-9]
    cmin = min(c for c, _ in cand)
    left = min(e for c, e in cand if c == cmin)
    return float(10 ** (left + width / 2)), int(cmin)


def principal_cos(A, B):
    """Cosines of the principal angles between span(A) and span(B) (columns need not be orthonormal)."""
    if A is None or B is None or A.size == 0 or B.size == 0:
        return np.array([])
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)


class ShrunkSizeData:
    """SizeData view in which every source window and every target window loses its top dDelta (T3)."""

    def __init__(self, sd, dDelta):
        self._sd, self.N, self.alpha, self.E0, self.coeff, self.meta = sd, sd.N, sd.alpha, sd.E0, sd.coeff, sd.meta
        dE = dDelta / sd.alpha; self._cache = {}
        self.blocks, self.parity, keepb = {}, {}, {}
        for k, b in sd.blocks.items():
            E = np.array(b["energy"]); cut = E.max() - dE; keep = E < cut
            if not keep.any():                     # the whole window lies inside the removed top slice
                continue
            nb = dict(b)
            for f, v in b.items():
                if isinstance(v, list) and len(v) == len(E):
                    nb[f] = [x for x, kp in zip(v, keep) if kp]
            self.blocks[k] = nb; keepb[k] = (keep, cut)
            if k in sd.parity:
                self.parity[k] = sd.parity[k][keep]
        self.edges, self._sl, self.ext_cut = {}, {}, {}
        for key, e in sd.edges.items():
            a, b, li, lf = key; sk_, tk_ = f"s{a}_s{b}_L{li}", f"s{a}_s{b}_L{lf}"
            if sk_ not in keepb:
                continue
            sk, _ = keepb[sk_]
            tE = np.array(e["target_E"])
            if e["target_status"] in BLOCK_TARGET_STATUS:
                if tk_ not in keepb:
                    continue
                tk, tcut = keepb[tk_]
                if len(tk) != len(tE):
                    raise ValueError(f"block-target edge {key} does not span its block window")
            else:
                tcut = tE.max() - dE; tk = tE < tcut; self.ext_cut[(a, b, lf)] = tcut
                if not tk.any():
                    continue
            fr = e["target_frontier"]; frn = float(tcut if fr is None else min(fr, tcut))
            ne = dict(e)
            for f in ("source_E", "source_C2", "source_parity"):
                if isinstance(e.get(f), list):
                    ne[f] = list(np.array(e[f])[sk])
            for f in ("target_E", "target_C2"):
                ne[f] = list(np.array(e[f])[tk])
            ne["target_frontier"] = frn; ne["nt"], ne["ns"] = int(tk.sum()), int(sk.sum())
            self.edges[key] = ne; self._sl[key] = (tk, sk)

    def delta(self, E):
        return self.alpha * (np.asarray(E) - self.E0)

    def edge_arrays(self, a, b, li, lf):
        k = (a, b, li, lf)
        if k not in self._cache:
            o = self._sd.edge_arrays(a, b, li, lf); tk, sk = self._sl[k]
            self._cache[k] = dict(o, M=o["M"][np.ix_(tk, sk)], full=o["full"][sk], sE=o["sE"][sk], tE=o["tE"][tk],
                                  sC2=o["sC2"][sk], tC2=o["tC2"][tk], frontier=self.edges[k]["target_frontier"])
        return self._cache[k]

    def shrink_parity_df(self, par):
        keep = np.ones(len(par), bool)
        for i, r in enumerate(par.itertuples()):
            if r.window == "target":
                cut = self.ext_cut.get((int(r.sA), int(r.sB), int(r.L)))
                if cut is not None and r.energy >= cut:
                    keep[i] = False
        return par[keep]


def level_key(N, c, l, p, e):
    return f"N{N}:C{c}:L{l}:P{int(p):+d}:{e:.7f}"


def primary_regression_ok(pr, b0, tol):
    """v0.4.1 (review R5): recomputed primary modes pr vs the baseline rows b0 of one N.  The pid sets are compared
    (unique, equal) before reindexing; v0.4.0 reindexed first, which cut a missing mode from the baseline as well."""
    pids_r = list(pr.pid) if len(pr) else []; pids_b = list(b0.pid)
    if len(pids_r) != len(set(pids_r)) or len(pids_b) != len(set(pids_b)) or set(pids_r) != set(pids_b):
        return False
    if not pids_r:
        return True
    bp = b0.set_index("pid").reindex(pids_r)
    return bool(np.max(np.abs(pr.Delta.values - bp.Delta.values)) <= tol)


def shrunk_sector_states(ex):
    """v0.4.1 (review R1): per (block, L, C2, X, P) of a shrunk variant, 'guarded_sector_present' when the KK block
    still has a guarded source in that sector, 'sources_unguarded' when it has sources but none guarded (coverage
    loss); sectors not listed are absent.  Presence no longer depends on a primary mode being selected."""
    ks = {}
    for (a, b, L), e in ex.items():
        blk = e["blk"]
        for ch in blk["channels"]:
            for sec in ch["sectors"]:
                ks[(f"s{a}_s{b}", L, ch["C2"], sec["X"], sec["P"])] = "guarded_sector_present"
            for i in ch["excluded"]:
                ks.setdefault((f"s{a}_s{b}", L, ch["C2"], int(blk["X"][i]), int(blk["P"][i])), "sources_unguarded")
    return ks


def t3_mode_check(m, cand, sstate, gate):
    """T3 mode test in one shrunk variant: best overlap with the variant's selected modes of the same sector.
    A surviving sector without a selected mode fails with overlap 0 (v0.4.0 counted it as an absent sector)."""
    best, dD = 0.0, np.nan
    for m2 in cand:
        n2 = len(m2["vec"]); o = float(np.dot(m["vec"][:n2], m2["vec"]) ** 2)
        if o > best:
            best, dD = o, abs(m2["Delta"] - m["Delta"])
    present = sstate == "guarded_sector_present"
    return dict(sector_state=sstate, sector_present=present, n_selected_variant=len(cand), best_overlap=best,
                dDelta_shift=dD, stable=bool(present and cand and best >= gate and dD < 0.01))


WINDOW_ORDER = ["window_unstable", "window_stable", "sector_absent", "not_tested"]


def worst_window_status(s):
    """v0.4.1 (review R2): worst T3 status over the copies of one phys_id."""
    return min(s, key=lambda x: WINDOW_ORDER.index(x) if x in WINDOW_ORDER else len(WINDOW_ORDER))


# ------------------------------------------------------------------------------------------------ main
def main(hcfg_path, out):
    hc = json.load(open(hcfg_path, encoding="utf-8"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"refusing to write into non-empty {out}")
    os.makedirs(out, exist_ok=True)
    bcfg_path = hc["baseline"]["config"]; brun = hc["baseline"]["run"]
    cfg = json.load(open(bcfg_path, encoding="utf-8"))
    base = os.path.dirname(os.path.dirname(os.path.abspath(bcfg_path)))
    P = lambda p: os.path.normpath(os.path.join(base, p))
    man = json.load(open(os.path.join(brun, "MANIFEST.json"), encoding="utf-8"))
    # ---- provenance gate: baseline outputs and the KK / physics inputs as recorded by the baseline run
    bad_out = [f for f, h in man["outputs"].items() if sha256_file(os.path.join(brun, f)) != h]
    if bad_out:
        sys.exit(f"baseline outputs changed since their MANIFEST: {bad_out[:5]}")
    roots = [os.path.join(P(cfg["kk_export"]), f"N{N}") for N in cfg["sizes"]] + \
            [os.path.join(P(cfg["physics"]), f"N{N}") for N in cfg["sizes"]]
    checked = 0; bad_in = []
    for rel, h in man["input_files_sha256"].items():
        p = os.path.normpath(os.path.join(base, rel))
        if any(os.path.commonpath([p, r]) == r for r in roots):
            checked += 1
            if sha256_file(p) != h:
                bad_in.append(rel)
    if bad_in or not checked:
        sys.exit(f"baseline KK/physics inputs differ from its MANIFEST ({checked} checked): {bad_in[:5]}")
    gates0 = json.load(open(os.path.join(brun, "gates.json"), encoding="utf-8"))
    thr = float(gates0["kappa_threshold"]); gate = float(cfg["raw_class_weight"]); Dmax = cfg["Delta_max"]
    sizes = cfg["sizes"]
    raw0 = pd.read_csv(os.path.join(brun, "raw_levels_all_copies.csv"))
    rawp0 = pd.read_csv(os.path.join(brun, "raw_levels_classified.csv"))
    prim0 = pd.read_csv(os.path.join(brun, "primary_modes_all_copies.csv"))
    rep0 = pd.read_csv(os.path.join(brun, "primaries_per_N.csv"))
    trk0 = pd.read_csv(os.path.join(brun, "primary_tracks.csv"))
    fam0 = pd.read_csv(os.path.join(brun, "family_spans.csv"))
    xsec0 = pd.read_csv(os.path.join(brun, "cross_sector_classification.csv"))
    ope0 = pd.read_csv(os.path.join(brun, "ope_rows_integer_exact.csv"), low_memory=False)
    R1 = hc["T1_dual_pass"]; R3 = hc["T3_window_shrink"]; R4 = hc["T4_kappa"]; R6 = hc["T6_near_short"]

    sds, pars, EX, RAW, PRIM, MV = {}, {}, {}, {}, {}, {}
    regress = {}
    for N in sizes:
        sds[N] = SizeData(os.path.join(P(cfg["kk_export"]), f"N{N}"), os.path.join(P(cfg["physics"]), f"N{N}"), N)
        pars[N] = pd.read_csv(os.path.join(P(cfg["physics"]), f"N{N}", "parity.csv"))
        EX[N] = {}
        pr, rw, fm, g, mv = analyse_size(sds[N], cfg, thr, pars[N], EX[N])
        rw = pd.DataFrame(rw); pr = pd.DataFrame(pr)
        b = raw0[raw0.N == N].reset_index(drop=True)
        # the baseline sorts primary modes by (N, C2, L, P, Delta) before writing: align on pid
        ok = (len(rw) == len(b) and (rw.cls.values == b.cls.values).all() and
              np.max(np.abs(rw.w_primary.values - b.w_primary.values), initial=0) <= hc["regression_gate"]["weight_abs_tol"] and
              np.max(np.abs(rw.w_descendant.values - b.w_descendant.values), initial=0) <= hc["regression_gate"]["weight_abs_tol"] and
              primary_regression_ok(pr, prim0[prim0.N == N], hc["regression_gate"]["Delta_abs_tol"]))
        regress[N] = dict(raw_rows=len(rw), primary_modes=len(pr), identical_classes=bool(ok))
        if not ok:
            sys.exit(f"regression gate failed at N{N}: recomputed classes/weights differ from the baseline")
        rw["phys_level_key"] = [level_key(N, c, l, p, e) for c, l, p, e in zip(rw.C2, rw.L, rw.P, rw.energy)]
        RAW[N] = rw; PRIM[N] = pr; MV[N] = {m["pid"]: m for m in mv if m is not None}
        print(f"N{N}: regression ok ({len(rw)} raw rows, {len(pr)} modes)", flush=True)

    # ============================================================================ T1
    t1_rows, span_rows, mode_rows = [], [], []
    for N in sizes:
        rw = RAW[N]
        pp = [dual_class(g_, a_, b_, gate) for g_, a_, b_ in zip(rw.guarded, rw.w_primary, rw.w_descendant)]
        rw["pass_P"] = [x[0] for x in pp]; rw["pass_D"] = [x[1] for x in pp]; rw["dual_pass"] = [x[2] for x in pp]
        for (a, b, L), ex in EX[N].items():
            bs, allD = ex["basis"], ex["allD"]
            keys = sorted(set(zip(np.round(bs.C2).astype(int), bs.X.astype(int), bs.P.astype(int))))
            # primary subspace of the analysis window: projected modes with Delta <= Delta_max of this sector
            # (run-1 used every kappa-selected mode up to the window top, which inflated near_overlap; see CHANGES)
            mv_sec = {}
            for m in MV[N].values():
                if m["block"] == f"s{a}_s{b}" and m["L"] == L and m["Delta"] <= Dmax:
                    mv_sec.setdefault((m["C2"], m["X"], m["P"]), []).append(bs.from_block(m["vec"][:, None])[:, 0])
            for key in keys:
                rows = np.where((np.abs(bs.C2 - key[0]) < 1e-5) & (bs.X == key[1]) & (bs.P == key[2]))[0]
                Q = np.array(mv_sec[key]).T if key in mv_sec else None
                dP = 0 if Q is None else Q.shape[1]
                if allD is not None and rows.size:
                    # descendant directions inside this sector: left singular vectors of the restricted span with
                    # sv > 0.5 (plain QR of allD[rows] would promote ~1e-16 cross-sector leakage to unit vectors)
                    U, sv, _ = np.linalg.svd(allD[rows, :], full_matrices=False)
                    keep = sv > R1["span_triage"]["effective_descendant_dim_sv"]; dD = int(keep.sum())
                    cos = principal_cos(Q[rows, :], U[:, keep]) if dP and dD else np.array([])
                else:
                    dD, cos = 0, np.array([])
                c2max = float(np.max(cos) ** 2) if cos.size else 0.0
                tri = "overcomplete" if dP + dD > rows.size else ("near_overlap" if c2max >= gate else "none")
                span_rows.append(dict(N=N, block=f"s{a}_s{b}", L=L, C2=key[0], rep=rep_name(key[0]), X=key[1], P=key[2],
                                      sector_dim=int(rows.size), primary_dim=dP, descendant_dim_eff=dD,
                                      max_cos2_primary_descendant=c2max,
                                      n_cos2_ge_gate=int(np.sum(cos ** 2 >= gate)) if cos.size else 0, triage=tri))
        # projected primary modes
        for pid, m in MV[N].items():
            ex = EX[N][(int(m["block"][1]), int(m["block"][4]), m["L"])]
            vb = ex["basis"].from_block(m["vec"][:, None])[:, 0]
            wDm = float(np.sum((ex["allD"].T @ vb) ** 2)) if ex["allD"] is not None else 0.0
            mode_rows.append(dict(N=N, pid=pid, block=m["block"], L=m["L"], C2=m["C2"], X=m["X"], P=m["P"],
                                  Delta=m["Delta"], w_descendant_of_mode=wDm, mode_in_descendant_span=bool(wDm >= gate)))
    span = pd.DataFrame(span_rows); modes = pd.DataFrame(mode_rows)
    rawall = pd.concat([RAW[N] for N in sizes], ignore_index=True)
    # copy-level aggregation per physical level
    # cross-copy split: some copy passes P only and another copy passes D only
    rawall["only_P"] = rawall.pass_P & ~rawall.pass_D; rawall["only_D"] = rawall.pass_D & ~rawall.pass_P
    agg = rawall.groupby("phys_level_key").agg(any_dual=("dual_pass", "any"), any_only_P=("only_P", "any"),
                                               any_only_D=("only_D", "any"), n_copies=("cls", "size"))
    agg["cross_copy_split"] = agg.any_only_P & agg.any_only_D
    # cross-sector P vs D conflicts (stageB vs integer)
    xs = xsec0[[{a, b} == {"primary", "descendant"} for a, b in zip(xsec0.cls_stageB, xsec0.cls_integer)]]
    xs_keys = set()
    for r in xs.itertuples():
        g = rawp0[(rawp0.N == r.N) & (rawp0.L == r.L) & (rawp0.rep.astype(str) == str(r.rep)) & (rawp0.P == r.P) &
                  (np.abs(rawp0.Delta - r.Delta) < 1e-6)]
        xs_keys |= set(g.phys_level_key)

    # ============================================================================ T2 algebra
    alg_rows, alg_modes = [], []
    for N in sizes:
        sd = sds[N]
        for (a, b, L), ex in EX[N].items():
            bs = ex["basis"]; Eb = np.array(sd.blocks[f"s{a}_s{b}_L{L}"]["energy"])
            req = required_lf(L)
            ea = {lf: sd.edge_arrays(a, b, L, lf) for lf in (L - 1, L, L + 1) if (a, b, L, lf) in sd.edges}
            missing = [lf for lf in req if lf not in ea]
            for r in range(bs.n):
                ib = int(bs.to_blk[r]); D = float(sd.delta(bs.E[r]))
                if ib < 0 or D > Dmax:
                    continue
                Kt = Pt = Pw = mid_t = 0.0; ks = []; ps = []; guard = not missing
                for lf, e in ea.items():
                    c = (2 * lf + 1) / (2 * L + 1); m = e["M"][:, ib]; sE = e["sE"][ib]
                    low = e["tE"] < sE - e["atol"]; high = e["tE"] > sE + e["atol"]
                    K = float(np.sum(m[low] ** 2)); mid = float(np.sum(m[~low & ~high] ** 2)); pw = float(np.sum(m[high] ** 2))
                    Pf = float(e["full"][ib]) - K - mid
                    Kt += c * K; Pt += c * Pf; Pw += c * pw; mid_t += c * mid
                    guard &= bool(sE < e["frontier"] - e["atol"])
                    if K > 0:
                        ks.append((sd.alpha * (sE - e["tE"][low]), c * m[low] ** 2))
                    if pw > 0:
                        ps.append((sd.alpha * (e["tE"][high] - sE), c * m[high] ** 2))
                comm = Pt - Kt
                wm = lambda lst: (float(np.sum(np.concatenate([x * w for x, w in lst])) / np.sum(np.concatenate([w for _, w in lst])))
                                  if lst else np.nan)
                alg_rows.append(dict(N=N, block=f"s{a}_s{b}", L=L, basis_index=r + 1, block_index=ib + 1, energy=float(bs.E[r]),
                                     Delta=D, C2=int(round(bs.C2[r])), P=int(bs.P[r]),
                                     phys_level_key=level_key(N, int(round(bs.C2[r])), L, int(bs.P[r]), float(bs.E[r])),
                                     K_sum=Kt, P_sum=Pt, P_in_window=Pw, equal_energy_part=mid_t, comm=comm, six_Delta=6 * D,
                                     residual=comm - 6 * D, rel_residual=(comm - 6 * D) / (6 * D) if D > 0.1 else np.nan,
                                     P_window_fraction=Pw / Pt if Pt > 0 else np.nan,
                                     mean_K_step=wm(ks), mean_P_step_window=wm(ps),
                                     coverage=("missing_edges" if missing else ("K_complete" if guard else "K_truncated"))))
        for pid, m in MV[N].items():
            a, b, L = int(m["block"][1]), int(m["block"][4]), m["L"]; v = m["vec"]
            req = required_lf(L); Kt = Pw = Pmax = 0.0; miss = False
            for lf in (L - 1, L, L + 1):
                if (a, b, L, lf) not in sd.edges:
                    miss |= lf in req; continue
                e = sd.edge_arrays(a, b, L, lf); c = (2 * lf + 1) / (2 * L + 1)
                Km = e["M"] * (e["tE"][:, None] < e["sE"][None, :] - e["atol"])
                Pm = e["M"] * (e["tE"][:, None] > e["sE"][None, :] + e["atol"])
                d = np.sqrt(np.maximum(e["full"] - np.sum(e["M"] ** 2, axis=0), 0.0))
                pk = float(np.sum((Km @ v) ** 2)); pw = float(np.sum((Pm @ v) ** 2)); out_b = float(np.sum(np.abs(v) * d))
                Kt += c * pk; Pw += c * pw; Pmax += c * (np.sqrt(pw) + out_b) ** 2
            D = m["Delta"]
            alg_modes.append(dict(N=N, pid=pid, block=m["block"], L=L, C2=m["C2"], P=m["P"], Delta=D, K_sum=Kt,
                                  comm_lower=Pw - Kt, comm_upper=Pmax - Kt, six_Delta=6 * D,
                                  six_Delta_inside_interval=bool(Pw - Kt <= 6 * D <= Pmax - Kt),
                                  rel_dev_lower=(Pw - Kt - 6 * D) / (6 * D) if D > 0.1 else np.nan,
                                  coverage="missing_edges" if miss else "window_plus_bound"))
    alg = pd.DataFrame(alg_rows); algm = pd.DataFrame(alg_modes)

    # ============================================================================ T3 window shrink
    t3_raw, t3_mode = [], []
    for dd in R3["dDelta"]:
        for N in sizes:
            ssd = ShrunkSizeData(sds[N], dd); ex2 = {}
            try:
                pr2, rw2, _, _, mv2 = analyse_size(ssd, cfg, thr, ssd.shrink_parity_df(pars[N]), ex2)
            except ValueError as err:
                t3_raw.append(dict(N=N, dDelta=dd, phys_level_key=None, status="variant_failed", note=str(err)[:120])); continue
            rw2 = pd.DataFrame(rw2)
            k2 = {(blk, l, lk): c for blk, l, lk, c in zip(rw2.block, rw2.L, [level_key(N, c, l, p, e) for c, l, p, e in
                                                                              zip(rw2.C2, rw2.L, rw2.P, rw2.energy)], rw2.cls)}
            for r in RAW[N].itertuples():
                c2 = k2.get((r.block, r.L, r.phys_level_key))
                t3_raw.append(dict(N=N, dDelta=dd, phys_level_key=r.phys_level_key, block=r.block, L=r.L,
                                   cls=r.cls, cls_variant=c2, status=("outside_shrunk_domain" if c2 is None else
                                                                      ("same" if c2 == r.cls else "changed"))))
            mvv = {}
            for m2 in mv2:
                if m2 is not None:
                    mvv.setdefault((m2["block"], m2["L"], m2["C2"], m2["X"], m2["P"]), []).append(m2)
            ksec = shrunk_sector_states(ex2)
            for pid, m in MV[N].items():
                skey = (m["block"], m["L"], m["C2"], m["X"], m["P"])
                t3_mode.append(dict(N=N, dDelta=dd, pid=pid, Delta=m["Delta"],
                                    **t3_mode_check(m, mvv.get(skey, []), ksec.get(skey, "sector_absent"), gate)))
            print(f"T3 dDelta={dd} N{N} done", flush=True)
    t3r = pd.DataFrame(t3_raw); t3m = pd.DataFrame(t3_mode)
    t3r_ok = t3r[t3r.phys_level_key.notna()]
    win_status = t3r_ok.groupby("phys_level_key").status.agg(
        lambda s: "window_unstable" if (s == "changed").any() else ("window_stable" if (s == "same").any() else "outside_shrunk_domain"))
    mode_win = t3m.groupby(["N", "pid"]).apply(
        lambda g: "window_unstable" if (g.sector_present & ~g.stable).any() else ("window_stable" if g.sector_present.any() else "sector_absent"),
        include_groups=False)

    # ============================================================================ T4 kappa
    gap_rows, abs_rows, kcorr = [], [], []
    pool = {tuple(p): [] for p in R4["pools"]}
    for N in sizes:
        for (a, b, L), ex in EX[N].items():
            if L > cfg["L_max"]:
                continue
            for ch in ex["blk"]["channels"]:
                for sec in ch["sectors"]:
                    sel = sec["Delta_bar"] <= Dmax; kap = sec["kappa"][sel]; k2 = sec["k2"][sel]
                    for p in pool:
                        if N in p:
                            pool[p] += list(kap)
                    below = kap[kap <= thr]; above = kap[kap > thr]
                    bmax = float(below.max()) if below.size else np.nan; amin = float(above.min()) if above.size else np.nan
                    pos = below[below > 0]
                    gap_rows.append(dict(N=N, block=f"s{a}_s{b}", L=L, C2=ch["C2"], rep=rep_name(ch["C2"]), X=sec["X"], P=sec["P"],
                                         n_modes=int(sel.sum()), n_selected=int(below.size),
                                         max_kappa_selected=bmax, min_kappa_rejected=amin,
                                         ratio_rejected_over_thr=amin / thr if above.size else np.nan,
                                         ratio_thr_over_selected_nonzero=(thr / pos.max()) if pos.size else np.nan,
                                         local_gap=("no_rejected_mode" if not above.size else
                                                    ("gap_x3" if amin >= R4["local_gap_factor"] * thr and
                                                     (not pos.size or pos.max() <= thr / R4["local_gap_factor"]) else "no_clear_gap"))))
                    for cut in R4["absolute_K2_cutoffs_compared"]:
                        abs_rows.append(dict(N=N, block=f"s{a}_s{b}", L=L, C2=ch["C2"], X=sec["X"], P=sec["P"], K2_cut=cut,
                                             n_kappa_rule=int(below.size), n_abs_rule=int(np.sum(k2 <= cut)),
                                             n_kappa_only=int(np.sum((kap <= thr) & (k2 > cut))),
                                             n_abs_only=int(np.sum((kap > thr) & (k2 <= cut)))))
                    # vector-level correspondence of mid modes with lo/hi modes
                    mid, _ = primary_modes(sds[N], ex["blk"], sec, thr)
                    alt = {t: primary_modes(sds[N], ex["blk"], sec, f)[0] for t, f in (("lo", thr / 3), ("hi", thr * 3))}
                    for m in mid:
                        if m["Delta"] > Dmax + 0.25:
                            continue
                        r_ = dict(N=N, block=f"s{a}_s{b}", L=L, C2=ch["C2"], X=sec["X"], P=sec["P"], Delta=m["Delta"])
                        for t, lst in alt.items():
                            ov = max((float(np.dot(m["vec"], x["vec"]) ** 2) for x in lst), default=0.0)
                            r_[f"best_overlap_{t}"] = ov
                        r_["robust_vector_x3"] = bool(r_["best_overlap_lo"] >= gate and r_["best_overlap_hi"] >= gate)
                        kcorr.append(r_)
    hist_rows = []
    for p, ks in pool.items():
        for w in R4["bin_widths"]:
            for off in R4["bin_offsets_fraction"]:
                t_, c_ = valley_threshold_offset(ks, tuple(cfg["kappa_rule"]["search_range"]), w, off)
                hist_rows.append(dict(pool="N" + "-".join(map(str, (p[0], p[-1]))), bin_width=w, offset_fraction=off,
                                      threshold=t_, valley_count=c_, ratio_to_baseline=t_ / thr))
    gaps = pd.DataFrame(gap_rows); absc = pd.DataFrame(abs_rows); kc = pd.DataFrame(kcorr); hist = pd.DataFrame(hist_rows)
    # attach the vector-level robustness to modes (match on N, block, L, C2, X, P, Delta)
    kc["_k"] = [f"{n}|{bl}|{l}|{c}|{x}|{p}|{d:.9f}" for n, bl, l, c, x, p, d in zip(kc.N, kc.block, kc.L, kc.C2, kc.X, kc.P, kc.Delta)]
    rob = dict(zip(kc._k, kc.robust_vector_x3))

    # ============================================================================ T6 near-short descendants
    ns_rows, norm_rows = [], []
    for N in sizes:
        pr = PRIM[N]; pr = pr[pr.Delta > 0.1]
        for m in pr[pr.Delta <= Dmax].itertuples():
            for p in pr[(pr.C2 == m.C2) & (pr.Delta < m.Delta - 0.5)].itertuples():
                for n in R6["levels_n"]:
                    if abs(m.Delta - p.Delta - n) > R6["match_Delta_tol"] or multiplicity(n, p.L, m.L) == 0:
                        continue
                    if m.P != p.P * (-1) ** (p.L + n + m.L):
                        continue
                    lam = lam_desc(p.Delta, n, m.L, p.L)
                    if lam >= R6["almost_primary_lambda_max"]:
                        continue
                    lo_, hi_ = R6["consistent_band"]
                    # run-3 fix: lambda <= 0 means the finite-N parent lies at/below its unitarity bound (Delta < L+1)
                    st = ("parent_at_or_below_unitarity_bound" if lam <= 0 else
                          "consistent_with_near_short_descendant" if lo_ * lam <= m.K2 <= hi_ * lam else
                          ("candidate_rejected_K2_small" if m.K2 < lo_ * lam else "candidate_K2_above_band"))
                    ns_rows.append(dict(N=N, pid=m.pid, block=m.block, L=m.L, C2=m.C2, rep=rep_name(m.C2), P=m.P, Delta=m.Delta,
                                        K2=m.K2, parent_pid=p.pid, parent_block=p.block, parent_L=p.L, parent_Delta=p.Delta,
                                        parent_X=p.X, X=m.X, level=n, lambda_pred=lam, status=st))
        # normalisation check on guarded descendant raw levels
        rw = RAW[N]; prd = PRIM[N].set_index("pid")
        for r in rw[(rw.cls == "descendant") & rw.guarded & rw.parent_pid.notna()].itertuples():
            ex = EX[N][(int(r.block[1]), int(r.block[4]), r.L)]; ib = int(ex["basis"].to_blk[r.basis_index - 1])
            for ch in ex["blk"]["channels"]:
                if ch["C2"] == r.C2 and "K2" in ch:
                    pos = np.where(ch["guarded"] == ib)[0]
                    if pos.size:
                        p = prd.loc[int(r.parent_pid)]
                        lam = lam_desc(p.Delta, int(r.parent_level), r.L, int(p.L))
                        norm_rows.append(dict(N=N, block=r.block, L=r.L, C2=r.C2, Delta=r.Delta, level=int(r.parent_level),
                                              parent_Delta=p.Delta, parent_L=int(p.L), w_descendant=r.w_descendant,
                                              K2_measured=float(ch["K2"][pos[0], pos[0]]), lambda_pred=lam,
                                              ratio=float(ch["K2"][pos[0], pos[0]]) / lam if lam > 1e-9 else np.nan))
    ns = pd.DataFrame(ns_rows); normc = pd.DataFrame(norm_rows)

    # ============================================================================ T7 mode statuses
    prim_all = pd.concat([PRIM[N] for N in sizes], ignore_index=True)
    modes = modes.merge(prim_all[["N", "pid", "kappa", "K2", "sigma_Delta", "Pnorm_ratio", "stable_kappa_x3"]],
                        on=["N", "pid"], how="left")
    modes = modes.merge(algm[["N", "pid", "comm_lower", "comm_upper", "six_Delta_inside_interval", "coverage"]], on=["N", "pid"], how="left")
    modes["window_status"] = [mode_win.get((n, p), "not_tested") for n, p in zip(modes.N, modes.pid)]
    modes["robust_vector_x3"] = [rob.get(f"{n}|{bl}|{l}|{c}|{x}|{p}|{d:.9f}") for n, bl, l, c, x, p, d in
                                 zip(modes.N, modes.block, modes.L, modes.C2, modes.X, modes.P, modes.Delta)]
    nsc = set(zip(ns[ns.status == "consistent_with_near_short_descendant"].N, ns[ns.status == "consistent_with_near_short_descendant"].pid)) if len(ns) else set()
    modes["near_short_consistent"] = [(n, p) in nsc for n, p in zip(modes.N, modes.pid)]

    def mode_status(r):
        if r.mode_in_descendant_span:
            return "conflict_mode_in_descendant_span"
        if r.near_short_consistent:
            return "uncertain_near_short_descendant"
        if r.window_status == "window_unstable":
            return "uncertain_window"
        if not (r.stable_kappa_x3 and r.robust_vector_x3 is not False):
            return "kappa_sensitive"
        return "no_flag"
    modes["mode_status"] = modes.apply(mode_status, axis=1)
    pid2phys = dict(zip(zip(prim0.N, prim0.pid), prim0.phys_id))
    modes["phys_id"] = [pid2phys.get((n, p)) for n, p in zip(modes.N, modes.pid)]
    order = ["conflict_mode_in_descendant_span", "uncertain_near_short_descendant", "uncertain_window", "kappa_sensitive", "no_flag"]
    phys_status = modes.groupby("phys_id").mode_status.agg(lambda s: min(s, key=order.index))

    # ============================================================================ T7 raw physical levels
    rp = rawp0.copy()
    rp["any_dual_pass"] = rp.phys_level_key.map(agg.any_dual).fillna(False).astype(bool)
    rp["cross_copy_split"] = rp.phys_level_key.map(agg.cross_copy_split).fillna(False).astype(bool)
    rp["cross_sector_PD_conflict"] = rp.phys_level_key.isin(xs_keys)
    rp["window_status"] = rp.phys_level_key.map(win_status).fillna("not_tested")
    dual = rawall[rawall.dual_pass].merge(span[["N", "block", "L", "C2", "X", "P", "triage"]], on=["N", "block", "L", "C2", "X", "P"], how="left")
    rp["dual_pass_triage"] = rp.phys_level_key.map(dual.groupby("phys_level_key").triage.agg(lambda s: "|".join(sorted(set(s.fillna("?"))))))
    rp["primary_mode_status"] = rp.primary_mode_phys_id.map(phys_status)
    am = alg.drop_duplicates("phys_level_key").set_index("phys_level_key")
    for c in ("rel_residual", "coverage", "mean_K_step", "mean_P_step_window"):
        rp[f"alg_{c}"] = rp.phys_level_key.map(am[c])

    def revise(r):
        c = r.cls
        if str(c).startswith("unclassifiable"):
            return c, "unchanged (not a guarded KK source / no descendant span)"
        if r.any_dual_pass:
            why = {"overcomplete": "descendant span fills the sector, w_D not discriminating (information insufficient)",
                   "near_overlap": "primary subspace nearly inside the descendant span"}.get(r.dual_pass_triage, "triage " + str(r.dual_pass_triage))
            return "ambiguous_primary_descendant_overlap", "a copy passes both w_P and w_D gates; " + why
        if r.cross_copy_split:
            return "ambiguous_cross_copy_split", "one copy passes only P, another only D"
        if r.cross_sector_PD_conflict:
            return "ambiguous_cross_sector", "stageB and integer copies disagree P vs D"
        if c == "primary":
            ms = r.primary_mode_status
            if ms == "conflict_mode_in_descendant_span":
                return "ambiguous_primary_descendant_overlap", "its primary mode lies in the descendant span"
            if ms == "uncertain_near_short_descendant":
                return "primary_uncertain_near_short_descendant", "K2 consistent with an almost-null descendant of a lower primary"
            if r.window_status == "window_unstable" or ms == "uncertain_window":
                return "primary_uncertain_window", "class or mode changes when windows shrink"
            if ms == "kappa_sensitive":
                return "primary_kappa_sensitive", "mode not reproduced at kappa_thr/3 and x3 (energy or vector)"
            return "primary", "all hardening checks passed or not applicable"
        if c == "descendant":
            if r.window_status == "window_unstable":
                return "descendant_uncertain_window", "class changes when windows shrink"
            return "descendant", "no conflict"
        return c, "unchanged"
    rv = rp.apply(revise, axis=1, result_type="expand"); rp["revised_class"] = rv[0]; rp["revised_reason"] = rv[1]

    # ============================================================================ T5 tracks
    # v0.4.1 (review R2): diagnostic statuses are aggregated over all copies of a phys_id (worst status); the
    # numerical K2 / kappa are those of the representative copy named in primaries_per_N.csv (N, pid)
    phys_agg = modes.groupby("phys_id").agg(
        window_status=("window_status", worst_window_status),
        near_short_consistent=("near_short_consistent", "any"),
        six_Delta_inside_interval=("six_Delta_inside_interval", lambda s: bool(s.fillna(False).all())))
    phys_agg["mode_status"] = phys_status
    rep = rep0[["N", "pid", "phys_id", "track_id", "blind_name", "Delta"]].merge(
        modes[["N", "pid", "K2", "kappa"]], on=["N", "pid"], how="left").merge(
        phys_agg.reset_index(), on="phys_id", how="left")
    lvl = rp[rp.primary_mode_phys_id.notna()].groupby("primary_mode_phys_id").revised_class.agg(lambda s: "|".join(sorted(set(s))))
    rep["dominant_level_revised"] = rep.phys_id.map(lvl)
    t5 = []
    for t in trk0.itertuples():
        g = rep[rep.track_id == t.track_id].sort_values("N")
        conflict = bool((g.mode_status == "conflict_mode_in_descendant_span").any() or
                        g.dominant_level_revised.fillna("").str.contains("ambiguous").any())
        st = ("insufficient_points" if t.nN < hc["T5_tracks"]["min_points"] else "identity_ambiguous" if t.any_ambiguous else
              "conflict" if conflict else "near_short_candidate" if g.near_short_consistent.fillna(False).any() else
              "window_sensitive" if (g.window_status == "window_unstable").any() else
              "kappa_sensitive" if (g.mode_status == "kappa_sensitive").any() else "no_flag")
        t5.append(dict(track_id=t.track_id, blind_name=t.blind_name, nN=t.nN, Ns=t.Ns, track_status=st,
                       K2_by_N=";".join(f"{n}:{k:.3g}" for n, k in zip(g.N, g.K2)),
                       mode_status_by_N=";".join(f"{n}:{s}" for n, s in zip(g.N, g.mode_status)),
                       level_by_N=";".join(f"{n}:{s}" for n, s in zip(g.N, g.dominant_level_revised.fillna("-")))))
    t5 = pd.DataFrame(t5)

    # ============================================================================ OPE identity status
    oz = ope0.copy()
    oz["X_mode_status"] = oz.X_phys_id.map(phys_status); oz["Z_mode_status"] = oz.Z_phys_id.map(phys_status)
    oz["ope_identity_status"] = np.where(oz.X_mode_status.eq("no_flag") & oz.Z_mode_status.eq("no_flag"), "ok",
                                         "caution:" + oz.X_mode_status.fillna("none") + "/" + oz.Z_mode_status.fillna("none"))

    # ============================================================================ write
    W = lambda df, f: df.to_csv(os.path.join(out, f), index=False)
    W(rawall.drop(columns=[]), "T1_raw_copies_dual_pass.csv"); W(span, "T1_span_overlap_by_sector.csv")
    W(alg, "T2_algebra_raw_states.csv"); W(algm, "T2_algebra_primary_modes.csv")
    W(t3r, "T3_window_shrink_raw.csv"); W(t3m, "T3_window_shrink_modes.csv")
    W(gaps, "T4_kappa_local_gaps.csv"); W(hist, "T4_threshold_rule_variants.csv"); W(absc, "T4_absolute_K2_cutoffs.csv")
    W(kc.drop(columns=["_k"]), "T4_mode_vector_correspondence.csv")
    W(t5, "T5_track_status.csv"); W(ns, "T6_near_short_candidates.csv"); W(normc, "T6_descendant_K2_normalisation.csv")
    W(modes, "T7_primary_modes_status.csv"); W(rp, "T7_levels_revised.csv"); W(oz, "T7_ope_rows_identity_status.csv")
    diff = rp.groupby(["N", "cls", "revised_class"]).size().rename("n").reset_index()
    W(diff, "T7_class_transitions.csv")
    a_in = alg[(alg.coverage == "K_complete") & (alg.Delta > 0.1)]
    summary = dict(
        harden_version=__version__, baseline_run=brun, regression=regress, kappa_threshold=thr, gate=gate,
        T1=dict(same_copy_dual_pass={int(N): int(RAW[N].dual_pass.sum()) for N in sizes},
                levels_with_dual_pass=int(rp.any_dual_pass.sum()), cross_copy_split=int(rp.cross_copy_split.sum()),
                cross_sector_PD_conflicts=int(rp.cross_sector_PD_conflict.sum()),
                modes_in_descendant_span=int(modes.mode_in_descendant_span.sum()),
                span_triage=span.triage.value_counts().to_dict()),
        T2=dict(states=len(alg), coverage=alg.coverage.value_counts().to_dict(),
                rel_residual_K_complete_quantiles=a_in.rel_residual.abs().quantile([.5, .9, .99]).round(4).tolist() if len(a_in) else None,
                modes_6Delta_inside_interval=int(algm.six_Delta_inside_interval.sum()), modes=len(algm)),
        T3=dict(raw_status=t3r.status.value_counts().to_dict(), level_window_status=rp.window_status.value_counts().to_dict(),
                mode_window_status=modes.window_status.value_counts().to_dict()),
        T4=dict(local_gap=gaps.local_gap.value_counts().to_dict(),
                threshold_variants_range=[float(hist.threshold.min()), float(hist.threshold.max())],
                modes_robust_vector_x3=int(kc.robust_vector_x3.sum()), modes_checked=len(kc)),
        T5=t5.track_status.value_counts().to_dict(),
        T6=dict(candidates=ns.status.value_counts().to_dict() if len(ns) else {},
                normalisation_ratio_quantiles=normc.ratio.quantile([.1, .5, .9]).round(4).tolist() if len(normc) else None,
                normalisation_levels=len(normc)),
        T7=dict(revised=rp.revised_class.value_counts().to_dict(), mode_status=modes.mode_status.value_counts().to_dict(),
                ope_rows=oz.ope_identity_status.str.split(":").str[0].value_counts().to_dict(),
                ope_publishable_rows_with_caution=int((oz.publishable & oz.ope_identity_status.ne("ok")).sum())))
    json.dump(summary, open(os.path.join(out, "SUMMARY.json"), "w"), indent=1, default=str)
    code = {f"{d}/{f}": sha256_file(os.path.join(root, d, f)) for d in ("blindpd", "tests") for f in sorted(os.listdir(os.path.join(root, d)))
            if f.endswith(".py")}
    mf = dict(harden_version=__version__, harden_config=hc, harden_config_sha256=sha256_file(hcfg_path),
              baseline_config_sha256=sha256_file(bcfg_path), baseline_manifest_sha256=sha256_file(os.path.join(brun, "MANIFEST.json")),
              baseline_inputs_verified=checked, code=code, python=platform.python_version(), numpy=np.__version__,
              pandas=pd.__version__, executable=sys.executable, created=datetime.datetime.now().isoformat(timespec="seconds"),
              note="post-freeze method revision; baseline outputs untouched; no published table read")
    mf["outputs"] = {f: sha256_file(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    json.dump(mf, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
