"""Standard sensitivity scan (v0.2): every judgement parameter varied one at a time, in memory.

Reported per N: number of raw levels whose class changes w.r.t. the baseline, and the class of every
level that is 'borderline' (any weight within the gate band) under each variant.  Also tracking-parameter
robustness of the cross-N tracks.  No variant is used for the main results.
"""
import copy
import numpy as np
import pandas as pd
from . import spectrum as S
from .tracking import build_tracks


def calibrations(sd, export_dir):
    """Generator coefficients from the frozen rule (min vacuum image s.t. a.c = sqrt(2 Delta)) for S, phi, T."""
    e = sd.edges[(0, 0, 0, 1)]; nt, ns = e["nt"], e["ns"]
    raw = np.fromfile(f"{export_dir}/{e['bin']}", "<f8")
    O1 = raw[:nt * ns].reshape((nt, ns), order="F"); O2 = raw[nt * ns:2 * nt * ns].reshape((nt, ns), order="F")
    G = raw[2 * nt * ns:].reshape((2, 2, ns), order="F")
    sE = np.array(e["source_E"]); sC = np.array(e["source_C2"]); tE = np.array(e["target_E"]); tC = np.array(e["target_C2"])
    v = sd.meta["calibration"]["vacuum"] - 1; out = {}
    for name, c2 in (("S", 0), ("phi", 2), ("T", 5)):
        s = sd.meta["calibration"]["singlet"] - 1 if c2 == 0 else [i for i in np.argsort(sE) if abs(sC[i] - c2) < 1e-5][0]
        Dl = sd.alpha * (sE[s] - sd.E0)
        ds = [i for i in range(nt) if abs(tC[i] - c2) < 1e-5]; d = ds[int(np.argmin(np.abs(tE[ds] - sE[s] - 1 / sd.alpha)))]
        a = np.array([O1[d, s], O2[d, s]]); x = np.linalg.solve(G[:, :, v], a); c = x * np.sqrt(2 * Dl) / (a @ x)
        out[name] = c * np.sign(c[0]) * np.sign(sd.coeff[0])
    return out


def _classes(sd, par, cfg, thr, coeff=None, null_sv=None):
    sd = copy.copy(sd); sd._cache = {}
    if coeff is not None:
        sd.coeff = np.array(coeff, float)
    old = S._orth_top.__defaults__
    if null_sv is not None:
        S._orth_top.__defaults__ = (null_sv,)
    try:
        pr, rw, fm, g, mv = S.analyse_size(sd, cfg, thr, par)
    finally:
        S._orth_top.__defaults__ = old
    rw = pd.DataFrame(rw)
    rw["k"] = [f"{c}:{l}:{p}:{e:.7f}" for c, l, p, e in zip(rw.C2, rw.L, rw.P, rw.energy)]
    order = {"primary": 0, "descendant": 1, "nonprimary_partial_family_coverage": 2, "nonprimary_no_family_coverage": 3,
             "mixed_primary_descendant": 4}
    rw["o"] = rw.cls.map(order).fillna(9)
    best = rw.sort_values(["o", "w_primary"], ascending=[True, False]).groupby("k").first()
    pr = pd.DataFrame(pr)
    return best, len(pr[pr.Delta <= cfg["Delta_max"]])


def scan(sds, pars, cfg, thr, export_root):
    rows, borderline = [], []
    for N, sd in sds.items():
        par = pars[N]
        base, nb = _classes(sd, par, cfg, thr)
        cal = calibrations(sd, f"{export_root}/N{N}")
        variants = [("baseline", {}, cfg)]
        for k in ("phi", "T"):
            variants.append((f"generator_calibrated_on_{k}", dict(coeff=cal[k]), cfg))
        for f in (0.99, 1.01):
            c = sd.coeff.copy(); c[1] *= f; variants.append((f"generator_ratio_x{f}", dict(coeff=c), cfg))
        for f in (1 / 3, 3):
            variants.append((f"kappa_thr_x{f:.3g}", dict(thr=thr * f), cfg))
        for w in (0.8, 0.95):
            c = dict(cfg); c["raw_class_weight"] = w; variants.append((f"weight_gate_{w}", {}, c))
        for sv in (0.1, 0.6):
            variants.append((f"null_singular_value_{sv}", dict(null_sv=sv), cfg))
        for name, kw, c in variants:
            t = kw.get("thr", thr)
            best, n = (base, nb) if name == "baseline" else _classes(sd, par, c, t, kw.get("coeff"), kw.get("null_sv"))
            changed = best.cls != base.cls.reindex(best.index)
            nb_ = best.cls.reindex(base.index)
            fl = base.index[((base.cls == "primary") & (nb_ == "descendant")) | ((base.cls == "descendant") & (nb_ == "primary"))]
            pp = len(fl)
            flipped = ";".join("{}|L{}|P{:+d}|{:.3f}:{}->{}".format(base.loc[k, "rep"], base.loc[k, "L"], int(base.loc[k, "P"]), base.loc[k, "Delta"], str(base.loc[k, "cls"])[:4], str(nb_[k])[:4]) for k in fl)
            rows.append(dict(N=N, variant=name, n_levels=len(best), class_changes=int(changed.sum()),
                             primary_descendant_flips=int(pp), flipped_levels=flipped, primary_modes_in_window=n,
                             coeff_ratio=float((kw.get("coeff", sd.coeff))[1] / (kw.get("coeff", sd.coeff))[0])))
            band = best[((best.w_primary > 0.05) & (best.w_primary < 0.95)) | ((best.w_descendant > 0.8) & (best.w_descendant < 0.95))]
            if name == "baseline":
                for k, r in band.iterrows():
                    borderline.append(dict(N=N, key=k, rep=r.rep, L=r.L, P=r.P, Delta=r.Delta, cls=r.cls,
                                           w_primary=r.w_primary, w_descendant=r.w_descendant))
    return pd.DataFrame(rows), pd.DataFrame(borderline)


def tracking_scan(pts, sizes, base_rule):
    def sig(tr):
        return {frozenset(q["phys_id"] for q in t["points"]) for t in tr}
    base = build_tracks(pts, sizes, **base_rule); B = sig(base)
    core = {frozenset(q["phys_id"] for q in t["points"]) for t in base
            if len(t["points"]) == len(sizes) and min(q["Delta"] for q in t["points"]) < 5.0}
    out = []
    for mj in (0.15, 0.2, 0.5):
        for am in (0.02, 0.05, 0.1):
            r = dict(base_rule); r.update(max_jump=mj, ambiguity_margin=am); t = build_tracks(pts, sizes, **r); Sg = sig(t)
            out.append(dict(max_jump=mj, ambiguity_margin=am, tracks=len(t), changed_vs_base=len(Sg - B),
                            core_tracks=len(core), core_changed=len(core - Sg),
                            ambiguous=sum(any(q["ambiguous"] for q in x["points"]) for x in t)))
    return pd.DataFrame(out)
