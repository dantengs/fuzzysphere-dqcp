"""Comparison with published tables (separate from the analysis; v0.5.1 moved it out of the blindpd package).
Compares frozen results with 2306.16435 Table VI / Table III.  The blindpd package never imports this module.

python -m comparison.unblind <blind_results_dir> <paper_table_vi_inventory.csv> <out_dir>
The blind directory is verified against its MANIFEST (outputs unchanged) before anything is read.
Nothing in the blind directory is modified.  Convention maps used only here:
  phi x phi -> T     : paper f = lambda_iso(5x5->14)            (trace = iso for this channel)
  phi x phi -> S     : paper f = lambda_iso(5x5->1) / sqrt(5)   (= lambda_iso(5x1->5), phi_a phi_b ~ delta_ab f S)
  phi x phi -> J     : paper f = lambda_iso(5x5->10) / sqrt(2)
  phi x phi -> stress: paper f = lambda_iso(5x5->1, canonical-I) / (sqrt(5) * 2 sqrt(2))
(r1 of this script omitted the 1/sqrt(5) of the singlet channels; kept as record in *_unblind.)
"""
import json, os, sys
import numpy as np
import pandas as pd
from blindpd.io import sha256_file

# conversion from our lambda_iso to the published f (moved here from blindpd.normalization in v0.5.1)
PAPER_CONVENTION_FACTORS = {"T": 1.0, "S": 1 / np.sqrt(5), "J": 1 / np.sqrt(2), "stress": 1 / (np.sqrt(5) * 2 * np.sqrt(2))}

TABLE_III = {  # arXiv:2306.16435 Table III (linear 1/N extrapolation, N <= 10)
    "T": (1.242, 0.007), "J": (0.771, 0.003), "S": (0.235, 0.008), "stress": (0.121, 0.008)}
REPMAP = {"35prime": "35p", "91": "C2=20{231|91}"}


def main(blind, inv_path, out):
    man = json.load(open(os.path.join(blind, "MANIFEST.json")))
    for f, h in man["outputs"].items():
        if sha256_file(os.path.join(blind, f)) != h:
            sys.exit(f"blind output {f} changed after freezing")
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"refusing to write into non-empty {out}")
    os.makedirs(out, exist_ok=True)
    raw = pd.read_csv(os.path.join(blind, "raw_levels_classified.csv"))
    prim = pd.read_csv(os.path.join(blind, "primaries_per_N.csv"))
    cur = pd.read_csv(os.path.join(blind, "ope_curves.csv"))
    inv = pd.read_csv(inv_path)
    r10 = raw[raw.N == 10]; p10 = prim[prim.N == 10]
    rows = []
    for x in inv.itertuples():
        rep = REPMAP.get(str(x.rep), str(x.rep)); L = int(x.paper_L); P = 1 if x.paper_parity == "+" else -1
        g = r10[(r10.rep == rep) & (r10.L == L) & (r10.P == P)]
        d = np.abs(g.Delta - x.paper_Delta) if len(g) else pd.Series(dtype=float)
        hit = g.iloc[int(np.argmin(d))] if len(g) and d.min() < 2e-3 else None
        h = p10[(p10.rep == rep) & (p10.L == L) & (p10.P == P)]
        dp = np.abs(h.Delta - x.paper_Delta) if len(h) else pd.Series(dtype=float)
        pm = h.iloc[int(np.argmin(dp))] if len(h) and dp.min() < 0.15 else None
        ours = None if hit is None else hit.cls
        paper = x.paper_PD if isinstance(x.paper_PD, str) else ""
        if hit is None:
            agree = "no_level_match"
        elif paper == "P":
            agree = "agree" if ours == "primary" else ("partial_mixed" if ours == "mixed_primary_descendant" else "disagree")
        elif paper == "D":
            # v0.3.2: non-primary without an identified descendant family is not counted as agreement with paper D
            agree = "agree" if ours in ("descendant", "descendant_parent_unresolved") else \
                    ("nonprimary_family_unresolved" if ours in ("nonprimary_no_family_coverage", "nonprimary_partial_family_coverage")
                     else ("partial_mixed" if ours == "mixed_primary_descendant" else "disagree"))
        else:
            agree = "paper_blank"
        rows.append(dict(paper_id=x.paper_id, rep=rep, L=L, P=P, paper_Delta=x.paper_Delta, paper_PD=paper,
                         our_Delta=None if hit is None else hit.Delta, our_class=ours,
                         our_w_primary=None if hit is None else hit.w_primary,
                         our_w_descendant=None if hit is None else hit.w_descendant,
                         our_parent=None if hit is None else hit.parent_blind_name,
                         nearest_primary_mode=None if pm is None else pm.blind_name,
                         nearest_primary_Delta=None if pm is None else pm.Delta, agreement=agree))
    sv = pd.DataFrame(rows)
    sv.to_csv(os.path.join(out, "tableVI_comparison_N10.csv"), index=False)
    # our N10 primaries with Delta <= 5.5 and L <= 3 that have no paper row
    paper_keys = set((REPMAP.get(str(a), str(a)), int(b), 1 if c == "+" else -1, float(d))
                     for a, b, c, d in zip(inv.rep, inv.paper_L, inv.paper_parity, inv.paper_Delta))
    extra = p10[(p10.Delta <= 5.5) & (p10.L <= 3)].copy()
    # Table VI lists raw energy levels: compare through the dominant raw level of each primary mode
    domD = {pid: d for pid, d in zip(r10.primary_mode_phys_id, r10.Delta) if isinstance(pid, str)}
    extra["dominant_raw_Delta"] = extra.phys_id.map(domD)
    extra["in_paper_table"] = [any(k[:3] == (r, l, p) and np.isfinite(dd) and abs(k[3] - dd) < 2e-3 for k in paper_keys)
                               for r, l, p, dd in zip(extra.rep, extra.L, extra.P, extra.dominant_raw_Delta.astype(float))]
    extra.to_csv(os.path.join(out, "our_N10_primaries_vs_paper_table.csv"), index=False)
    # Table III
    phi = "5|L0|P-1|b1"
    src = "source" if "source" in cur.columns else "namespace"
    ex_ns = "integer_exact_blind_vectors" if src == "source" else "integer_KK"
    hf_ns = ("half_exact_blind_vectors" if (cur[src] == "half_exact_blind_vectors").any() else "half_production_vectors_energy_match") \
        if src == "source" else "half_projected_local"
    f = PAPER_CONVENTION_FACTORS
    spec = {"T": (ex_ns, "phi", 0, "14|L0|P+1|b1", f["T"]), "S": (ex_ns, "phi", 0, "1|L0|P+1|b2", f["S"]),
            "J": (hf_ns, "phi", 1, "10|L1|P+1|b1", f["J"]),
            "stress": (ex_ns, "phi", 2, "1|L2|P+1|b1", f["stress"])}
    t3 = []
    for k, (ns, route, rank, z, fac) in spec.items():
        g = cur[(cur[src] == ns) & (cur.route == route) & (cur["rank"] == rank) &
                (cur.X_blind_name == phi) & (cur.Z_blind_name == z)]
        if g.empty:
            t3.append(dict(channel=k, status="curve_missing")); continue
        c = g.iloc[0]; v, e = c.iso_intercept * fac, c.iso_err_main * fac
        md = c.get("iso_model_diff", np.nan) * fac
        pv, pe = TABLE_III[k]
        t3.append(dict(channel=f"phi x phi -> {k}", Z=z, source=ns, ours_paper_convention=v, ours_err_main=e,
                       ours_model_diff=md, paper=pv, paper_err=pe, diff=v - pv,
                       diff_over_combined_err=(v - pv) / np.hypot(e, pe),
                       diff_over_combined_err_incl_model=(v - pv) / np.sqrt(e ** 2 + pe ** 2 + (md if md == md else 0) ** 2),
                       lambda_iso_intercept=c.iso_intercept, convention_factor=fac, fit_Ns=c.iso_fit_Ns))
    t3 = pd.DataFrame(t3); t3.to_csv(os.path.join(out, "tableIII_comparison.csv"), index=False)
    # dossier of every paper row we do not plainly agree with: class, weights and Delta at every N
    rawall = raw
    dos = []
    for x in sv[sv.agreement.isin(["disagree", "partial_mixed", "no_level_match"])].itertuples():
        key10 = r10[(r10.rep == x.rep) & (r10.L == x.L) & (r10.P == x.P)]
        if not len(key10):
            continue
        d10 = key10.iloc[int(np.argmin(np.abs(key10.Delta - x.paper_Delta)))]
        # follow the level across N by nearest Delta continuity within the same (rep, L, P)
        prevD = d10.Delta
        for N in sorted(rawall.N.unique(), reverse=True):
            g = rawall[(rawall.N == N) & (rawall.rep == x.rep) & (rawall.L == x.L) & (rawall.P == x.P)]
            if not len(g):
                continue
            h = g.iloc[int(np.argmin(np.abs(g.Delta - prevD)))]
            if abs(h.Delta - prevD) > 0.3:
                continue
            prevD = h.Delta
            dos.append(dict(paper_id=x.paper_id, paper=f"{x.rep} L{x.L}{'+' if x.P > 0 else '-'} {x.paper_Delta} ({x.paper_PD})",
                            N=int(N), Delta=h.Delta, cls=h.cls, w_primary=h.w_primary, w_descendant=h.w_descendant,
                            w_primary_kappa_div3=h.w_primary_thr_lo, w_primary_kappa_x3=h.w_primary_thr_hi,
                            parent=h.parent_blind_name, parent_level=h.parent_level, primary_mode=h.primary_blind_name))
    pd.DataFrame(dos).to_csv(os.path.join(out, "disputed_states_dossier.csv"), index=False)
    summ = dict(blind_manifest_sha256=sha256_file(os.path.join(blind, "MANIFEST.json")),
                tableVI_agreement=sv.agreement.value_counts().to_dict(),
                tableVI_by_paperPD={k: g.agreement.value_counts().to_dict() for k, g in sv.groupby("paper_PD")},
                our_N10_primaries_le5p5_L_le3=int(len(extra)), of_which_not_in_paper_table=int((~extra.in_paper_table).sum()),
                tableIII=t3.to_dict("records"))
    json.dump(summ, open(os.path.join(out, "UNBLIND_SUMMARY.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:4])
