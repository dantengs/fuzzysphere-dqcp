"""v0.5.0-dev: descendant_span_v1 S3 (conflict resolution) and S4 (propagation).
python -m blindpd.resolve_s3 <run_dir> <harden_dir> <out_dir>

run_dir     a run.py output made with cfg['descendant_span'] (S1/S2 classes, conflict records, S3(b) evidence)
harden_dir  harden.py on that run (T3 window statuses with the same rule active in the shrunk variants: S3(c))
Integer sectors only: stageB sectors are not hardened (no T3), so their S2 classes are reported unresolved.
"""
import json, os, sys, hashlib
import numpy as np
import pandas as pd

AMB = ("ambiguous_primary_descendant_conflict", "ambiguous_primary_descendant_overlap")
P_T = ("primary", "primary_resolved"); D_T = ("descendant", "descendant_resolved")
CLS_ORDER = ["primary", "primary_resolved", "descendant", "descendant_resolved", "nonprimary_partial_family_coverage",
             "nonprimary_no_family_coverage", "mixed_primary_descendant", "unclassifiable_unguarded", "unclassifiable_not_KK_source"]


def decide(sup_P, sup_D, oppose):
    """S3 decision: >= 1 item supports a side, none supports the other, and (c) does not oppose."""
    if oppose:
        return None
    if sup_P and not sup_D:
        return "P"
    if sup_D and not sup_P:
        return "D"
    return None


def resolve_levels(raw, conf_ev, win):
    """Per raw copy: S3 evidence and the resolved class.  raw: integer raw copies of the run; conf_ev: conflict_id ->
    evidence_b; win: phys_level_key -> T3 window status."""
    clean = raw[(raw.sector_n_conflicts == 0) & raw.cls.isin(["primary", "descendant"])]
    clean_by = clean.groupby("phys_level_key")
    clean_sets = {k: (set(g.block), set(g.cls)) for k, g in clean_by}
    rows = []
    for r in raw.itertuples():
        ev_a = ev_b = ev_c = "-"; fin = r.cls; why = ""
        if r.cls in AMB:
            sup_P = sup_D = False
            cs = clean_sets.get(r.phys_level_key)
            others = clean[(clean.phys_level_key == r.phys_level_key) & (clean.block != r.block)] if cs else clean.iloc[:0]
            if len(others):
                ev_a = "|".join(sorted(set(others.cls)))
                sup_P |= "primary" in set(others.cls); sup_D |= "descendant" in set(others.cls)
            else:
                ev_a = "no_conflict_free_copy"
            if isinstance(r.conflict_id, str):
                ev_b = conf_ev.get(r.conflict_id, "missing")
                sup_P |= ev_b == "supports_P"; sup_D |= ev_b == "supports_D"
            else:
                ev_b = "no_conflict_direction"
            ev_c = win.get(r.phys_level_key, "not_tested")
            side = decide(sup_P, sup_D, ev_c == "window_unstable")
            if side == "P":
                fin, why = "primary_resolved", "S3 evidence supports P"
            elif side == "D":
                fin, why = "descendant_resolved", "S3 evidence supports D"
            else:
                why = ("window unstable (opposes both)" if ev_c == "window_unstable" else
                       "contradicting evidence" if (sup_P and sup_D) else "no independent evidence")
        rows.append(dict(cls_final=fin, S3_a=ev_a, S3_b=ev_b, S3_c=ev_c, S3_reason=why))
    return pd.concat([raw.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def resolve_modes(prim, conf_ev, mwin):
    rows = []
    clean_phys = set(prim[~prim.conflict_pair.astype(bool)].phys_id)
    for m in prim.itertuples():
        if not bool(m.conflict_pair):
            rows.append(dict(mode_identity="clean", S3_a="-", S3_b="-", S3_c="-")); continue
        others = prim[(prim.phys_id == m.phys_id) & (prim.pid != m.pid) & (prim.N == m.N) & ~prim.conflict_pair.astype(bool)]
        sup_P = len(others) > 0; ev_a = "clean_copy" if sup_P else "no_clean_copy"
        ev_b = conf_ev.get(m.conflict_id, "missing") if isinstance(m.conflict_id, str) else "no_conflict_direction"
        sup_P |= ev_b == "supports_P"; sup_D = ev_b == "supports_D"
        ev_c = mwin.get((m.N, m.pid), "not_tested")
        side = decide(sup_P, sup_D, ev_c == "window_unstable")
        rows.append(dict(mode_identity={"P": "primary_resolved", "D": "descendant_resolved"}.get(side, "conflict_unresolved"),
                         S3_a=ev_a, S3_b=ev_b, S3_c=ev_c))
    return pd.concat([prim.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def phys_mode_identity(s):
    s = set(s)
    if "conflict_unresolved" in s or ("descendant_resolved" in s and s & {"primary_resolved", "clean"}):
        return "conflict"
    for x in ("descendant_resolved", "primary_resolved"):
        if x in s:
            return x
    return "clean"


def phys_level_class(s):
    s = list(s)
    for a in AMB:
        if a in s:
            return a
    if set(s) & set(P_T) and set(s) & set(D_T):
        return "ambiguous_cross_copy_split"
    return min(s, key=lambda x: CLS_ORDER.index(x) if x in CLS_ORDER else len(CLS_ORDER))


def main(run, hard, out):
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"refusing to write into non-empty {out}")
    os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(run, "MANIFEST.json"), encoding="utf-8"))
    if not man["config"].get("descendant_span"):
        sys.exit("run was made without descendant_span")
    hman = json.load(open(os.path.join(hard, "MANIFEST.json"), encoding="utf-8"))
    if os.path.normcase(os.path.normpath(hman["harden_config"]["baseline"]["run"])) != os.path.normcase(os.path.normpath(run)):
        sys.exit("harden output does not belong to this run")
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    bad = [f for f, h in man["outputs"].items() if sha(os.path.join(run, f)) != h] + \
          [f for f, h in hman["outputs"].items() if sha(os.path.join(hard, f)) != h]
    if bad:
        sys.exit(f"inputs changed since their MANIFEST: {bad[:5]}")
    rd = lambda d, f: pd.read_csv(os.path.join(d, f), low_memory=False)
    raw = rd(run, "raw_levels_all_copies.csv"); prim = rd(run, "primary_modes_all_copies.csv")
    conf = rd(run, "descendant_span_conflicts.csv")
    conf_ev = dict(zip(conf.conflict_id, conf.evidence_b)) if len(conf) else {}
    lv = rd(hard, "T7_levels_revised.csv"); win = dict(zip(lv.phys_level_key, lv.window_status))
    hm = rd(hard, "T7_primary_modes_status.csv"); mwin = {(n, p): s for n, p, s in zip(hm.N, hm.pid, hm.window_status)}
    lev = resolve_levels(raw, conf_ev, win)
    mod = resolve_modes(prim, conf_ev, mwin)
    # ---- S4
    pmi = mod.groupby("phys_id").mode_identity.agg(phys_mode_identity)
    plc = lev.groupby("phys_level_key").agg(cls_final_phys=("cls_final", phys_level_class),
                                            classes_final=("cls_final", lambda x: "|".join(sorted(set(x)))),
                                            classes_v035=("cls_v035", lambda x: "|".join(sorted(set(x)))),
                                            n_copies=("cls", "size")).reset_index()
    rep = rd(run, "primaries_per_N.csv"); trk = rd(run, "primary_tracks.csv")
    rep["phys_mode_identity"] = rep.phys_id.map(pmi)
    tr = []
    for t in trk.itertuples():
        g = rep[rep.track_id == t.track_id]
        tr.append(dict(track_id=t.track_id, blind_name=t.blind_name, nN=t.nN,
                       track_identity=("conflict" if g.phys_mode_identity.eq("conflict").any() else
                                       "contains_descendant_resolved" if g.phys_mode_identity.eq("descendant_resolved").any() else
                                       "contains_primary_resolved" if g.phys_mode_identity.eq("primary_resolved").any() else "clean"),
                       identity_by_N=";".join(f"{n}:{s}" for n, s in zip(g.N, g.phys_mode_identity))))
    tr = pd.DataFrame(tr)
    ope = rd(run, "ope_rows_integer_exact.csv")
    ope["X_identity"] = ope.X_phys_id.map(pmi); ope["Z_identity"] = ope.Z_phys_id.map(pmi)
    badid = ope.X_identity.isin(["conflict", "descendant_resolved"]) | ope.Z_identity.isin(["conflict", "descendant_resolved"])
    ope["ds_identity_status"] = np.where(badid, "ambiguous_or_descendant", "ok")
    ope["publishable_ds"] = ope.publishable.astype(bool) & ~badid
    W = lambda df, f: df.to_csv(os.path.join(out, f), index=False)
    W(lev, "S3_raw_copies_resolved.csv"); W(mod, "S3_modes_resolved.csv"); W(plc, "S4_levels_final.csv")
    W(tr, "S4_tracks_identity.csv"); W(ope, "S4_ope_rows_identity.csv")
    amb = lev[lev.cls.isin(AMB)]
    summ = dict(option=man["config"]["descendant_span"]["option"],
                S2_copy_classes=lev.cls.value_counts().to_dict(),
                S3_ambiguous_copies=len(amb), S3_resolved=amb.cls_final.value_counts().to_dict(),
                S3_resolved_by_b_alone=int(((amb.cls_final != amb.cls) & ~amb.S3_a.isin(["primary", "descendant", "descendant|primary"]) &
                                            amb.S3_b.isin(["supports_P", "supports_D"])).sum()),
                S3_b_evidence=conf.evidence_b.value_counts().to_dict() if len(conf) else {},
                conflict_pair_modes=int(mod.conflict_pair.astype(bool).sum()),
                mode_identity=mod.mode_identity.value_counts().to_dict(),
                phys_mode_identity=pmi.value_counts().to_dict(),
                S4_level_classes=plc.cls_final_phys.value_counts().to_dict(),
                tracks=tr.track_identity.value_counts().to_dict(),
                ope_publishable_before=int(ope.publishable.astype(bool).sum()), ope_publishable_after=int(ope.publishable_ds.sum()))
    json.dump(summ, open(os.path.join(out, "SUMMARY.json"), "w"), indent=1, default=str)
    mf = dict(run=run, harden=hard, run_manifest_sha256=sha(os.path.join(run, "MANIFEST.json")),
              harden_manifest_sha256=sha(os.path.join(hard, "MANIFEST.json")),
              code_sha256=sha(os.path.abspath(__file__)), rule=man["config"]["descendant_span"])
    mf["outputs"] = {f: sha(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    json.dump(mf, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
