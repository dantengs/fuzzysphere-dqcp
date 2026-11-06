"""Primary/descendant analysis driver.  python -m blindpd.run <config.json> <out_dir>

Reads only: KK export and fullQ OPE-edge export (cluster), per-N parity.csv and ope3.toml, and the
production per-N OPE rows (half-integer sector values; group-theory/normalisation constants).
No published table is read.  Refuses to write into a non-empty output directory.
"""
import json, os, sys, platform, datetime
import numpy as np
import pandas as pd
from . import __version__
from .io import SizeData, sha256_file, tree_hashes, READ_LOG
from .kk import block_kk
from .thresholds import valley_threshold
from .spectrum import analyse_size
from .tracking import build_tracks, hungarian
from .fits import fit_series
from .so5 import rep_name, types
from . import ope as opem
from . import ope_exact as ox
from . import sensitivity as sens
from . import stageb as sb
from . import normalization as nm

ROUTE = {"gamma2": "phi", "singlet": "S", "tensorG": "T"}
ANTISYM = {"phi": {"10"}, "S": set(), "T": {"10", "81"}}       # antisymmetric parts of X (x) X for X = 5, 1, 14
CLS_ORDER = {"primary": 0, "descendant": 1, "nonprimary_partial_family_coverage": 2, "nonprimary_no_family_coverage": 3,
             "mixed_primary_descendant": 4, "unclassifiable_unguarded": 5, "unclassifiable_not_KK_source": 6,
             # v0.5.0-dev (descendant_span_v1 S2); the per-copy resolution and S4 propagation are in resolve_s3.py
             "ambiguous_primary_descendant_overlap": 1.5, "ambiguous_primary_descendant_conflict": 1.6}


def code_manifest(root):
    out = {}
    for d in ("blindpd", "cluster", "tests"):
        for f in sorted(os.listdir(os.path.join(root, d))):
            if f.endswith((".py", ".jl")):
                out[f"{d}/{f}"] = sha256_file(os.path.join(root, d, f))
    return out


def dim_of(c2):
    t = types(c2)
    return t[0][2] if len(t) == 1 else np.nan


def main(cfg_path, out):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(cfg_path))
    base = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))
    P = lambda p: os.path.normpath(os.path.join(base, p))
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"refusing to write into non-empty {out}")
    os.makedirs(out, exist_ok=True)
    sizes = cfg["sizes"]; Dmax = cfg["Delta_max"]
    # v0.3.2 (A5): every input tree that is read, hashed at start and re-verified at the end of the run
    imagenorm_path = P(cfg.get("stageB_imagenorm", "inputs/stageB_imagenorm.csv"))
    input_roots = [os.path.join(P(cfg["kk_export"]), f"N{N}") for N in sizes] + \
                  [os.path.join(P(cfg["ope_export"]), f"N{N}") for N in sizes] + \
                  [os.path.join(P(cfg["physics"]), f"N{N}") for N in sizes] + [P(cfg["ope_per_N_rows"])]
    for spec in cfg.get("stageB", []):
        input_roots += [os.path.join(P(spec["export"]), f"N{N}") for N in spec["sizes"]]
        input_roots += [P(spec["production_rows"]).format(N=N) for N in spec["sizes"]]
        if spec.get("rawtarget"):
            input_roots += [P(spec["rawtarget"].format(N=N)) for N in spec.get("rawtarget_sizes", [])]
    if cfg.get("stageB"):
        input_roots.append(imagenorm_path)
    input_roots += [P(v) for v in cfg.get("ope_fullq_rows_extra", {}).values()]
    input_tree_start = tree_hashes(input_roots)
    sds, pars = {}, {}
    for N in sizes:
        sds[N] = SizeData(os.path.join(P(cfg["kk_export"]), f"N{N}"), os.path.join(P(cfg["physics"]), f"N{N}"), N)
        pars[N] = pd.read_csv(os.path.join(P(cfg["physics"]), f"N{N}", "parity.csv"))
    # ---- step 2a: kappa threshold
    pool = []
    for N in cfg["kappa_rule"]["pool_sizes"]:
        sd = sds[N]
        for key in sd.blocks:
            a, b, L = int(key[1]), int(key[4]), int(key.split("L")[1])
            if L > cfg["L_max"]:
                continue
            for ch in block_kk(sd, a, b, L)["channels"]:
                for s in ch["sectors"]:
                    pool += list(s["kappa"][s["Delta_bar"] <= Dmax])
    thr, thr_info = valley_threshold(pool, tuple(cfg["kappa_rule"]["search_range"]))
    # ---- steps 2-3 per N
    P_rows, R_rows, F_rows, G, MV = [], [], [], {}, {}
    DS_conf, DS_drop = [], []                        # v0.5.0-dev: descendant_span_v1 records
    for N in sizes:
        pr, rw, fm, g, mv = analyse_size(sds[N], cfg, thr, pars[N])
        DS_conf += [dict(r, sector="integer") for r in g.pop("ds_conflicts", [])]
        DS_drop += [dict(r, sector="integer") for r in g.pop("ds_dropped", [])]
        P_rows += pr; R_rows += rw; F_rows += fm; G[N] = g; MV[N] = mv
        print(f"N{N}: {len(pr)} primary modes, {len(rw)} raw levels", flush=True)
    prim = pd.DataFrame(P_rows); raw = pd.DataFrame(R_rows); fam = pd.DataFrame(F_rows)
    # ---- physical de-duplication of primary copies across (sA,sB) blocks
    prim = prim.sort_values(["N", "C2", "L", "P", "Delta"]).reset_index(drop=True)
    copy_tol = cfg.get("copy_match_tol", 0.02); phys = [None] * len(prim)
    for k, g in prim.groupby(["N", "C2", "L", "P"]):
        clusters = []
        for blk, h in sorted(g.groupby("block"), key=lambda x: -len(x[1])):
            idx = list(h.index); Dv = h.Delta.values; used = set()
            if clusters:
                C = np.array([[abs(d - c[0]) for c in clusters] for d in Dv])
                for i, j in hungarian(C):
                    if C[i, j] <= copy_tol:
                        phys[idx[i]] = clusters[j][1]; used.add(i)
            for i in [i for i in range(len(idx)) if i not in used]:
                r = prim.loc[idx[i]]; pidp = f"N{r.N}:C{r.C2}:L{r.L}:P{r.P:+d}:{r.Delta:.6f}"
                clusters.append([r.Delta, pidp]); phys[idx[i]] = pidp
    prim["phys_id"] = phys
    copies = prim.groupby("phys_id").agg(n_copies=("pid", "size"), copy_Delta_spread=("Delta", lambda x: x.max() - x.min()))
    # v0.3.2 (A6): keep one complete source row per key (groupby.first() fills each column independently)
    # v0.3.5: stable sort, so ties in sigma_Delta keep the (N, C2, L, P, Delta) order instead of quicksort order
    rep_prim = prim.sort_values("sigma_Delta", kind="stable").drop_duplicates("phys_id", keep="first")
    rep_prim = rep_prim[["phys_id"] + [c for c in rep_prim.columns if c != "phys_id"]].reset_index(drop=True).merge(copies, on="phys_id")
    pid2phys = {(r.N, r.pid): r.phys_id for r in prim.itertuples()}
    raw["parent_phys_id"] = [pid2phys.get((n, p)) if p is not None and p == p else None for n, p in zip(raw.N, raw.parent_pid)]
    raw.loc[~raw.cls.eq("descendant"), ["parent_phys_id"]] = None
    raw["phys_level_key"] = [f"N{n}:C{c}:L{l}:P{p:+d}:{e:.7f}" for n, c, l, p, e in zip(raw.N, raw.C2, raw.L, raw.P, raw.energy)]
    raw["_r"] = raw.cls.map(CLS_ORDER)
    agg = raw.groupby("phys_level_key").agg(n_copies=("cls", "size"), classes=("cls", lambda x: "|".join(sorted(set(x)))),
                                            w_primary_min=("w_primary", "min"), w_primary_max=("w_primary", "max"))
    raw_phys = raw.sort_values(["_r", "w_primary"], ascending=[True, False]).drop_duplicates("phys_level_key", keep="first")
    raw_phys = raw_phys[["phys_level_key"] + [c for c in raw_phys.columns if c != "phys_level_key"]].reset_index(drop=True)
    raw_phys = raw_phys.merge(agg, on="phys_level_key").drop(columns=["_r"])
    raw_phys["copy_class_consistent"] = ~raw_phys.classes.str.contains(r"\|")
    raw_phys["gate_band"] = ((raw_phys.w_primary > 0.8) & (raw_phys.w_primary < 0.95)) | \
                            ((raw_phys.w_descendant > 0.8) & (raw_phys.w_descendant < 0.95))
    # ---- step 4: tracks
    pts = [dict(N=int(r.N), sector=(int(r.C2), int(r.L), int(r.P)), Delta=float(r.Delta), phys_id=r.phys_id,
                sigma_Delta=float(r.sigma_Delta), kappa=float(r.kappa), stable=bool(r.stable_kappa_x3))
           for r in rep_prim.itertuples() if r.Delta <= Dmax + 0.25]
    tracks = build_tracks(pts, sizes, **cfg["tracking_rule"])
    trows, pmap = [], {}
    for t in tracks:
        c2, L, Pp = t["sector"]; Ns = [p["N"] for p in t["points"]]; Ds = [p["Delta"] for p in t["points"]]
        f = fit_series(Ns, Ds, cfg)
        for p in t["points"]:
            pmap[p["phys_id"]] = t["track_id"]
        trows.append(dict(track_id=t["track_id"], C2=c2, rep=rep_name(c2), L=L, P=Pp, Ns=";".join(map(str, Ns)), nN=len(Ns),
                          any_ambiguous=any(p["ambiguous"] for p in t["points"]),
                          max_match_cost=float(np.nanmax([p["match_cost"] for p in t["points"]] + [0])),
                          max_sigma_Delta=max(p["sigma_Delta"] for p in t["points"]),
                          all_kappa_stable=all(p["stable"] for p in t["points"]),
                          in_window_all_N=all(d <= Dmax for d in Ds),
                          **{f"Delta_N{n}": d for n, d in zip(Ns, Ds)}, **{f"fit_{k}": v for k, v in f.items()}))
    trk = pd.DataFrame(trows)
    trk = trk.sort_values(["C2", "L", "P", "Delta_N10" if "Delta_N10" in trk else "track_id"]).reset_index(drop=True)
    names = {}
    for (c2, L, Pp), g in trk.groupby(["C2", "L", "P"], sort=False):
        for k, tid in enumerate(g.track_id, 1):
            names[tid] = f"{g.rep.iloc[0]}|L{L}|P{Pp:+d}|b{k}"
    trk.insert(1, "blind_name", trk.track_id.map(names))
    rep_prim["track_id"] = rep_prim.phys_id.map(pmap); rep_prim["blind_name"] = rep_prim.track_id.map(names)
    raw_phys["parent_track"] = raw_phys.parent_phys_id.map(pmap); raw_phys["parent_blind_name"] = raw_phys.parent_track.map(names)
    dom = {(r.N, r.block, r.L, r.dominant_index): r.phys_id for r in prim.itertuples()}
    raw_phys["primary_mode_phys_id"] = [dom.get((n, b, l, i)) for n, b, l, i in zip(raw_phys.N, raw_phys.block, raw_phys.L, raw_phys.block_index)]
    raw_phys["primary_track"] = raw_phys.primary_mode_phys_id.map(pmap); raw_phys["primary_blind_name"] = raw_phys.primary_track.map(names)
    # insertion operators identified by quantum numbers only: phi = lowest C2=2 L0, T = lowest C2=5 L0,
    # S = lowest non-vacuum C2=0 L0 (v0.5.1: no parity is assumed; the parity of each pick is measured, see gates)
    def lowest(c2, skip_zero=False):
        g = trk[(trk.C2 == c2) & (trk.L == 0) & (trk.nN == len(sizes))]
        g = g[g.Delta_N10 > 1e-6] if skip_zero else g
        return g.sort_values("Delta_N10").track_id.iloc[0]
    ins_track = {"phi": lowest(2), "T": lowest(5), "S": lowest(0, True)}
    # ---- step 6a: exact recontraction (integer sector)
    rows_prod = opem.load_rows(P(cfg["ope_per_N_rows"]))
    full_rows = pd.read_csv(P(cfg["ope_per_N_rows"]), low_memory=False)
    # v0.3.1 blind guard: the release rows carry paper-derived columns (X_paper_PD, Z_paper_PD); drop every such
    # column before anything is used, so no published label can enter the blind analysis.
    full_rows = full_rows[[c for c in full_rows.columns if "paper" not in c.lower()]]
    cgt, spin, Zt = ox.tables_from_rows(full_rows)
    # v0.3.4: sizes without release rows (N11) take Z from their ope3.toml; for the others the release Z must agree
    import re as _re
    for N in sizes:
        t = open(os.path.join(P(cfg["physics"]), f"N{N}", "ope3.toml")).read()
        g_ = lambda pat: float(_re.search(pat, t, _re.M).group(1))
        z3 = {"gamma2": g_(r"^phi_normalization = (\S+)"), "tensorG": g_(r"^Z_T = (\S+)"), "singlet": g_(r"^Z_S_raw = (\S+)")}
        for ins, z in z3.items():
            if (N, ins) in Zt:
                if abs(float(Zt[(N, ins)]) - z) > 1e-12 * max(1.0, abs(z)):
                    sys.exit(f"release Z differs from ope3.toml: N{N} {ins} {Zt[(N, ins)]} vs {z}")
            else:
                Zt[(N, ins)] = z
    ex_rows, reg_rows = [], []
    for N in sizes:
        od = ox.OpeData(os.path.join(P(cfg["ope_export"]), f"N{N}"), os.path.join(P(cfg["physics"]), f"N{N}"), None)
        mv = [dict(m, phys_id=pid2phys[(N, m["pid"])]) for m in MV[N] if m is not None]
        o, rg = ox.recontract(N, sds[N], od, mv, cgt, spin, Zt, Dmax)
        ex_rows += o; reg_rows += rg
    ex = pd.DataFrame(ex_rows); reg = pd.DataFrame(reg_rows)
    pi = full_rows[full_rows.namespace == "integer_KK"][["N", "sector", "insertion", "rank", "X_label", "Z_label", "me_1", "me_2"]]
    # v0.3.4: sizes without release rows are regressed against their cluster fullQ export rows (same me_1/me_2 source)
    for N, path in sorted(cfg.get("ope_fullq_rows_extra", {}).items()):
        fq = pd.read_csv(P(path), low_memory=False)
        pi = pd.concat([pi, fq[fq.N == int(N)][["N", "sector", "insertion", "rank", "X_label", "Z_label", "me_1", "me_2"]]], ignore_index=True)
    pi = pi.assign(X_label="kk_" + pi.X_label.str.replace("kk_", "", regex=False))
    m = pi.merge(reg, on=["N", "sector", "insertion", "rank", "X_label", "Z_label"], suffixes=("_rel", "_rec"))
    dd = np.maximum(np.abs(np.abs(m.me_1_rel) - np.abs(m.me_1_rec)), np.abs(np.abs(m.me_2_rel) - np.abs(m.me_2_rec)))
    ope_regression = dict(release_rows=len(pi), matched=len(m), max_abs_diff=float(dd.max()),
                          max_rel_diff=float((dd / np.maximum(1e-3, np.maximum(np.abs(m.me_1_rel), np.abs(m.me_2_rel)))).max()))
    if ope_regression["matched"] != ope_regression["release_rows"] or ope_regression["max_rel_diff"] > 1e-8:
        sys.exit(f"OPE regression gate failed: {ope_regression}")
    ex["namespace"] = "integer_exact_blind_vectors"; ex["route"] = ex.insertion.map(ROUTE)
    ex["X_track"] = ex.X_phys_id.map(pmap); ex["Z_track"] = ex.Z_phys_id.map(pmap)
    ex["X_blind_name"] = ex.X_track.map(names); ex["Z_blind_name"] = ex.Z_track.map(names)
    ex["Z_rep"] = ex.Z_C2.map(rep_name)
    ex["lambda_3j_blind"] = ex.lambda_blind * np.sqrt(ex.Z_C2.map(dim_of))
    bose = [(xt == ins_track[rt]) and ((rk % 2 == 0) == (zr in ANTISYM[rt])) for xt, rt, rk, zr in zip(ex.X_track, ex.route, ex["rank"], ex.Z_rep)]
    ex["Bose_forbidden"] = bose
    ex["odd_tensor_unverified"] = (ex.insertion == "tensorG") & (ex["rank"] % 2 == 1)   # v0.3.1: values use (-1)^l; publication policy unchanged (release: global odd-l certification off)
    ex["publishable"] = ex.cg_status.eq("supported") & ex.parity_allowed & ~ex.Bose_forbidden & ~ex.odd_tensor_unverified & ex.X_track.notna() & ex.Z_track.notna()
    # ---- step 6b: half-integer sector (production vectors; identity by energy only)
    half = rows_prod[rows_prod.namespace == "half_projected_local"].copy()
    allmodes = prim[["N", "C2", "L", "P", "Delta", "phys_id"]].copy(); allmodes["track_id"] = allmodes.phys_id.map(pmap)
    half = opem.coefficient(opem.match_states(half, allmodes))
    half["namespace"] = "half_production_vectors_energy_match"
    half["X_blind_name"] = half.X_track.map(names); half["Z_blind_name"] = half.Z_track.map(names)
    half["publishable"] = half.X_track.notna() & half.Z_track.notna() & half.normalization_status.eq("ok") & \
                          half.physical_publication_allowed.astype(str).eq("True")
    # ---- stageB sectors (v0.3): half-integer windows and extended (1,1) windows, blind KK + exact OPE
    SB_rows, SB_raw, SB_prim, SB_gates, xsec = [], [], [], {}, []
    imgn = ox.load_imagenorm(imagenorm_path) if cfg.get("stageB") else None
    rep_by = {k: g for k, g in rep_prim.groupby(["N", "C2", "L", "P"])}
    rawi = raw_phys.copy()
    for spec in cfg.get("stageB", []):
        sec, Ns_ = spec["sector"], spec["sizes"]
        for N in Ns_:
            sdb = sb.StageBData(os.path.join(P(spec["export"]), f"N{N}"), N, sec)
            sdb.assign_parity(lambda L, c, ps, N=N: ps * (-1) ** (N + L))
            gk = dict(gauge=sb.gauge_check(sdb), kk_regression=sb.kk_regression(sdb))
            if gk["kk_regression"]["modes_reproduced"] != gk["kk_regression"]["production_modes"] or \
               gk["kk_regression"]["max_one_minus_overlap"] > 1e-9 or gk["gauge"]["conjugacy_max_rel_residual"] > 1e-10:
                sys.exit(f"stageB gate failed {sec} N{N}: {gk}")
            pr_, rw_, fm_, g_, mv_ = analyse_size(sdb, cfg, thr, None)
            DS_conf += [dict(r, sector=sec) for r in g_.pop("ds_conflicts", [])]
            DS_drop += [dict(r, sector=sec) for r in g_.pop("ds_dropped", [])]
            prS = pd.DataFrame(pr_); rwS = pd.DataFrame(rw_)
            # parity consistency with integer copies (same energy, C2, L)
            rN = rawi[rawi.N == N]; bad = tot = 0
            for r in rwS.itertuples():
                g = rN[(rN.L == r.L) & (rN.C2 == r.C2) & (np.abs(rN.energy - r.energy) < 1e-7)]
                if len(g):
                    tot += 1; bad += int((g.P != r.P).any())
                    xsec.append(dict(N=N, sector=sec, L=r.L, rep=r.rep, P=r.P, Delta=r.Delta, cls_stageB=r.cls,
                                     w_primary_stageB=r.w_primary, w_descendant_stageB=r.w_descendant,
                                     cls_integer=g.iloc[0].cls, w_primary_integer=g.iloc[0].w_primary,
                                     w_descendant_integer=g.iloc[0].w_descendant))
            gk["parity_vs_integer"] = dict(matched_levels=tot, mismatches=bad)
            if bad:
                sys.exit(f"stageB parity rule disagrees with integer parity ({sec} N{N}): {bad}/{tot}")
            # map stageB primaries to integer physical primaries (Hungarian, |dDelta| <= copy_tol)
            prS["phys_id"] = None; prS["dDelta_to_integer"] = np.nan
            for k, g in prS.groupby(["N", "C2", "L", "P"]):
                h = rep_by.get(k)
                if h is None:
                    continue
                C = np.abs(g.Delta.values[:, None] - h.Delta.values[None, :])
                for i, j in hungarian(C):
                    if C[i, j] <= copy_tol:
                        prS.loc[g.index[i], "phys_id"] = h.phys_id.values[j]; prS.loc[g.index[i], "dDelta_to_integer"] = C[i, j]
            prS["sector"] = sec; rwS["sector"] = sec
            pid2 = dict(zip(prS.pid, prS.phys_id))
            mvw = [dict(m, phys_id=pid2.get(m["pid"])) for m in sb.blind_modes_window_coords(sdb, mv_)]
            mvw = [m for m in mvw if m["phys_id"] is not None]
            prod = pd.read_csv(P(spec["production_rows"]).format(N=N))
            od = ox.OpeData(os.path.join(P(cfg["ope_export"]), f"N{N}"), os.path.join(P(cfg["physics"]), f"N{N}"), None)
            rtd = spec.get("rawtarget")
            rawt = ox.load_rawtarget(P(rtd.format(N=N))) if rtd and N in spec.get("rawtarget_sizes", []) else None
            o, rg, order = ox.recontract_stageb(N, sdb, od, mvw, prod, cgt, spin, Zt, Dmax, spec["cg_sector"],
                                                imagenorm=imgn, imagenorm_sector={"half": "half", "11ext": "11"}[sec],
                                                rel_bound_max=cfg.get("truncation_rel_bound_max", 0.01), rawtarget=rawt)
            rg = pd.DataFrame(rg)
            mm = prod[["insertion", "rank", "X_label", "Z_label", "me_1", "me_2"]].merge(rg, on=["insertion", "rank", "X_label", "Z_label"], suffixes=("_rel", "_rec"))
            dd = np.maximum(np.abs(mm.me_1_rel - mm.me_1_rec), np.abs(mm.me_2_rel - mm.me_2_rec))
            gk["ope_regression"] = dict(production_rows=len(prod), matched=len(mm), max_signed_abs_diff=float(dd.max()) if len(dd) else None,
                                        source_order={str(k): v for k, v in order.items()})
            if len(mm) != len(prod) or (len(dd) and dd.max() > 1e-10):
                sys.exit(f"stageB OPE regression failed {sec} N{N}: {gk['ope_regression']}")
            for r in o:
                r["namespace"] = f"{sec}_exact_blind_vectors"
            SB_rows += o; SB_prim.append(prS); SB_raw.append(rwS); SB_gates[f"{sec}_N{N}"] = gk
            print(f"stageB {sec} N{N}: primaries {len(prS)}, ope rows {len(o)}", flush=True)
    sbx = pd.DataFrame(SB_rows)
    if len(sbx):
        sbx["route"] = sbx.insertion.map(ROUTE); sbx["X_track"] = sbx.X_phys_id.map(pmap); sbx["Z_track"] = sbx.Z_phys_id.map(pmap)
        sbx["X_blind_name"] = sbx.X_track.map(names); sbx["Z_blind_name"] = sbx.Z_track.map(names); sbx["Z_rep"] = sbx.Z_C2.map(rep_name)
        sbx["Bose_forbidden"] = [(xt == ins_track[rt]) and ((rk % 2 == 0) == (zr in ANTISYM[rt])) for xt, rt, rk, zr in zip(sbx.X_track, sbx.route, sbx["rank"], sbx.Z_rep)]
        sbx["odd_tensor_unverified"] = (sbx.insertion == "tensorG") & (sbx["rank"] % 2 == 1)
        sbx["publishable"] = sbx.cg_status.eq("supported") & sbx.parity_allowed & ~sbx.Bose_forbidden & ~sbx.odd_tensor_unverified & \
            sbx.X_track.notna() & sbx.Z_track.notna() & \
            sbx.contraction_status.isin(["exact_raw_window", "exact_in_saved_span", "truncated_certified"])  # v0.3.2 (A1), v0.3.3
        if "lambda_blind" not in sbx:
            sbx["lambda_blind"] = np.nan
    half_ex = sbx[sbx.namespace == "half_exact_blind_vectors"].copy() if len(sbx) else pd.DataFrame()
    ext11 = sbx[sbx.namespace == "11ext_exact_blind_vectors"].copy() if len(sbx) else pd.DataFrame()
    if len(ext11):
        ext11["sector"] = "s1_s1_ext"
        ex = pd.concat([ex, ext11], ignore_index=True)

    # ---- curves
    cur = []
    def add_curves(df, valcol, keys, label):
        """One value per (curve, N).  Integer sector: the zonal s0_s0 component defines the phase convention for
        signed diagonal channels; when s0_s0 is absent the magnitude of the first available sector is used.
        Cross-sector |lambda| agreement is recorded as a diagnostic."""
        for k, g in df[df.publishable & np.isfinite(df[valcol].astype(float))].groupby(keys):
            Ns, vals, spread, secs, signs_ok, tb = [], [], [], [], True, []
            phase = g.phase_convention.iloc[0]
            for N, h in g.groupby("N"):
                if "sector" in h.columns and (label.startswith("integer") or label.startswith("half_exact")):
                    h = h.assign(_p=h.sector.map(lambda x: 0 if x == "s0_s0" else (2 if x == "s1_s1_ext" else 1))).sort_values("_p")
                    first = h.iloc[0]
                    v = float(first[valcol])
                    if phase.startswith("signed") and first.sector not in ("s0_s0", "half"):
                        phase = "magnitude_non_zonal_sector"   # v0.3.1: applied to every N below, not only this one
                    secs.append(first.sector)
                    tb.append(float(first["truncation_bound_lambda"]) if "truncation_bound_lambda" in h.columns and
                              pd.notna(first.get("truncation_bound_lambda")) else 0.0)
                    signs_ok &= bool(np.all(np.sign(h[valcol]) == np.sign(first[valcol])))
                    spread.append(float(np.abs(h[valcol]).max() - np.abs(h[valcol]).min()))
                else:
                    v = float(np.median(h[valcol])); spread.append(float(h[valcol].max() - h[valcol].min()))
                Ns.append(int(N)); vals.append(v)
            if phase == "magnitude_non_zonal_sector":
                vals = [abs(x) for x in vals]
            f = fit_series(Ns, vals, cfg); r = dict(zip(keys, k))
            r.update(source=label, Ns=";".join(map(str, Ns)), nN=len(Ns), phase_convention=phase,
                     sector_used=";".join(sorted(set(secs))) if secs else "half",
                     cross_sector_abs_spread=max(spread), cross_sector_signs_consistent=signs_ok if secs else np.nan,
                     max_truncation_bound_lambda=max(tb) if tb else 0.0,
                     **{f"lambda_iso_N{n}": v for n, v in zip(Ns, vals)}, **{f"iso_{a}": b for a, b in f.items()})
            r["physical_status"] = "negative_magnitude_intercept_diagnostic_only" if (phase.startswith("magnitude") and f.get("intercept", 1) < 0) else "finite_window_estimate"
            cur.append(r)
    add_curves(ex, "lambda_blind", ["route", "rank", "X_track", "Z_track"], "integer_exact_blind_vectors")
    half["phase_convention"] = np.where(half.phase_convention.eq("signed_unique_invariant"),
                                        "signed_production_half_sector_phase", half.phase_convention)
    if len(half_ex):
        add_curves(half_ex, "lambda_blind", ["route", "rank", "X_track", "Z_track"], "half_exact_blind_vectors")
    else:
        add_curves(half, "lambda_iso_blind", ["route", "rank", "X_track", "Z_track"], "half_production_vectors_energy_match")
    cur = pd.DataFrame(cur)
    if len(cur):
        cur.insert(3, "X_blind_name", cur.X_track.map(names)); cur.insert(5, "Z_blind_name", cur.Z_track.map(names))
    # ---- sensitivity (standard output)
    sscan, border = sens.scan(sds, pars, cfg, thr, P(cfg["kk_export"]))
    tscan = sens.tracking_scan(pts, sizes, cfg["tracking_rule"])
    # ---- gates
    anchors = {}
    for N in sizes:
        pN = rep_prim[rep_prim.N == N]
        def near(c2, L, D):
            g = pN[(pN.C2 == c2) & (pN.L == L)]
            return None if g.empty else g.iloc[int(np.argmin(np.abs(g.Delta - D)))][["Delta", "kappa", "Pnorm_ratio"]].to_dict()
        # vacuum, J and stress sit at their calibration conditions (Delta = 0, 2, 3 by construction);
        # phi and T are the lowest primaries of their sector (v0.5.1: no expected Delta is used)
        anchors[N] = dict(vacuum=near(0, 0, 0), J_current=near(3, 1, 2.0), stress=near(0, 2, 3.0), phi=near(2, 0, 0.0), T=near(5, 0, 0.0))
    gates = dict(
        kappa_threshold=thr, kappa_rule=thr_info, max_sector_leak={N: G[N]["max_sector_leak"] for N in sizes},
        window_gauge_check={N: sds[N].gauge_check for N in sizes},
        target_source_windows={N: sorted(k for k, b in sds[N].blocks.items() if b.get("source_window") == "target") for N in sizes},
        windows=[w for N in sizes for w in G[N]["windows"] if w["extended"] or w["complete_to_Delta"] < Dmax],
        max_copy_Delta_spread=float(copies.copy_Delta_spread.max()),
        copy_spread_quantiles=copies.copy_Delta_spread.quantile([.5, .9, .99, 1]).tolist(),
        raw_copy_class_inconsistent=int((~raw_phys.copy_class_consistent).sum()),
        ope_regression=ope_regression, stageB=SB_gates,
        cross_sector_class_agreement=(None if not xsec else dict(
            levels=len(xsec), same_class=int(sum(x["cls_stageB"] == x["cls_integer"] for x in xsec)),
            primary_vs_descendant_conflicts=int(sum({x["cls_stageB"], x["cls_integer"]} == {"primary", "descendant"} for x in xsec)))),
        insertion_tracks={k: names[v] for k, v in ins_track.items()},
        anchors=anchors,
        anchor_roles=dict(calibration_conditions=["Delta_J = 2 (energy scale)", "Delta_stress = 3 (coupling g)",
                                                   "|P S| = sqrt(2 Delta_S) (generator normalisation)", "vacuum image minimal (generator ratio)"],
                          nontrivial_checks=["kappa(J) = 0, kappa(stress) = 0, kappa(phi) = 0 (primary by KK)",
                                             "P-norm closure sum|P O|^2 / 6 Delta -> 1 for phi, T, stress (S excluded: calibrated)",
                                             "reduced Hermitian conjugacy (sign fixing) consistency",
                                             "K2 regression vs production", "OPE matrix-element regression vs production"]))
    # ---- write
    W = lambda df, f: df.to_csv(os.path.join(out, f), index=False)
    W(prim, "primary_modes_all_copies.csv"); W(rep_prim, "primaries_per_N.csv")
    W(raw.drop(columns=["_r"]), "raw_levels_all_copies.csv"); W(raw_phys, "raw_levels_classified.csv")
    W(fam, "family_spans.csv"); W(trk, "primary_tracks.csv")
    W(ex, "ope_rows_integer_exact.csv"); W(half, "ope_rows_half_production.csv"); W(cur, "ope_curves.csv")
    if len(sbx):
        W(sbx, "ope_rows_stageB_exact.csv")
    if SB_prim:
        W(pd.concat(SB_prim, ignore_index=True), "stageB_primaries.csv"); W(pd.concat(SB_raw, ignore_index=True), "stageB_raw_levels.csv")
    xs = pd.DataFrame(xsec); W(xs, "cross_sector_classification.csv")
    W(nm.per_N_constants(sds, P(cfg["physics"]), thr, sizes), "normalization_constants.csv")
    W(nm.cg_table(full_rows), "cg_table.csv"); W(nm.convention_factors(), "convention_factors.csv")
    W(sscan, "sensitivity_scan.csv"); W(border, "borderline_levels.csv"); W(tscan, "tracking_sensitivity.csv")
    if cfg.get("descendant_span"):
        W(pd.DataFrame(DS_conf), "descendant_span_conflicts.csv"); W(pd.DataFrame(DS_drop), "descendant_span_dropped.csv")
    json.dump(gates, open(os.path.join(out, "gates.json"), "w"), indent=1, default=float)
    inputs = {f"N{N}/{k}": v for N in sizes for k, v in sds[N].input_hashes.items()}
    for N in sizes:
        inputs[f"N{N}/ope_meta.json"] = sha256_file(os.path.join(P(cfg["ope_export"]), f"N{N}", "ope_meta.json"))
    inputs["ope_per_N_rows"] = sha256_file(P(cfg["ope_per_N_rows"]))
    input_tree_end = tree_hashes(input_roots)
    if input_tree_end != input_tree_start:
        sys.exit("input files changed during the run")
    rel = lambda a: os.path.relpath(a, base).replace(os.sep, "/")
    missing = [k for k in READ_LOG if k not in input_tree_end]
    if missing:
        sys.exit(f"binary inputs read outside the hashed input roots: {missing[:3]}")
    man = dict(pipeline_version=__version__, config=cfg, config_sha256=sha256_file(cfg_path), code=code_manifest(root),
               inputs=inputs, input_files_sha256={rel(k): v for k, v in sorted(input_tree_end.items())},
               verified_binaries=len(READ_LOG), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               created=datetime.datetime.now().isoformat(timespec="seconds"), reads_published_tables=False,
               note="No published spectrum/OPE table was read by this run.")
    man["outputs"] = {f: sha256_file(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f != "MANIFEST.json"}
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1, default=str)
    print("kappa_threshold", thr, "tracks", len(trk), "ope curves", len(cur), "ope regression", ope_regression)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
