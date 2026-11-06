"""Step 6 (v0.2): exact recontraction of three-point matrix elements with blind primary vectors.

Input: fullQ insertion edges  overlaps_p[t, j] = <t, block (sector, L=rank) || O_p^(rank) || source mode j>,
t = raw eigenstates of the run block (same gauge as the KK export), j = production source modes
(KK threshold K2 <= 0.25, scalar, Delta <= 5).  Chain (reproduces the production reduction exactly):
  gamma2 (phi) : me = me_1
  tensorG (T)  : me = me_1 + (-1)^l me_2        (v0.3.1: odd-l pairing sign, cross-sector covariance 3.8e-14)
  singlet (S)  : me = beta_1 me_1 + beta_2 me_2 - <O_S>_vac <Z|X>,  then x [P_X P_Z = +1]   (S_even projection)
  lambda_iso   = c_l * me / (Z_O * CG_iso),  Z_O = |Z| for phi and T, Z_O = Z_S_raw (signed, -1) for S
                 (v0.3.1: O_S-hat = O_S / Z_S_raw with the S phase anchored by <phi|n0|S> > 0, as ope3.jl)
Selection rule (all ranks): phi insertion needs P_X P_Z = -1; S and T insertions need P_X P_Z = +1.
"""
import json, os, re
import numpy as np
import pandas as pd
from .kk import block_kk
from .io import read_f64

INS_P = {"gamma2": -1, "singlet": 1, "tensorG": 1}


def production_modes(sd, a, b, L, thr=0.25):
    """Production KK modes (frozen kk_classify): per C2 channel, K2 eigen <= thr, H diagonalised, ascending."""
    blk = block_kk(sd, a, b, L); n = len(blk["E"]); out = {}
    for ch in blk["channels"]:
        if "K2" not in ch:
            continue
        w, U = np.linalg.eigh(ch["K2"]); Q = U[:, w <= thr]
        if Q.shape[1] == 0:
            continue
        ev, V = np.linalg.eigh(Q.T @ np.diag(blk["E"][ch["guarded"]]) @ Q); Q = Q @ V
        F = np.zeros((n, Q.shape[1])); F[ch["guarded"], :] = Q
        out[ch["C2"]] = (F, ev)
    return out, blk


class OpeData:
    def __init__(self, export_dir, physics_dir, rows_path):
        self.meta = json.load(open(os.path.join(export_dir, "ope_meta.json")))
        self.dir = export_dir
        t = open(os.path.join(physics_dir, "ope3.toml")).read()
        self.beta = [float(x) for x in re.search(r"^beta = \[([^\]]+)\]", t, re.M).group(1).split(",")]
        self.vac = float(re.search(r"^vacuum_expectation_Os = (\S+)", t, re.M).group(1))

    def parts(self, e):
        raw = read_f64(os.path.join(self.dir, e["bin"]), e.get("bin_sha256"))
        nt, ns, P = e["nt"], e["ns"], e["parts"]
        if raw.size != P * nt * ns:
            raise ValueError("bad ope edge size " + e["name"])
        return [raw[p * nt * ns:(p + 1) * nt * ns].reshape((nt, ns), order="F") for p in range(P)]


def z_den(Zt, N, ins):
    """Normalisation denominator: |Z| for phi and T; signed Z_S_raw for the S insertion (O_S-hat = O_S/Z_S_raw)."""
    z = float(Zt[(N, ins)])
    return z if ins == "singlet" else abs(z)


def tables_from_rows(rows):
    """Group-theory / normalisation constants taken from the production per-N rows (not physics results)."""
    i = rows[rows.namespace == "integer_KK"]
    a = rows                                   # CG keys carry the sector ('half' for the half-integer sector)
    sup = a[a.cg_status == "supported"]
    cg = sup.groupby(["insertion", "sector", "rank", "X_C2", "Z_C2"]).cg_iso.agg(lambda x: sorted(set(np.round(x, 12))))
    spin = i.dropna(subset=["spin_factor"]).groupby(["insertion", "rank"]).spin_factor.first().to_dict()
    Z = i.groupby(["N", "insertion"]).normalization_Z.first().to_dict()
    st = a.groupby(["insertion", "sector", "rank", "X_C2", "Z_C2"]).cg_status.agg(lambda x: "|".join(sorted(set(str(v) for v in x if isinstance(v, str)))) or "CG_status_missing")
    cgd = cg.to_dict()
    sg = a.dropna(subset=["cg_iso_signed"]).groupby(["insertion", "sector", "rank", "X_C2", "Z_C2"]).cg_iso_signed.agg(
        lambda x: sorted(set(np.round(x, 12))))
    cgd["__signed__"] = {k: v[0] for k, v in sg.items() if len(v) == 1}
    for k, v in st.items():
        if k not in cgd:
            cgd[k] = v                      # e.g. 'CG_representation_forbidden', 'CG_unsupported', 'CG_component_zero'
    return cgd, spin, Z


def me_value(ins, parts_vals, od, overlap_zx, PX, PZ, rank):
    if ins == "gamma2":
        me = parts_vals[0]
    elif ins == "tensorG":
        me = parts_vals[0] + (-1) ** int(rank) * parts_vals[1]
    else:
        me = od.beta[0] * parts_vals[0] + od.beta[1] * parts_vals[1] - od.vac * overlap_zx
        me = me if PX * PZ == 1 else 0.0
    return me


def recontract(N, sd, od, prim_modes, cgt, spin, Zt, Dmax):
    """prim_modes: list of dicts (block, L, C2, X, P, Delta, phys_id, vec (block coords)). Returns rows + regression."""
    out, reg = [], []
    by_block = {}
    for m in prim_modes:
        by_block.setdefault((m["block"], m["L"]), []).append(m)
    for e in od.meta["edges"]:
        idt = e["identity"]; sec = idt["sector"]; ins = idt["insertion"]; r = int(idt["rank"])
        a, b = int(sec[1]), int(sec[4])
        Ms = od.parts(e)
        srcm, sblk = production_modes(sd, a, b, 0)
        tgtm, tblk = production_modes(sd, a, b, r)
        if e["nt"] != len(tblk["E"]):
            raise ValueError(f"ope edge {e['name']} target size != block size")
        # source vectors from labels kk_C2_<c>_m<j>; the sign of the reconstruction is fixed to the gauge of
        # the cluster's source vector through the singlet insertion: <x|beta.O_S|x> = <O_S>_vac + O(1) > 0
        sing = [f for f in od.meta["edges"] if f["identity"]["sector"] == sec and f["identity"]["insertion"] == "singlet"
                and int(f["identity"]["rank"]) == 0]
        Ms0 = od.parts(sing[0]) if sing else None
        srcs = []
        for col, lab in zip(e["scols"], idt["source_labels"]):
            c, j = map(int, re.match(r"kk_C2_(\d+)_m(\d+)", lab).groups())
            F, ev = srcm[c]; x = F[:, j - 1].copy()
            if Ms0 is None:
                raise ValueError(f"no singlet r0 edge to fix source gauge in {sec}")
            diag = od.beta[0] * float(x @ Ms0[0][:, col - 1]) + od.beta[1] * float(x @ Ms0[1][:, col - 1])
            if abs(diag) < 0.5 * od.vac:
                raise ValueError(f"source gauge test inconclusive for {lab} in {sec}: {diag}")
            x *= np.sign(diag)
            srcs.append((col - 1, lab, c, x, sd.delta(ev[j - 1])))
        # regression: production targets
        for (col, lab, c, x, DX) in srcs:
            for cz, (F, ev) in tgtm.items():
                for jz in range(F.shape[1]):
                    z = F[:, jz]; vals = [float(z @ M[:, col]) for M in Ms]
                    reg.append(dict(N=N, sector=sec, insertion=ins, rank=r, X_label=lab, Z_label=f"kk_C2_{cz}_m{jz + 1}",
                                    me_1=vals[0], me_2=vals[1] if len(vals) > 1 else 0.0))
        # blind targets
        blind_src = by_block.get((sec, 0), [])
        for (col, lab, c, x, DX) in srcs:
            # identity of the source among blind primaries (vector overlap)
            ov = [(abs(float(x @ m["vec"])), m) for m in blind_src if m["C2"] == c]
            ovb = max(ov, key=lambda t: t[0]) if ov else (0.0, None)
            xm = ovb[1] if ovb[0] > 1 - 1e-8 else None
            PX = int(np.sign(np.sum(x ** 2 * tblk["P"]) if r == 0 else 0)) if False else None
            # source parity from its block (L0) states
            PX = int(round(float(np.sum(x ** 2 * sblk["P"]))))
            for m in by_block.get((sec, r), []):
                if m["Delta"] > Dmax:
                    continue
                z = m["vec"]
                same = xm is not None and m["phys_id"] == xm["phys_id"]
                if same and r == 0:                     # same state: align the eigenvector gauge, z -> sign(z.x) z
                    z = z * np.sign(float(z @ x))
                vals = [float(z @ M[:, col]) for M in Ms]
                if len(vals) == 1:
                    vals.append(0.0)
                ozx = float(z @ x) if r == 0 else 0.0
                me = me_value(ins, vals, od, ozx, PX, m["P"], r)
                allowed = (PX * m["P"] == INS_P[ins])
                key = (ins, sec, r, c, m["C2"]); cgs = cgt.get(key)
                row = dict(N=N, sector=sec, insertion=ins, rank=r, X_label=lab, X_C2=c, X_P=PX, X_Delta=DX,
                           X_source_overlap=ovb[0], X_phys_id=None if xm is None else xm["phys_id"],
                           Z_phys_id=m["phys_id"], Z_C2=m["C2"], Z_L=r, Z_P=m["P"], Z_Delta=m["Delta"],
                           me_1=vals[0], me_2=vals[1], me=me, overlap_ZX=ozx, parity_allowed=bool(allowed),
                           diagonal_same_state=bool(same))
                if cgs is None:
                    row.update(cg_status="CG_combination_not_in_production_table")
                elif isinstance(cgs, str):
                    row.update(cg_status=cgs)
                elif len(cgs) > 1:
                    row.update(cg_status="CG_multiplicity_or_ambiguous", cg_iso_candidates=";".join(map(str, cgs)))
                else:
                    cgv = cgs[0]; c_l = spin.get((ins, r)); Zo = z_den(Zt, N, ins)
                    if same and key in cgt["__signed__"]:      # certified signed CG for diagonal channels
                        cgv = cgt["__signed__"][key]
                    lam = c_l * me / (Zo * cgv)
                    row.update(cg_status="supported", cg_iso=cgv, cg_signed_used=bool(same and key in cgt["__signed__"]), spin_factor=c_l,
                               normalization_Z_abs=abs(Zo), normalization_Z_used=Zo,
                               lambda_iso_signed=lam, lambda_iso_magnitude=abs(lam),
                               lambda_blind=lam if same else abs(lam),
                               phase_convention="signed_diagonal_same_state" if same else "magnitude_phase_not_fixed")
                out.append(row)
    return out, reg


# --------------------------------------------------------------------------------------------- stageB (v0.3)
def stageb_sources(sdb, Dsrc_max=5.0):
    """Production L0 source modes (coefficient vectors exported verbatim from the cluster) with Delta <= 5."""
    out = []
    for md in sdb.raw_meta["kk_modes"]["L0"]["modes"]:
        D = float(sdb.delta(md["energy"]))
        if D <= Dsrc_max + 1e-9:
            out.append(dict(label=md["label"], c2=int(md["c2"]), P=int(md["P"]), Delta=D, vec=np.array(md["coeff"], float)))
    return out


def stageb_production_targets(sdb, L):
    return [dict(label=md["label"], c2=int(md["c2"]), P=int(md["P"]), Delta=float(sdb.delta(md["energy"])),
                 vec=np.array(md["coeff"], float)) for md in sdb.raw_meta["kk_modes"][f"L{L}"]["modes"]]


def stageb_edges(sdb):
    out = []
    for e in sdb.raw_meta["ope_edges"]:
        name = e["name"]                                  # <sector>_<insertion>_L<rank>
        mt = re.match(r"^[^_]+_(gamma2|singlet|tensorG)_L(\d+)$", name)
        if mt is None:                                    # e.g. half_raw_phi_J_control (control edge, not a source set)
            continue
        ins = mt.group(1); r = int(mt.group(2))
        raw = read_f64(os.path.join(sdb.export_dir, e["bin"]), e.get("bin_sha256")); nt, ns, P = e["nt"], e["ns"], e["parts"]
        Ms = [raw[p * nt * ns:(p + 1) * nt * ns].reshape((nt, ns), order="F") for p in range(P)]
        out.append(dict(name=name, insertion=ins, rank=r, Ms=Ms, scols=[int(c) for c in e["scols"]]))
    return out


def stageb_target_basis(sdb, L, Dt_max=6.6):
    """Target side of stageB insertion edges: production KK modes at L with Delta <= 6.6, in list order."""
    md = [m for m in sdb.raw_meta["kk_modes"][f"L{L}"]["modes"] if float(sdb.delta(m["energy"])) <= Dt_max + 1e-9]
    T = np.array([m["coeff"] for m in md], float).T                 # window coords x modes
    return T, [m["label"] for m in md]


def stageb_source_order(sdb, edges, prod_rows):
    """Source columns -> production labels, by regression of me_1 against the production rows (gamma2 edges)."""
    srcs = stageb_sources(sdb); best = {}
    for e in edges:
        if e["insertion"] != "gamma2":
            continue
        T, labs = stageb_target_basis(sdb, e["rank"])
        if T.shape[1] != e["Ms"][0].shape[0]:
            raise ValueError(f"target basis size mismatch in {e['name']}")
        pr = prod_rows[(prod_rows.insertion == "gamma2") & (prod_rows["rank"] == e["rank"])]
        for j in range(e["Ms"][0].shape[1]):
            for lab in pr.X_label.unique():
                q = pr[pr.X_label == lab].set_index("Z_label").me_1
                diffs = [abs(float(e["Ms"][0][t, j]) - float(q[zl])) for t, zl in enumerate(labs) if zl in q.index]
                if diffs:
                    best[(j, lab)] = min(best.get((j, lab), np.inf), max(diffs))
    order = {}
    for j in sorted(set(k[0] for k in best)):
        cand = sorted((v, lab) for (jj, lab), v in best.items() if jj == j)
        order[j] = cand[0][1] if cand and cand[0][0] < 1e-12 else None
    return order, srcs


def load_imagenorm(path):
    """Image norms of the stageB OPE edges (one row per sector, N, insertion, rank and source column: Gram, C and P norms):
    G[p,q] = <O_p x|O_q x> over the full Hilbert space, C[p,q] = the same Gram restricted to the saved target modes."""
    df = pd.read_csv(path); out = {}
    for r in df.itertuples():
        out[(r.sector, int(r.N), r.insertion, int(r.rank), int(r.source_col))] = (
            np.array([[r.G11, r.G12], [r.G12, r.G22]]), np.array([[r.C11, r.C12], [r.C12, r.C22]]), int(r.P))
    return out


def load_rawtarget(export_dir):
    """v0.3.3: Stage B supplement edges onto the FULL raw window (cluster xy_half_rawtarget.jl, rawtarget-v1).
    Returns {edge name: dict(Ms=[nt x ns parts], meta)}; every bin is verified against its recorded sha256."""
    import json
    m = json.load(open(os.path.join(export_dir, "rawtarget_meta.json"))); out = {}
    for e in m["edges"]:
        nt, ns, P = int(e["nt"]), int(e["ns"]), int(e["parts"])
        raw = read_f64(os.path.join(export_dir, e["bin"]), e["bin_sha256"])
        if raw.size != P * nt * ns:
            raise ValueError("bad rawtarget payload size " + e["name"])
        if max(e["regression_projection_rel"], e["regression_gram_rel"]) > 1e-10:
            raise ValueError("rawtarget edge failed its cluster regression gate " + e["name"])
        out[e["name"]] = dict(Ms=[raw[p * nt * ns:(p + 1) * nt * ns].reshape((nt, ns), order="F") for p in range(P)], meta=e)
    return out


def insertion_vector(ins, rank, od):
    """Coefficients c of the physical insertion O_c = sum_p c_p O_p (same chain as me_value)."""
    if ins == "gamma2":
        return np.array([1.0, 0.0])
    if ins == "tensorG":
        return np.array([1.0, (-1.0) ** int(rank)])
    return np.array(od.beta, float)


def recontract_stageb(N, sdb, od, prim_modes, prod_rows, cgt, spin, Zt, Dmax, cg_sector, span_tol=1e-8,
                      imagenorm=None, imagenorm_sector=None, rel_bound_max=0.01, rawtarget=None):
    """Sources: exported production L0 modes (cluster gauge).  Targets: blind primary z expanded in the orthonormal
    production target-mode basis, z = sum_k c_k q_k + residual;  me = sum_k c_k <q_k|O|x>  (exact when residual ~ 0).
    v0.3.2 (A1): the missing term <(1-P_T) z|O|x> is bounded by ||(1-P_T) z|| * ||(1-P_T) O x||, the second factor from
    the cluster image Gram (full Hilbert space) minus the captured Gram on the saved targets.  A row is
      exact_in_saved_span            rigorous bound exactly zero,
      truncated_certified            finite rigorous |d lambda| / |lambda| <= rel_bound_max,
      truncated_uncertified          otherwise, including a missing or non-finite bound (diagnostic only, never published).
    v0.4.1: the bound is no longer zeroed for residual <= span_tol (span_tol now only labels target_contraction)."""
    edges = stageb_edges(sdb)
    order, srcs = stageb_source_order(sdb, edges, prod_rows)
    lab2src = {s_["label"]: s_ for s_ in srcs}
    out, reg = [], []
    for e in edges:
        ins, r, Ms = e["insertion"], e["rank"], e["Ms"]
        T, labs = stageb_target_basis(sdb, r)
        rt = None if rawtarget is None else rawtarget.get(e["name"])
        if rawtarget is not None and rt is None:
            raise ValueError(f"rawtarget edge missing: {e['name']} N{N}")
        if rt is not None:
            W = sdb.windows[r]
            if len(W["E"]) != rt["Ms"][0].shape[0] or np.max(np.abs(np.asarray(W["E"]) - np.asarray(rt["meta"]["target_E"]))) > 1e-9:
                raise ValueError(f"rawtarget window/energy mismatch {e['name']} N{N}")
            if len(rt["Ms"]) != len(Ms) or rt["Ms"][0].shape[1] != Ms[0].shape[1]:
                raise ValueError(f"rawtarget shape mismatch {e['name']} N{N}")
            # consistency: production-mode projection of the raw edge reproduces the exported Stage B edge
            if max(float(np.max(np.abs(T.T @ Mr - M))) / max(float(np.max(np.abs(M))), 1e-300) for Mr, M in zip(rt["Ms"], Ms)) > 1e-9:
                raise ValueError(f"rawtarget projection != exported Stage B edge {e['name']} N{N}")
        for j in range(Ms[0].shape[1]):
            lab = order.get(j)
            if lab is None or lab not in lab2src:
                continue
            if rt is not None and rt["meta"]["source_labels"][j] != lab:
                raise ValueError(f"rawtarget source column order differs {e['name']} N{N} col {j}")
            x = lab2src[lab]["vec"]; c = lab2src[lab]["c2"]; PX = lab2src[lab]["P"]; DX = lab2src[lab]["Delta"]
            out_img = np.nan
            if imagenorm is not None and rt is None:      # v0.3.4: the bound is only needed without raw-target edges
                Gm, Cm, Pn = imagenorm[(imagenorm_sector, N, ins, r, j + 1)]
                Cx = np.zeros((2, 2))
                for a_ in range(len(Ms)):
                    for b_ in range(len(Ms)):
                        Cx[a_, b_] = float(Ms[a_][:, j] @ Ms[b_][:, j])
                if Pn != len(Ms) or np.max(np.abs(Cx - Cm)) > 1e-9 * max(1.0, np.max(np.abs(Cm))):
                    raise ValueError(f"imagenorm column mismatch {imagenorm_sector} N{N} {ins} r{r} col{j + 1}")
                cv = insertion_vector(ins, r, od)
                out_img = float(np.sqrt(max(cv @ (Gm - Cm) @ cv, 0.0)))
            for t, zl in enumerate(labs):
                reg.append(dict(N=N, insertion=ins, rank=r, X_label=lab, Z_label=zl, me_1=float(Ms[0][t, j]),
                                me_2=float(Ms[1][t, j]) if len(Ms) > 1 else 0.0))
            ov = [(abs(float(x @ m["vec"])), m) for m in prim_modes if m["L"] == 0 and m["C2"] == c]
            ovb = max(ov, key=lambda q: q[0]) if ov else (0.0, None)
            xm = ovb[1] if ovb[0] > 1 - 1e-8 else None
            for m in prim_modes:
                if m["L"] != r or m["Delta"] > Dmax:
                    continue
                z = m["vec"]; same = xm is not None and m["phys_id"] == xm["phys_id"]
                if same and r == 0:
                    z = z * np.sign(float(z @ x))
                # v0.4.1 (review R3): direct squared residual ||z - T T^T z||^2 (1 - ||T^T z||^2 cancels)
                cz = T.T @ z; resid = float(np.sum((z - T @ cz) ** 2))
                vals_proj = [float(cz @ M[:, j]) for M in Ms] + [0.0]
                vals = [float(z @ Mr[:, j]) for Mr in rt["Ms"]] + [0.0] if rt is not None else vals_proj
                ozx = float(z @ x) if r == 0 else 0.0
                me = me_value(ins, vals, od, ozx, PX, m["P"], r)
                key = (ins, cg_sector, r, c, m["C2"]); cgs = cgt.get(key)
                row = dict(N=N, sector=cg_sector, insertion=ins, rank=r, X_label=lab, X_C2=c, X_P=PX, X_Delta=DX,
                           X_source_overlap=ovb[0], X_phys_id=None if xm is None else xm["phys_id"], Z_phys_id=m["phys_id"],
                           Z_C2=m["C2"], Z_L=r, Z_P=m["P"], Z_Delta=m["Delta"], me_1=vals[0], me_2=vals[1], me=me,
                           overlap_ZX=ozx, parity_allowed=bool(PX * m["P"] == INS_P[ins]), diagonal_same_state=bool(same),
                           target_span_residual=resid,
                           target_contraction="exact_in_production_mode_span" if resid <= span_tol else "partial_span_truncated",
                           truncation_out_image_norm=out_img,
                           # v0.4.1 (review R3): the bound is kept for every saved-span row, however small resid is
                           truncation_bound_me=(0.0 if rt is not None else float(np.sqrt(resid)) * out_img),
                           me_projected_span=me_value(ins, vals_proj, od, ozx, PX, m["P"], r),
                           raw_window_contraction=rt is not None)
                if cgs is None:
                    row.update(cg_status="CG_combination_not_in_production_table")
                elif isinstance(cgs, str):
                    row.update(cg_status=cgs)
                elif len(cgs) > 1:
                    row.update(cg_status="CG_multiplicity_or_ambiguous", cg_iso_candidates=";".join(map(str, cgs)))
                else:
                    cgv = cgs[0]
                    if same and key in cgt["__signed__"]:
                        cgv = cgt["__signed__"][key]
                    Zo = z_den(Zt, N, ins)
                    lam = spin.get((ins, r)) * me / (Zo * cgv)
                    bl = spin.get((ins, r)) * row["truncation_bound_me"] / (abs(Zo) * abs(cgv))
                    # v0.4.1 (review R4): a missing / non-finite bound is no evidence and never certifies a row
                    rb = (np.nan if not np.isfinite(bl) else 0.0 if bl == 0 else bl / max(abs(lam), 1e-300))
                    if rt is not None:
                        status = "exact_raw_window"
                    elif not np.isfinite(rb):
                        status = "truncated_uncertified"
                    elif bl == 0:
                        status = "exact_in_saved_span"          # zero omitted weight or zero omitted image
                    else:
                        status = "truncated_certified" if rb <= rel_bound_max else "truncated_uncertified"
                    row.update(truncation_bound_lambda=bl, truncation_rel_bound=rb, contraction_status=status)
                    row.update(cg_status="supported", cg_iso=cgv, cg_signed_used=bool(same and key in cgt["__signed__"]),
                               spin_factor=spin.get((ins, r)), normalization_Z_abs=abs(Zo), normalization_Z_used=Zo,
                               lambda_iso_signed=lam, lambda_iso_magnitude=abs(lam), lambda_blind=lam if same else abs(lam),
                               phase_convention="signed_diagonal_same_state" if same else "magnitude_phase_not_fixed")
                out.append(row)
    return out, reg, order
