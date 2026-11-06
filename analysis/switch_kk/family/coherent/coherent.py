"""Coherent family check: signed P-image descendant subspaces and Loewdin decomposition of eigenstates.

Rules: protocol.md (this directory).  Inputs: v3 KK inputs (run.load against the v3 freeze) and v3 results
states.csv / modes.csv (checked against the v3 receipt).  No literature values, labels, or incoherent results."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
V3 = FAMILY.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(FAMILY))
import run  # noqa: E402
import family as fm  # noqa: E402
from spectrum import irrep  # noqa: E402

GATE = run.GATE
CODE_FILES = ("coherent.py", "protocol.md", "test_coherent.py")
UPSTREAM = {"family.py": FAMILY / "family.py", "spectrum.py": V3 / "spectrum.py"}
REUSED = ("reused_run_window", "block_top_frontier")
MAP_REL = 1e-9        # max|Phi_direct - Phi_reverse| / max|Phi|
SIGN_REL = 1e-6       # max|row - sigma * c * column| / max|edge|
SYM_REL = 1e-8        # max|Phi - Phi^T| / max|Phi| for Delta L = 0
ZERO_ROW = 1e-9       # rows below this (relative to max|edge|) carry no sign information
SV_EPS = 1e-8
LOWDIN_EPS = 1e-10
ANOMALY = 0.5
COVER_TOL = 1e-9
TEST_MARGIN = 0.5


def L_of(key):
    return int(key.rsplit("L", 1)[1])


# ----------------------------------------------------------------------------- linear algebra

def align_signs(A, B, rows, c, scale):
    """Guarded rows A[j,:] = sigma_t * c * B[:, t] with t = rows[j].  Returns {t: sigma}, max residual."""
    out, worst = {}, 0.0
    for j, t in enumerate(rows):
        if t < 0:
            continue
        x, y = A[j, :], c * B[:, t]
        if max(np.abs(x).max(), np.abs(y).max()) <= ZERO_ROW * scale:
            continue
        sigma = 1.0 if float(x @ y) >= 0 else -1.0
        r = float(np.abs(x - sigma * y).max()) / scale
        worst = max(worst, r)
        out[int(t)] = (sigma, r)
    return out, worst


def lowdin(Z):
    """Symmetric orthonormalisation Z O^{-1/2} on the non-null part of O = Z^T Z."""
    O = Z.T @ Z
    lam, U = np.linalg.eigh(O)
    keep = lam > LOWDIN_EPS
    Uk = U[:, keep]
    Zt = Z @ Uk @ np.diag(lam[keep] ** -0.5) @ Uk.T
    return Zt, float(lam.min()) if len(lam) else np.nan, int((~keep).sum())


def slot_basis(Y, m):
    """Leading min(m, numerical rank) left singular vectors of Y, all singular values."""
    if Y.size == 0:
        return np.zeros((Y.shape[0], 0)), np.zeros(0)
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    k = int(min(m, np.sum(s > SV_EPS)))
    return U[:, :k], s


def max_cos2(bases):
    worst = 0.0
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            if bases[i].shape[1] and bases[j].shape[1]:
                worst = max(worst, float(np.linalg.svd(bases[i].T @ bases[j], compute_uv=False)[0] ** 2))
    return worst


def orth(Z):
    if Z.shape[1] == 0:
        return Z
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, s > np.sqrt(LOWDIN_EPS)]


def label(rho):
    """rho: {(F, n): weight} for one eigenstate."""
    prim = sum(v for (F, n), v in rho.items() if n == 0)
    cover = sum(rho.values())
    ranked = sorted(rho.items(), key=lambda kv: -kv[1])
    desc = [(k, v) for k, v in ranked if k[1] >= 1]
    by_family = {}
    for (F, n), v in rho.items():
        if n >= 1:
            by_family[F] = by_family.get(F, 0.0) + v
    if prim >= GATE:
        tag = "primary"
    elif cover < GATE:
        tag = "uncovered"
    elif desc and desc[0][1] >= GATE:
        tag = "descendant"
    elif by_family and max(by_family.values()) >= GATE:
        tag = "family_mixed_level"
    else:
        tag = "mixed"
    return tag, ranked[:2], prim, cover


# ----------------------------------------------------------------------------- generator maps

def block_info(blocks, parity, alpha, e0):
    info = {}
    for key, b in blocks.items():
        rows = sorted(parity[key], key=lambda r: int(r["index"]))
        if [int(r["index"]) for r in rows] != list(range(1, len(b["energy"]) + 1)):
            raise ValueError(f"Parity rows do not index {key}")
        info[key] = dict(delta=alpha * (np.asarray(b["energy"]) - e0), c2=np.asarray(b["c2"], float),
                         x=np.asarray(b["x"]), p=np.array([int(r["P"]) for r in rows]), L=L_of(key))
    return info


def target_rows(item, info):
    tgt = item["target"]
    d, c2 = info[tgt]["delta"], info[tgt]["c2"]
    if item["edge"]["target_status"] in REUSED:
        if len(item["delta"]) != len(d):
            raise ValueError(f"Reused window size differs from block on {item['edge']['key']}")
        return np.arange(len(d))
    tc2 = np.asarray(item["edge"]["target_C2"])
    out = np.full(len(item["delta"]), -1)
    for j in range(len(out)):
        cand = np.flatnonzero((np.abs(d - item["delta"][j]) < fm.MATCH_DELTA) & (np.abs(c2 - tc2[j]) < fm.MATCH_C2))
        if len(cand) > 1:
            raise ValueError(f"Non-unique match on {item['edge']['key']}")
        if len(cand) == 1:
            out[j] = cand[0]
    hit = out[out >= 0]
    if len(set(hit.tolist())) != len(hit):
        raise ValueError(f"Two guarded rows match one block state on {item['edge']['key']}")
    return out


def guarded_signs(indexed, info):
    sig, worst = {}, 0.0
    for a, out in sorted(indexed.items()):
        for lf, item in sorted(out.items()):
            T = item["target"]
            if T == a or T not in info or item["edge"]["target_status"] in REUSED:
                continue
            rev = indexed.get(T, {}).get(L_of(a))
            if rev is None or rev["edge"]["target_status"] not in REUSED:
                continue
            rows = target_rows(item, info)
            c = -np.sqrt((2 * L_of(a) + 1) / (2 * L_of(T) + 1))
            scale = float(np.abs(item["matrix"]).max())
            found, r = align_signs(item["matrix"], rev["matrix"], rows, c, scale)
            worst = max(worst, r)
            gauge = tuple(item["edge"]["target_colsha"])
            store = sig.setdefault(T, {"gauge": gauge, "sigma": {}})
            if store["gauge"] != gauge:
                raise ValueError(f"Two guarded gauges for {T}")
            for t, (s, _) in found.items():
                if store["sigma"].get(t, s) != s:
                    raise ValueError(f"Conflicting guarded sign for {T}#{t + 1}")
                store["sigma"][t] = s
    return sig, worst


def generator_maps(indexed, info, check=True):
    """phi[(c, b)] (n_c x n_b), physically normalised, block gauge on both sides."""
    sig, sign_worst = guarded_signs(indexed, info)
    diag = dict(sign_rel=sign_worst, map_rel=0.0, sym_rel=0.0, dropped_L4=[], maps={})
    if check and sign_worst > SIGN_REL:
        raise ValueError(f"Guarded sign residual {sign_worst}")
    phi = {}
    for b, out in sorted(indexed.items()):
        for lf, item in sorted(out.items()):
            c = item["target"]
            if c not in info:
                diag["dropped_L4"].append(item["edge"]["key"])
                continue
            Lb, Lc = L_of(b), L_of(c)
            status = item["edge"]["target_status"]
            if Lb != Lc:
                cand = []
                if status in REUSED:
                    cand.append(("direct", np.sqrt((2 * Lc + 1) / (2 * Lb + 1)) * item["matrix"]))
                rev = indexed.get(c, {}).get(Lb)
                if rev is not None and rev["edge"]["target_status"] in REUSED:
                    cand.append(("reverse", -rev["matrix"].T))
                if not cand:
                    raise ValueError(f"No block-gauge map for {c}<-{b}")
                if len(cand) == 2:
                    rel = float(np.abs(cand[0][1] - cand[1][1]).max() / np.abs(cand[0][1]).max())
                    diag["map_rel"] = max(diag["map_rel"], rel)
                    if check and rel > MAP_REL:
                        raise ValueError(f"Direct and reverse maps disagree for {c}<-{b}: {rel}")
                phi[(c, b)] = cand[0][1]
                diag["maps"][f"{c}<-{b}"] = "+".join(k for k, _ in cand)
            else:
                if status in REUSED:
                    M = item["matrix"].copy()
                    diag["maps"][f"{c}<-{b}"] = "direct"
                else:
                    rows = target_rows(item, info)
                    n = len(info[c]["delta"])
                    if sorted(rows[rows >= 0].tolist()) != list(range(n)):
                        raise ValueError(f"Guarded rows do not cover block {c}")
                    if c not in sig or sig[c]["gauge"] != tuple(item["edge"]["target_colsha"]):
                        raise ValueError(f"No sign reference for guarded {c}")
                    scale = float(np.abs(item["matrix"]).max())
                    M = np.zeros((n, item["matrix"].shape[1]))
                    for j, t in enumerate(rows):
                        if t < 0:
                            continue
                        s = sig[c]["sigma"].get(int(t))
                        if s is None:
                            if np.abs(item["matrix"][j]).max() > ZERO_ROW * scale:
                                raise ValueError(f"Unsigned non-zero guarded row {c}#{t + 1}")
                            continue
                        M[t] = s * item["matrix"][j]
                    diag["maps"][f"{c}<-{b}"] = "guarded_aligned"
                rel = float(np.abs(M - M.T).max() / np.abs(M).max())
                diag["sym_rel"] = max(diag["sym_rel"], rel)
                if check and rel > SYM_REL:
                    raise ValueError(f"Delta L = 0 map of {c} is not symmetric: {rel}")
                phi[(c, b)] = M
    return phi, diag


def raising_maps(phi, info, width, atol):
    out = {}
    for (c, b), M in phi.items():
        gap = info[c]["delta"][:, None] - info[b]["delta"][None, :]
        f, active = run.lowering_weight(gap, width, atol)
        out[(c, b)] = M * (1 - f) * active
    return out


# ----------------------------------------------------------------------------- families and subspaces

def family_vectors(modes, method, info):
    vec = {}
    for r in modes[modes.method == method].itertuples():
        fid = f"{r.block}:C{r.so5_c2}:X{r.X}:P{r.P}:m{r.mode}"
        v = np.zeros(len(info[r.block]["delta"]))
        for idx, c in json.loads(r.vector_json).items():
            v[int(idx) - 1] = c
        if abs(np.linalg.norm(v) - 1) > 1e-9:
            raise ValueError(f"Mode {fid} is not normalised")
        vec[fid] = v
    return vec


def sector_rows(info, block, c2, x, p):
    i = info[block]
    return (np.abs(i["c2"] - c2) < fm.SAME_C2) & (i["x"] == x) & (i["p"] == p)


def build(fam, vec, praise, info, cover, method):
    Q, slots = {}, []
    for F, f in fam.items():
        Q[(F, 0, f["block"])] = vec[F][:, None]
        nmax = int(np.floor(run.DELTA_MAX - f["delta"] + 1e-12))
        for n in range(1, nmax + 1):
            for j in range(4):
                c = f"s0_s{f['s_B']}_L{j}"
                if c not in info:
                    continue
                parts = [praise[(c, b)] @ Q[(F, n - 1, b)] for b in sorted({k[2] for k in Q if k[0] == F
                                                                         and k[1] == n - 1})
                         if (c, b) in praise]
                Y = np.hstack(parts) if parts else np.zeros((len(info[c]["delta"]), 0))
                cols = []
                for p in (1, -1):
                    m = fm.expected(n, f["L"], j, p * f["P"], f["conserved"])
                    rows = sector_rows(info, c, f["so5_c2"], f["X"], p)
                    U, s = slot_basis(Y[rows], max(m, 0))
                    q = np.zeros((len(rows), U.shape[1]))
                    q[rows] = U
                    cols.append(q)
                    if m <= 0 and (s.size == 0 or s[0] <= SV_EPS):
                        continue
                    nominal = f["delta"] + n <= run.DELTA_MAX
                    energy = float(np.mean((q ** 2).T @ info[c]["delta"])) if q.shape[1] else np.nan
                    slots.append(dict(
                        method=method, family=F, irrep=f["irrep"], family_L=f["L"], family_P=f["P"],
                        family_delta=f["delta"], family_kind=f["kind"], conserved=f["conserved"], level=n, j=j, P=p,
                        block=c, expected=m, rank=q.shape[1], deficit=max(m, 0) - q.shape[1],
                        sigma1=float(s[0]) if s.size else 0.0,
                        redundancy=float(s[m] / s[m - 1]) if m > 0 and s.size > m and s[m - 1] > 0 else np.nan,
                        singular_values=json.dumps([round(float(v), 9) for v in s[:max(m, 0) + 3]]),
                        energy=energy, energy_gap=energy - f["delta"] if q.shape[1] else np.nan,
                        nominal=nominal, complete=bool(nominal and f["delta"] + n + fm.SLOT_MARGIN <= cover[c])))
                Q[(F, n, c)] = np.hstack(cols)
    slots = pd.DataFrame(slots)
    allowed = slots[slots.expected > 0].groupby(["family", "level"]).sigma1.max()
    slots["forbidden_ratio"] = [r.sigma1 / allowed.get((r.family, r.level), np.nan) if r.expected <= 0 else np.nan
                                for r in slots.itertuples()]
    return Q, slots


def decompose(fam, vec, Q, slots, info, states, cover, method):
    """Loewdin decomposition per sector (block, C2, X, P) of eligible states."""
    complete = {(r.family, r.level, r.block, r.P) for r in slots.itertuples() if r.complete and r.rank > 0}
    elig = states[states.status == "eligible"]
    rho, sectors = {}, []
    for (block, c2, x, p), g in elig.groupby(["block", "so5_c2", "X", "P"]):
        rows = sector_rows(info, block, c2, x, p)
        cols, labels, by_family = [], [], {}
        for F, f in fam.items():
            if (f["so5_c2"], f["X"]) != (c2, x):
                continue
            pieces = []
            if f["block"] == block and f["P"] == p:
                pieces.append((0, vec[F][:, None]))
            for n in range(1, 8):
                if (F, n, block, p) in complete:
                    q = Q[(F, n, block)]
                    q = q[:, np.abs(q[~rows]).sum(axis=0) < 1e-12] if q.size else q
                    pieces.append((n, q))
            for n, q in pieces:
                for k in range(q.shape[1]):
                    cols.append(q[rows, k])
                    labels.append((F, n))
                by_family.setdefault(F, []).append(q[rows])
        pos = np.flatnonzero(rows)
        local = {int(t): i for i, t in enumerate(pos)}
        Z = np.array(cols).T if cols else np.zeros((len(pos), 0))
        Zt, lam_min, dropped = lowdin(Z) if cols else (Z, np.nan, 0)
        test_lim = min(cover[block] - fm.SLOT_MARGIN, run.DELTA_MAX)
        test = [local[r.raw_index - 1] for r in g.itertuples() if r.delta + TEST_MARGIN <= test_lim]
        resid = np.nan
        if test:
            Pi = Zt @ Zt.T if cols else np.zeros((len(pos), len(pos)))
            R = np.eye(len(test)) - Pi[np.ix_(test, test)]
            resid = float(np.linalg.eigvalsh(R).max())
        for r in g.itertuples():
            i = local[r.raw_index - 1]
            w = {}
            for k, key in enumerate(labels):
                w[key] = w.get(key, 0.0) + float(Zt[i, k] ** 2)
            if sum(w.values()) > 1 + COVER_TOL or any(v < 0 for v in w.values()):
                raise ValueError(f"Invalid coverage for {block}#{r.raw_index}")
            rho[(block, r.raw_index)] = w
        sectors.append(dict(method=method, block=block, so5_c2=int(c2), X=int(x), P=int(p), states=int(len(g)),
                            sector_dim=int(rows.sum()), columns=len(cols), families=len(by_family),
                            overlap_min_eig=lam_min, lowdin_dropped=dropped,
                            family_max_cos2=max_cos2([orth(np.hstack(v)) for v in by_family.values()]),
                            test_states=len(test), residual_max_eig=resid,
                            min_test_coverage=min((sum(rho[(block, int(pos[i]) + 1)].values()) for i in test),
                                                  default=np.nan)))
    return rho, pd.DataFrame(sectors)


# ----------------------------------------------------------------------------- inputs and driver

def upstream_hashes():
    out = fm.v3_hashes()
    out.update({name: run.sha(path) for name, path in UPSTREAM.items()})
    return out


def freeze(path):
    if path.exists():
        raise FileExistsError(path)
    run.write_json(path, {"code": {name: run.sha(HERE / name) for name in CODE_FILES},
                          "upstream": upstream_hashes(), "numpy": np.__version__, "pandas": pd.__version__})


def load(source, threshold, lock):
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    if {name: run.sha(HERE / name) for name in manifest["code"]} != manifest["code"]:
        raise ValueError("Coherent code, protocol or tests differ from the freeze")
    if upstream_hashes() != manifest["upstream"]:
        raise ValueError("Upstream code or v3 results differ from the freeze")
    blocks, indexed, parity, _, _, alpha, e0 = run.load(source, threshold, V3 / "freeze.json")
    states = pd.read_csv(V3 / "results/states.csv")
    modes = pd.read_csv(V3 / "results/modes.csv")
    return blocks, indexed, parity, alpha, e0, states, modes


def cover_of(states):
    cover = {}
    for block, g in states.groupby("block"):
        bad = g[(g.status != "eligible") & (g.status != "vacuum")].delta
        cover[block] = float(bad.min()) if len(bad) else np.inf
    return cover


def analyse(blocks, indexed, parity, alpha, e0, states, modes):
    info = block_info(blocks, parity, alpha, e0)
    phi, mapdiag = generator_maps(indexed, info)
    atol = max(item["atol"] for out in indexed.values() for item in out.values())
    cover = cover_of(states)
    elig = states[states.status == "eligible"].sort_values("delta", kind="stable")
    order = [(r.block, r.raw_index) for r in elig.itertuples()]
    per_state, all_slots, all_sectors, fams = {}, [], [], {}
    for method in run.CLASSIFY:
        _, fam = fm.families_of(modes, method, states)
        vec = family_vectors(modes, method, info)
        praise = raising_maps(phi, info, run.WIDTHS[method], atol)
        Q, slots = build(fam, vec, praise, info, cover, method)
        rho, sectors = decompose(fam, vec, Q, slots, info, states, cover, method)
        fams[method] = fam
        all_slots.append(slots)
        all_sectors.append(sectors)
        for t in order:
            tag, top, prim, cov = label(rho[t])
            row = per_state.setdefault(t, {})
            row[f"{method}_label"] = tag
            row[f"{method}_primary_weight"] = prim
            row[f"{method}_coverage"] = cov
            for rank, ((F, n), v) in enumerate(top + [(("", -1), np.nan)] * (2 - len(top)), 1):
                row[f"{method}_top{rank}_family"] = F
                row[f"{method}_top{rank}_level"] = n
                row[f"{method}_top{rank}_share"] = v
    info_s = elig.set_index(["block", "raw_index"])
    out = []
    for t in order:
        s = info_s.loc[t]
        r = per_state[t]
        same = len({(r[f"{m}_label"], r[f"{m}_top1_family"], r[f"{m}_top1_level"]) for m in run.CLASSIFY}) == 1
        out.append(dict(block=t[0], raw_index=t[1], L=int(s.L), so5_c2=int(s.so5_c2),
                        **irrep(int(s.so5_c2), int(t[0].split("_")[1][1:])),
                        X=int(s.X), P=int(s.P), delta=float(s.delta), v3_support=s.support,
                        consensus=r["hard_label"] if same else "method_dependent", **r))
    mapdiag = {k: v for k, v in mapdiag.items()}
    return (pd.DataFrame(out), pd.concat(all_slots, ignore_index=True), pd.concat(all_sectors, ignore_index=True),
            pd.DataFrame(list(fams["hard"].values())), mapdiag)


def summarise(states_out, slots, sectors, fam_rows, mapdiag):
    fam = fam_rows.set_index("fid")
    desc = states_out[states_out.hard_label == "descendant"]
    gaps = {}
    for n, g in desc.groupby("hard_top1_level"):
        d = g.delta - fam.loc[g.hard_top1_family, "delta"].to_numpy()
        gaps[int(n)] = dict(count=int(len(g)), gap_mean=float(d.mean()), gap_min=float(d.min()),
                            gap_max=float(d.max()))
    slot_summary, sector_summary = {}, {}
    for method, g in slots.groupby("method"):
        comp = g[g.complete & (g.expected > 0)]
        forb = g[g.expected <= 0]
        egap = {int(n): dict(mean=float(h.energy_gap.mean()), min=float(h.energy_gap.min()),
                             max=float(h.energy_gap.max()))
                for n, h in comp[comp["rank"] > 0].groupby("level")}
        slot_summary[method] = dict(
            complete_slots=int(len(comp)), complete_expected=int(comp.expected.sum()),
            complete_rank=int(comp["rank"].sum()), complete_deficit=int(comp.deficit.sum()),
            redundancy_max=float(comp.redundancy.max()), redundancy_median=float(comp.redundancy.median()),
            redundancy_anomalies=int((comp.redundancy >= ANOMALY).sum()),
            forbidden_slots=int(len(forb)), forbidden_ratio_max=float(forb.forbidden_ratio.max())
            if len(forb) else 0.0,
            forbidden_anomalies=int((forb.forbidden_ratio >= ANOMALY).sum()),
            complete_energy_gap_by_level=egap)
    for method, g in sectors.groupby("method"):
        sector_summary[method] = dict(
            sectors=int(len(g)), overlap_min_eig=float(g.overlap_min_eig.min()),
            overlap_anomalies=int((g.overlap_min_eig < ANOMALY).sum()),
            family_max_cos2=float(g.family_max_cos2.max()),
            duplicate_anomalies=int((g.family_max_cos2 >= ANOMALY).sum()),
            residual_max_eig=float(g.residual_max_eig.max()),
            residual_anomalies=int((g.residual_max_eig >= ANOMALY).sum()),
            min_test_coverage=float(g.min_test_coverage.min()))
    non_primary = states_out[states_out.v3_support != "robust"]
    return dict(
        states=int(len(states_out)),
        maps=dict(sign_rel=mapdiag["sign_rel"], map_rel=mapdiag["map_rel"], sym_rel=mapdiag["sym_rel"],
                  dropped_L4=mapdiag["dropped_L4"]),
        labels={m: states_out[f"{m}_label"].value_counts().to_dict() for m in run.CLASSIFY},
        consensus=states_out.consensus.value_counts().to_dict(),
        consensus_non_primary=non_primary.consensus.value_counts().to_dict(),
        primary_label_matches_v3_robust=bool(((states_out.hard_label == "primary")
                                              == (states_out.v3_support == "robust")).all()),
        descendant_gaps_by_level=gaps, slots=slot_summary, sectors=sector_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "run"):
        p = sub.add_parser(name)
        p.add_argument("inputs", type=Path)
        p.add_argument("threshold", type=Path)
        p.add_argument("--lock", type=Path, default=HERE / "freeze.json")
        p.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.lock)
        print(args.lock.read_text(encoding="utf-8"))
        return
    data = load(args.inputs, args.threshold, args.lock)
    states_out, slots, sectors, fam_rows, mapdiag = analyse(*data)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {"state_families.csv": states_out, "slots.csv": slots, "sectors.csv": sectors,
             "families.csv": fam_rows}
    for name, df in files.items():
        df.to_csv(args.output / name, index=False, lineterminator="\n", float_format="%.12g")
    run.write_json(args.output / "maps.json", mapdiag)
    summary = summarise(states_out, slots, sectors, fam_rows, mapdiag)
    run.write_json(args.output / "summary.json", summary)
    run.write_json(args.output / "receipt.json", {
        "freeze_sha256": run.sha(args.lock),
        "outputs": {name: run.sha(args.output / name) for name in (*files, "maps.json", "summary.json")}})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
