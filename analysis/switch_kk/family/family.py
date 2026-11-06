"""Family check after the v3 freeze: K-image parent propagation and conformal-multiplet counting.

Rules: protocol.md (this directory).  Inputs: the v3 KK inputs (checked by run.load against the v3 freeze) and
v3 results/states.csv, modes.csv (checked against results/receipt.json).  No literature values or labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3))
import run  # noqa: E402
from spectrum import irrep  # noqa: E402

GATE = run.GATE
CODE_FILES = ("family.py", "protocol.md", "test_family.py")
V3_RESULTS = ("states.csv", "modes.csv")
MATCH_DELTA, MATCH_C2, SAME_C2 = 1e-6, 1e-4, 1e-5
SLOT_MARGIN, ANOMALY, OCC_EPS = 0.25, 0.5, 1e-3
CONSERVED = {("s0_s1_L1", 1), ("s0_s0_L2", 2)}          # J and T anchors of the v3 protocol
SUM_TOL = 1e-9


# ----------------------------------------------------------------------------- representation theory

def mult(n, ell, j, rel):
    """Level-n descendants of a generic spin-ell primary with spin j and parity rel = P/P_F."""
    if n < 0:
        return 0
    return sum(1 for k in range(n % 2, n + 1, 2)
               if abs(k - ell) <= j <= k + ell and (-1) ** (k + ell - j) == rel)


def expected(n, ell, j, rel, conserved):
    """Generic count, minus the null module (spin ell-1 at level 1, relative parity +) for a conserved current."""
    m = mult(n, ell, j, rel)
    if conserved:
        m -= mult(n - 1, ell - 1, j, rel)
    return m


def channel_cft(delta, ell, j):
    lam = {ell + 1: ell, ell: -1, ell - 1: -(ell + 1)}[j]
    return (2 * j + 1) / (2 * ell + 1) * 2 * (delta + lam)


# ----------------------------------------------------------------------------- propagation

def propagate(order, prim, parents):
    """order: states by increasing Delta; prim[t] = {F: a}; parents[t] = ({s: weight}, lost weight).

    rho[t][(F, n)], unres[t] with sum(rho[t]) + unres[t] = 1 (incoherent K-image propagation)."""
    rho, unres = {}, {}
    for t in order:
        a = prim.get(t, {})
        w = float(sum(a.values()))
        if w > 1 + SUM_TOL or any(v < 0 for v in a.values()):
            raise ValueError(f"Invalid primary weights for {t}")
        rest = max(1.0 - w, 0.0)
        dist, lost = parents[t]
        if lost < 0 or any(v < 0 for v in dist.values()):
            raise ValueError(f"Negative parent weight for {t}")
        total = float(sum(dist.values()) + lost)
        r = {(F, 0): float(v) for F, v in a.items()}
        u = rest if total <= 0 else rest * lost / total
        if total > 0:
            for s, v in dist.items():
                if s not in rho:
                    raise ValueError(f"Parent {s} of {t} is not processed before it")
                share = rest * v / total
                for (F, n), x in rho[s].items():
                    r[(F, n + 1)] = r.get((F, n + 1), 0.0) + share * x
                u += share * unres[s]
        if abs(sum(r.values()) + u - 1) > SUM_TOL:
            raise ValueError(f"Weights of {t} do not sum to one")
        rho[t], unres[t] = r, u
    return rho, unres


def label(a, r, u):
    w = sum(a.values())
    ranked = sorted(r.items(), key=lambda kv: -kv[1])
    desc = [(k, v) for k, v in ranked if k[1] >= 1]
    by_family = {}
    for (F, n), v in r.items():
        if n >= 1:
            by_family[F] = by_family.get(F, 0.0) + v
    if w >= GATE:
        tag = "primary"
    elif u >= 1 - GATE:
        tag = "unresolved"
    elif desc and desc[0][1] >= GATE:
        tag = "descendant"
    elif by_family and max(by_family.values()) >= GATE:
        tag = "family_mixed_level"
    else:
        tag = "mixed"
    return tag, ranked[:2]


# ----------------------------------------------------------------------------- inputs

def v3_hashes():
    receipt = json.loads((V3 / "results/receipt.json").read_text(encoding="utf-8"))
    out = {}
    for name in V3_RESULTS:
        digest = run.sha(V3 / "results" / name)
        if receipt["outputs"].get(name) != digest:
            raise ValueError(f"v3 {name} differs from its receipt")
        out[name] = digest
    out["receipt.json"] = run.sha(V3 / "results/receipt.json")
    out["v3_freeze.json"] = run.sha(V3 / "freeze.json")
    return out


def freeze(path):
    if path.exists():
        raise FileExistsError(path)
    run.write_json(path, {"code": {name: run.sha(HERE / name) for name in CODE_FILES},
                          "v3": v3_hashes(), "numpy": np.__version__, "pandas": pd.__version__})


def load(source, threshold, lock):
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    if {name: run.sha(HERE / name) for name in manifest["code"]} != manifest["code"]:
        raise ValueError("Family code, protocol or tests differ from the freeze")
    if v3_hashes() != manifest["v3"]:
        raise ValueError("v3 results or freeze differ from the family freeze")
    blocks, indexed, parity, cuts, _, alpha, e0 = run.load(source, threshold, V3 / "freeze.json")
    states = pd.read_csv(V3 / "results/states.csv")
    modes = pd.read_csv(V3 / "results/modes.csv")
    return blocks, indexed, alpha, e0, states, modes, manifest


# ----------------------------------------------------------------------------- analysis

def block_deltas(blocks, alpha, e0):
    return {k: alpha * (np.asarray(b["energy"]) - e0) for k, b in blocks.items()}


def parent_table(blocks, indexed, deltas, states, width):
    """parents[(block, raw_index)] = ({(block, raw_index): w}, lost) with lost split by reason."""
    status = {(r.block, r.raw_index): r.status for r in states.itertuples()}
    eligible = states[states.status == "eligible"]
    parents, reasons = {}, {}
    for r in eligible.itertuples():
        pos, ell = r.raw_index - 1, r.L
        dist, lost = {}, {"L4": 0.0, "frontier": 0.0, "vacuum": 0.0}
        for lf, item in indexed[r.block].items():
            same = np.abs(np.asarray(item["edge"]["target_C2"]) - r.so5_c2) < SAME_C2
            gap = item["delta"] - r.delta
            down = same & (gap < -item["atol"])
            f, _ = run.lowering_weight(gap, width, item["atol"])
            values = (f ** 2) * item["matrix"][:, pos] ** 2 * (2 * lf + 1) / (2 * ell + 1)
            tgt = item["target"]
            for j in np.flatnonzero(down & (values > 0)):
                if tgt not in blocks:
                    lost["L4"] += values[j]
                    continue
                cand = np.flatnonzero((np.abs(deltas[tgt] - item["delta"][j]) < MATCH_DELTA)
                                      & (np.abs(np.asarray(blocks[tgt]["c2"]) - item["edge"]["target_C2"][j]) < MATCH_C2))
                if len(cand) != 1:
                    raise ValueError(f"No unique block match for a parent of {r.block}#{r.raw_index}")
                key = (tgt, int(cand[0]) + 1)
                st = status.get(key)
                if st == "eligible":
                    dist[key] = dist.get(key, 0.0) + float(values[j])
                elif st == "vacuum":
                    lost["vacuum"] += values[j]
                else:
                    lost["frontier"] += values[j]
        parents[(r.block, r.raw_index)] = (dist, float(sum(lost.values())))
        reasons[(r.block, r.raw_index)] = lost
    return parents, reasons


def families_of(modes, method, states):
    info = states.set_index(["block", "raw_index"])
    prim, fam = {}, {}
    for r in modes[modes.method == method].itertuples():
        fid = f"{r.block}:C{r.so5_c2}:X{r.X}:P{r.P}:m{r.mode}"
        sb = int(r.block.split("_")[1][1:])
        fam[fid] = dict(fid=fid, **irrep(int(r.so5_c2), sb), block=r.block, s_B=sb, so5_c2=int(r.so5_c2),
                        X=int(r.X), L=int(r.L), P=int(r.P), delta=float(r.delta),
                        dominant=f"{r.block}#{r.dominant_index}", dominant_weight=float(r.dominant_weight),
                        dominant_delta=float(info.loc[(r.block, r.dominant_index)].delta),
                        kind="concentrated" if r.dominant_weight >= GATE else "dispersed",
                        conserved=(r.block, int(r.dominant_index)) in CONSERVED and r.dominant_weight >= GATE)
        for idx, c in json.loads(r.vector_json).items():
            if c * c > 0:
                prim.setdefault((r.block, int(idx)), {})[fid] = c * c
    return prim, fam


def slot_table(fam, rho, states, cover, method):
    elig = states[states.status == "eligible"]
    occ = {}
    for r in elig.itertuples():
        for (F, n), v in rho[(r.block, r.raw_index)].items():
            if n >= 1:
                key = (F, n, r.L, int(r.P))
                occ[key] = occ.get(key, 0.0) + v
    rows = []
    for F, f in fam.items():
        nmax = max([n for (G, n, _, _) in occ if G == F] + [int(np.floor(run.DELTA_MAX - f["delta"]))])
        for n in range(1, nmax + 1):
            for j in range(0, 4):
                for p in (1, -1):
                    m = expected(n, f["L"], j, p * f["P"], f["conserved"])
                    o = occ.get((F, n, j, p), 0.0)
                    if m == 0 and o < OCC_EPS:
                        continue
                    block = f"s0_s{f['s_B']}_L{j}"
                    nominal = f["delta"] + n <= run.DELTA_MAX
                    rows.append(dict(method=method, family=F, irrep=f["irrep"], family_L=f["L"], family_P=f["P"],
                                     family_delta=f["delta"], family_kind=f["kind"], conserved=f["conserved"],
                                     level=n, j=j, P=p, expected=m, occupancy=o,
                                     nominal=nominal,
                                     complete=bool(nominal and f["delta"] + n + SLOT_MARGIN <= cover[block])))
    return pd.DataFrame(rows)


def channel_norms(fam, blocks, indexed, states):
    info = states.set_index(["block", "raw_index"])
    rows = []
    for f in fam.values():
        if f["kind"] != "concentrated":
            continue
        block, idx = f["dominant"].split("#")
        idx = int(idx)
        s = info.loc[(block, idx)]
        ell, pos = int(s.L), idx - 1
        for lf, item in sorted(indexed[block].items()):
            col = item["matrix"][:, pos]
            same = np.abs(np.asarray(item["edge"]["target_C2"]) - s.so5_c2) < SAME_C2
            gap = item["delta"] - s.delta
            w = (2 * lf + 1) / (2 * ell + 1)
            missing = max(float(item["norm"][pos] - np.sum(col ** 2)), 0.0)
            up = w * (float(np.sum((col ** 2)[same & (gap > item["atol"])])) + missing)
            down = w * float(np.sum((col ** 2)[same & (gap < -item["atol"])]))
            cft = channel_cft(float(s.delta), ell, lf)
            rows.append(dict(family=f["fid"], irrep=f["irrep"], L=ell, P=f["P"], delta=float(s.delta), j=lf,
                             P_norm=up, K_norm=down, missing=w * missing, cft=cft,
                             ratio=up / cft if cft > 1e-12 else np.nan))
    return pd.DataFrame(rows)


def analyse(blocks, indexed, alpha, e0, states, modes):
    deltas = block_deltas(blocks, alpha, e0)
    elig = states[states.status == "eligible"].sort_values("delta", kind="stable")
    order = [(r.block, r.raw_index) for r in elig.itertuples()]
    cover = {}
    for block, g in states.groupby("block"):
        bad = g[(g.status != "eligible") & (g.status != "vacuum")].delta
        cover[block] = float(bad.min()) if len(bad) else np.inf
    per_state, slots, fams, lost_rows = {}, [], {}, []
    for method in run.CLASSIFY:
        prim, fam = families_of(modes, method, states)
        parents, reasons = parent_table(blocks, indexed, deltas, states, run.WIDTHS[method])
        rho, unres = propagate(order, prim, parents)
        fams[method] = fam
        for t in order:
            a = prim.get(t, {})
            tag, top = label(a, rho[t], unres[t])
            row = per_state.setdefault(t, {})
            row[f"{method}_label"] = tag
            row[f"{method}_primary_weight"] = float(sum(a.values()))
            row[f"{method}_unresolved"] = unres[t]
            for rank, ((F, n), v) in enumerate(top + [(("", -1), np.nan)] * (2 - len(top)), 1):
                row[f"{method}_top{rank}_family"] = F
                row[f"{method}_top{rank}_level"] = n
                row[f"{method}_top{rank}_share"] = v
            if method == "hard":
                lost = reasons[t]
                total = sum(parents[t][0].values()) + parents[t][1]
                row.update({f"K_lost_{k}": (v / total if total > 0 else 0.0) for k, v in lost.items()})
        slots.append(slot_table(fam, rho, states, cover, method))
    info = elig.set_index(["block", "raw_index"])
    out = []
    for t in order:
        s = info.loc[t]
        r = per_state[t]
        same = len({(r[f"{m}_label"], r[f"{m}_top1_family"], r[f"{m}_top1_level"]) for m in run.CLASSIFY}) == 1
        out.append(dict(block=t[0], raw_index=t[1], L=int(s.L), so5_c2=int(s.so5_c2),
                        **irrep(int(s.so5_c2), int(t[0].split("_")[1][1:])),
                        X=int(s.X), P=int(s.P), delta=float(s.delta), v3_support=s.support,
                        consensus=r["hard_label"] if same else "method_dependent", **r))
    states_out = pd.DataFrame(out)
    slots = pd.concat(slots, ignore_index=True)
    fam_rows = pd.DataFrame(list(fams["hard"].values()))
    norms = channel_norms(fams["hard"], blocks, indexed, states)
    return states_out, slots, fam_rows, norms


def summarise(states_out, slots, fam_rows):
    hard = states_out
    desc = hard[hard.hard_label == "descendant"]
    fam = fam_rows.set_index("fid")
    gaps = desc.apply(lambda r: r.delta - fam.loc[r.hard_top1_family, "delta"], axis=1) if len(desc) else pd.Series()
    by_level = {}
    for n, g in gaps.groupby(desc.hard_top1_level):
        by_level[int(n)] = dict(count=int(len(g)), gap_mean=float(g.mean()), gap_min=float(g.min()),
                                gap_max=float(g.max()), gap_per_level_mean=float((g / n).mean()))
    slot_summary = {}
    for method, g in slots.groupby("method"):
        comp = g[g.complete & (g.expected > 0)]
        anomalies = comp[(comp.occupancy - comp.expected).abs() >= ANOMALY]
        slot_summary[method] = dict(
            complete_slots=int(len(comp)), complete_expected=int(comp.expected.sum()),
            complete_occupancy=float(comp.occupancy.sum()),
            sum_abs_deviation=float((comp.occupancy - comp.expected).abs().sum()),
            anomalies=int(len(anomalies)),
            overfull=int((anomalies.occupancy > anomalies.expected).sum()),
            underfull=int((anomalies.occupancy < anomalies.expected).sum()),
            forbidden_occupancy=float(g[g.expected <= 0].occupancy.sum()),
            negative_expected_slots=int((g.expected < 0).sum()))
    non_primary = hard[hard.v3_support != "robust"]
    return dict(
        states=int(len(hard)),
        labels={m: hard[f"{m}_label"].value_counts().to_dict() for m in run.CLASSIFY},
        consensus=hard.consensus.value_counts().to_dict(),
        consensus_non_primary=non_primary.consensus.value_counts().to_dict(),
        primary_label_matches_v3_robust=bool(((hard.hard_label == "primary") == (hard.v3_support == "robust")).all()),
        K_lost_max={k: float(hard[k].max()) for k in ("K_lost_L4", "K_lost_frontier", "K_lost_vacuum")},
        descendant_gaps_by_level=by_level, slots=slot_summary)


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
    blocks, indexed, alpha, e0, states, modes, _ = load(args.inputs, args.threshold, args.lock)
    states_out, slots, fam_rows, norms = analyse(blocks, indexed, alpha, e0, states, modes)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {"state_families.csv": states_out, "slots.csv": slots, "families.csv": fam_rows,
             "channel_norms.csv": norms}
    for name, df in files.items():
        df.to_csv(args.output / name, index=False, lineterminator="\n", float_format="%.12g")
    summary = summarise(states_out, slots, fam_rows)
    run.write_json(args.output / "summary.json", summary)
    run.write_json(args.output / "receipt.json", {
        "freeze_sha256": run.sha(args.lock),
        "outputs": {name: run.sha(args.output / name) for name in (*files, "summary.json")}})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
