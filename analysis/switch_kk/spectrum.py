"""Post-freeze readout: merge raw block copies of each SO(5) multiplet and list the primary spectrum of the analysed size.

Reads only results/ of this directory (hash-checked against receipt.json). No literature values or labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run

COPY_TOL = 1e-6          # Delta agreement of copies of one multiplet (raw energies agree to ~1e-9)


def irrep(c2_half, s_b):
    """SO(5) irrep from the exported Casimir (normalised so that the vector has C2 = 2) and the block.

    Highest weight (l1, l2), l1 >= l2 >= 0: C2 = [l1 (l1 + 3) + l2 (l2 + 1)] / 2.  A block s0_s{k} holds
    the SO(4) component with segment spins (0, k); by the SO(5) > SO(4) branching rule this requires
    l2 = k, so each irrep appears in exactly one s0_s{k} block and l2 = s_B fixes Casimir degeneracies."""
    hits = [(l1, l2) for l1 in range(8) for l2 in range(l1 + 1)
            if l1 * (l1 + 3) + l2 * (l2 + 1) == 2 * c2_half]
    match = [h for h in hits if h[1] == s_b]
    if len(match) != 1:
        raise ValueError(f"No unique SO(5) irrep for C2={c2_half} in an s_B={s_b} block: {hits}")
    l1, l2 = match[0]
    dim = (2 * l1 + 3) * (2 * l2 + 1) * (l1 - l2 + 1) * (l1 + l2 + 2) // 6
    return dict(irrep=f"({l1},{l2})", dim=dim)


def merge(states):
    levels = []
    for (ell, c2), group in states.groupby(["L", "so5_c2"]):
        group = group.sort_values("delta")
        cluster, start = [], None
        for _, row in group.iterrows():
            if start is not None and row.delta - start > COPY_TOL:
                levels.append(cluster)
                cluster = []
            if not cluster:
                start = row.delta
            cluster.append(row)
        levels.append(cluster)
    rows = []
    for cluster in levels:
        c = pd.DataFrame(cluster)
        eligible = c[c.status == "eligible"]
        labels = sorted(set(eligible.support))
        if eligible.empty:
            verdict = "not_covered" if (c.status == "frontier_excluded").any() else c.status.iloc[0]
        else:
            verdict = labels[0] if len(labels) == 1 else "copy_conflict"
        weights = eligible[[f"{m}_weight_{cut}" for m in run.CLASSIFY for cut in run.BAND]]
        rows.append(dict(
            L=int(c.L.iloc[0]), so5_c2=int(c.so5_c2.iloc[0]),
            **irrep(int(c.so5_c2.iloc[0]), int(c.block.iloc[0].split("_")[1][1:])),
            P=json.dumps(sorted(set(int(p) for p in c.P))), delta=float(c.delta.mean()),
            delta_spread=float(c.delta.max() - c.delta.min()),
            copies=len(c), eligible_copies=len(eligible),
            copy_keys=";".join(f"{b}#{i}" for b, i in zip(c.block, c.raw_index)),
            verdict=verdict,
            min_band_weight=float(weights.min().min()) if len(eligible) else np.nan,
            max_band_weight=float(weights.max().max()) if len(eligible) else np.nan,
            paper_window_central=(float(eligible["paper_window_weight_central"].min()) if len(eligible) else np.nan),
            hard_comm_minus_delta=(float(eligible.hard_comm_minus_delta.mean()) if len(eligible) else np.nan),
        ))
    return pd.DataFrame(rows).sort_values(["so5_c2", "L", "delta"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=run.ROOT / "results")
    args = parser.parse_args()
    receipt = json.loads((args.results / "receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["outputs"].items():
        if run.sha(args.results / name) != digest:
            raise ValueError(f"{name} differs from its receipt")
    states = pd.read_csv(args.results / "states.csv")
    states = states[states.status != "vacuum"]
    levels = merge(states)
    if (levels.delta_spread > COPY_TOL).any() or (levels.P.str.count(",") > 0).any():
        raise ValueError("Copies of one multiplet disagree in Delta or parity")
    levels.to_csv(args.results / "levels.csv", index=False, lineterminator="\n")
    # completeness: every in-scope level below this Delta in a block is eligible (frontier - 0.5)
    first_gap = {}
    for block, group in states.groupby("block"):
        bad = group[group.status != "eligible"].delta
        first_gap[block] = float(bad.min()) if len(bad) else None
    # lowest L = 0 level of each rank-q symmetric tensor (q, 0)
    sym = levels[(levels.L == 0) & levels.irrep.str.match(r"^\(\d,0\)$")]
    charge = {r.irrep: dict(delta=r.delta, verdict=r.verdict)
              for r in sym.sort_values("delta").groupby("irrep").head(1).itertuples()}
    summary = dict(
        first_unclassified_delta_by_block=first_gap, lowest_symmetric_tensor_L0=charge,
        levels=len(levels), verdicts=levels.verdict.value_counts().to_dict(),
        copies_per_level=levels.copies.value_counts().sort_index().to_dict(),
        primaries_by_irrep=levels[levels.verdict == "robust"].groupby("irrep").size().to_dict(),
        copy_conflicts=levels[levels.verdict == "copy_conflict"].copy_keys.tolist(),
        levels_sha256=run.sha(args.results / "levels.csv"),
    )
    (args.results / "levels_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    show = levels[levels.verdict.isin(["robust", "sensitive", "copy_conflict"])]
    print(show[["irrep", "dim", "L", "P", "delta", "copies", "eligible_copies", "verdict",
                "min_band_weight", "paper_window_central", "hard_comm_minus_delta"]].to_string())


if __name__ == "__main__":
    main()
