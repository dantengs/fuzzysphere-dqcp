"""Post-freeze readout of the family check (not part of the frozen rules).

1. Joint counting: complete slots summed over all families of one SO(5) irrep, per (j, P).  Family-to-family
   redistribution inside one irrep cancels here; a missed primary would show up as overfull.
2. Sensitivity variant: drop the 13 dispersed directions from the family sources (hard split) and propagate
   again.  If they were descendants rather than primaries, the counts should then close; if they are
   primaries, the variant overfills.
3. Tables for the report: descendant towers and mixed states.
Reads the frozen family results (hash-checked) and the KK inputs through family.load."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import family as fm

R = fm.HERE / "results"


def joint(slots):
    comp = slots[slots.complete & (slots.expected > 0)]
    g = comp.groupby(["irrep", "j", "P"]).agg(expected=("expected", "sum"), occupancy=("occupancy", "sum"))
    g["deviation"] = g.occupancy - g.expected
    return comp, g


def stats(slots):
    comp, g = joint(slots)
    return dict(complete_expected=int(comp.expected.sum()), complete_occupancy=float(comp.occupancy.sum()),
                joint_groups=int(len(g)), joint_sum_abs_dev=float(g.deviation.abs().sum()),
                joint_max_over=float(g.deviation.max()), joint_max_under=float(g.deviation.min()),
                joint_over_ge_0_5=int((g.deviation >= fm.ANOMALY).sum()),
                joint_under_ge_0_5=int((g.deviation <= -fm.ANOMALY).sum()),
                forbidden_occupancy=float(slots[slots.expected <= 0].occupancy.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("threshold", type=Path)
    args = parser.parse_args()
    receipt = json.loads((R / "receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["outputs"].items():
        if fm.run.sha(R / name) != digest:
            raise ValueError(f"{name} differs from its receipt")
    slots = pd.read_csv(R / "slots.csv")
    states = pd.read_csv(R / "state_families.csv")
    fams = pd.read_csv(R / "families.csv").set_index("fid")
    out = {"frozen": {m: stats(slots[slots.method == m]) for m in fm.run.CLASSIFY}}
    _, g = joint(slots[slots.method == "hard"])
    g.reset_index().to_csv(R / "joint_counts_hard.csv", index=False, lineterminator="\n", float_format="%.6g")

    # sensitivity variant: dispersed directions are not family sources
    blocks, indexed, alpha, e0, v3_states, modes, _ = fm.load(args.inputs, args.threshold, fm.HERE / "freeze.json")
    deltas = fm.block_deltas(blocks, alpha, e0)
    elig = v3_states[v3_states.status == "eligible"].sort_values("delta", kind="stable")
    order = [(r.block, r.raw_index) for r in elig.itertuples()]
    cover = {}
    for block, grp in v3_states.groupby("block"):
        bad = grp[(grp.status != "eligible") & (grp.status != "vacuum")].delta
        cover[block] = float(bad.min()) if len(bad) else np.inf
    prim, fam = fm.families_of(modes, "hard", v3_states)
    keep = {f for f, v in fam.items() if v["kind"] == "concentrated"}
    prim_c = {t: {f: a for f, a in d.items() if f in keep} for t, d in prim.items()}
    fam_c = {f: v for f, v in fam.items() if f in keep}
    parents, _ = fm.parent_table(blocks, indexed, deltas, v3_states, fm.run.WIDTHS["hard"])
    rho, unres = fm.propagate(order, prim_c, parents)
    variant = fm.slot_table(fam_c, rho, v3_states, cover, "hard_without_dispersed")
    # compare on the same slot set (families kept in both runs)
    base = slots[(slots.method == "hard") & slots.family.isin(keep)]
    out["concentrated_families_only"] = {"frozen_run": stats(base), "variant_without_dispersed": stats(variant)}
    _, gv = joint(variant)
    gv.reset_index().to_csv(R / "joint_counts_without_dispersed.csv", index=False, lineterminator="\n",
                            float_format="%.6g")

    # report tables
    def name(f):
        if not isinstance(f, str):
            return ""
        r = fams.loc[f]
        return f"{r.irrep} L{r.L}{'+' if r.P > 0 else '−'} {r.delta:.3f}{' (分散)' if r.kind == 'dispersed' else ''}"
    st = states.copy()
    st["family1"] = st.hard_top1_family.map(name)
    st["family2"] = st.hard_top2_family.map(name)
    st["gap_to_family1"] = st.delta - st.hard_top1_family.map(fams.delta)
    cols = ["block", "raw_index", "irrep", "L", "P", "delta", "v3_support", "consensus", "hard_primary_weight",
            "family1", "hard_top1_level", "hard_top1_share", "family2", "hard_top2_level", "hard_top2_share",
            "gap_to_family1", "hard_unresolved"]
    st[cols].to_csv(R / "state_table.csv", index=False, lineterminator="\n", float_format="%.6g")
    mixed = st[st.consensus == "mixed"]
    out["mixed"] = dict(
        count=int(len(mixed)),
        with_primary_part=int((mixed.hard_primary_weight >= 0.02).sum()),
        two_families_ge_0_9=int((mixed.hard_top1_share + mixed.hard_top2_share >= fm.GATE).sum()),
        same_irrep_pairs=int(sum(isinstance(b, str) and fams.loc[a, "irrep"] == fams.loc[b, "irrep"]
                                 for a, b in zip(mixed.hard_top1_family, mixed.hard_top2_family))))
    norms = pd.read_csv(R / "channel_norms.csv")
    ok = norms[norms.cft > 1e-9]
    out["channel_norms"] = dict(
        channels=int(len(ok)), ratio_min=float(ok.ratio.min()), ratio_max=float(ok.ratio.max()),
        ratio_median=float(ok.ratio.median()),
        ratio_median_delta_lt_4=float(ok[ok.delta < 4].ratio.median()),
        ratio_median_delta_ge_4=float(ok[ok.delta >= 4].ratio.median()),
        null_channels=norms[norms.cft <= 1e-9][["irrep", "L", "delta", "j", "P_norm"]].to_dict("records"))
    fm.run.write_json(R / "readout.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
