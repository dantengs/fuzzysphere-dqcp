"""Post-freeze readout of the coherent family check, and comparison with the incoherent version.

Not part of the frozen rules.  Reads coherent results (hash-checked against results/receipt.json) and the
incoherent results in ../results (hash-checked against ../results/receipt.json)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import run  # noqa: E402


def checked(folder, names):
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    for name in names:
        if receipt["outputs"][name] != run.sha(folder / name):
            raise ValueError(f"{folder / name} differs from its receipt")
    return [pd.read_csv(folder / name) for name in names]


def main():
    coh, slots, sectors, fams = checked(HERE / "results", ["state_families.csv", "slots.csv", "sectors.csv",
                                                           "families.csv"])
    inc, = checked(HERE.parent / "results", ["state_families.csv"])
    fam = fams.set_index("fid")
    cover = {}
    states = pd.read_csv(HERE.parent.parent / "results/states.csv")
    for block, g in states.groupby("block"):
        bad = g[(g.status != "eligible") & (g.status != "vacuum")].delta
        cover[block] = float(bad.min()) if len(bad) else np.inf
    coh["test_region"] = [r.delta + 0.5 <= min(cover[r.block] - 0.25, run.DELTA_MAX) for r in coh.itertuples()]

    out = {}
    # uncovered states: where are they
    unc = coh[coh.hard_label == "uncovered"]
    out["uncovered"] = dict(count=int(len(unc)), in_test_region=int(unc.test_region.sum()),
                            delta_min=float(unc.delta.min()), coverage_max=float(unc.hard_coverage.max()),
                            test_region_coverage_min=float(coh[coh.test_region].hard_coverage.min()))
    out["test_region"] = dict(states=int(coh.test_region.sum()),
                              labels=coh[coh.test_region].hard_label.value_counts().to_dict())

    # mixed states: what mixes
    def kind(r):
        f1, f2 = r.hard_top1_family, r.hard_top2_family
        k1 = fam.loc[f1, "kind"] if f1 in fam.index else ""
        k2 = fam.loc[f2, "kind"] if isinstance(f2, str) and f2 in fam.index else ""
        if "dispersed" in (k1, k2):
            return "involves_dispersed"
        if r.hard_top1_level == 0 or r.hard_top2_level == 0:
            return "primary_vs_descendant"
        return "two_descendants"
    mixed = coh[coh.hard_label == "mixed"].copy()
    mixed["mix_kind"] = [kind(r) for r in mixed.itertuples()]
    mixed["top_sum"] = mixed.hard_top1_share + mixed.hard_top2_share
    out["mixed"] = dict(count=int(len(mixed)), kinds=mixed.mix_kind.value_counts().to_dict(),
                        top2_sum_median=float(mixed.top_sum.median()),
                        top1_share_median=float(mixed.hard_top1_share.median()))

    # comparison with the incoherent check
    m = coh.merge(inc, on=["block", "raw_index"], suffixes=("_coh", "_inc"))
    cross = pd.crosstab(m.hard_label_inc, m.hard_label_coh)
    same_assign = ((m.hard_top1_family_coh == m.hard_top1_family_inc)
                   & (m.hard_top1_level_coh == m.hard_top1_level_inc))
    both_desc = m[(m.hard_label_coh == "descendant") & (m.hard_label_inc == "descendant")]
    out["vs_incoherent"] = dict(
        crosstab={i: {c: int(v) for c, v in row.items() if v} for i, row in cross.iterrows()},
        both_descendant=int(len(both_desc)),
        both_descendant_same_family_level=int(same_assign[both_desc.index].sum()),
        top1_same_family_level_all=int(same_assign.sum()))
    inc_mixed_to_desc = m[(m.hard_label_inc == "mixed") & (m.hard_label_coh == "descendant")]
    out["vs_incoherent"]["incoherent_mixed_now_descendant"] = int(len(inc_mixed_to_desc))

    # per-state table
    keep = ["block", "raw_index", "irrep", "L", "P", "delta", "v3_support", "test_region", "consensus",
            "hard_label", "hard_coverage", "hard_top1_family", "hard_top1_level", "hard_top1_share",
            "hard_top2_family", "hard_top2_level", "hard_top2_share"]
    table = coh[keep].merge(inc[["block", "raw_index", "hard_label", "hard_top1_family", "hard_top1_level",
                                 "hard_top1_share"]].rename(columns=lambda c: c.replace("hard_", "incoherent_")),
                            on=["block", "raw_index"])
    table = table.sort_values("delta", kind="stable")
    table.to_csv(HERE / "results/state_table.csv", index=False, lineterminator="\n", float_format="%.6g")
    mixed.sort_values("delta")[["block", "raw_index", "irrep", "L", "P", "delta", "mix_kind", "hard_top1_family",
                                "hard_top1_level", "hard_top1_share", "hard_top2_family", "hard_top2_level",
                                "hard_top2_share"]].to_csv(HERE / "results/mixed.csv", index=False,
                                                           lineterminator="\n", float_format="%.6g")
    run.write_json(HERE / "results/readout.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(mixed.sort_values("delta")[["block", "raw_index", "irrep", "delta", "mix_kind", "hard_top1_family",
                                      "hard_top1_level", "hard_top1_share", "hard_top2_family", "hard_top2_level",
                                      "hard_top2_share"]].to_string())
    md = coh[coh.consensus == "method_dependent"]
    print(md[["block", "raw_index", "delta"] + [f"{x}_label" for x in run.CLASSIFY]
             + [f"{x}_top1_share" for x in run.CLASSIFY]].to_string())


if __name__ == "__main__":
    main()
