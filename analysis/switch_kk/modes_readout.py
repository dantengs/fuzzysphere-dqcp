"""Post-freeze: read out the central low-K^dagger K subspace mode by mode (one size N).

The central subspace of each (block, C2, X, P) sector is spanned by `modes.csv` vectors (energy-rotated, one
row per method and mode).  A mode is `concentrated` when one raw level carries >= GATE of it, otherwise
`dispersed` across several raw levels.  For every mode we list both energies: the raw level energy of the
dominant component and the projected mode energy <v|D|v>.  Reads only results/ (hash-checked).  No literature
values or labels."""

import json

import numpy as np
import pandas as pd

import run
from spectrum import irrep

R = run.ROOT / "results"
receipt = json.loads((R / "receipt.json").read_text(encoding="utf-8"))
for name, digest in receipt["outputs"].items():
    if run.sha(R / name) != digest:
        raise ValueError(f"{name} differs from its receipt")
modes = pd.read_csv(R / "modes.csv")
states = pd.read_csv(R / "states.csv").set_index(["block", "raw_index"])
key = ["block", "L", "so5_c2", "X", "P", "mode"]
vectors = {(r.method, r.block, r.so5_c2, r.X, r.P, r.mode): json.loads(r.vector_json) for r in modes.itertuples()}

rows = []
for r in modes[modes.method == "hard"].itertuples():
    v = vectors[("hard", r.block, r.so5_c2, r.X, r.P, r.mode)]
    a = np.array(list(v.values()))
    cos2 = {}
    for other in ("switch_0.25", "switch_0.50"):
        w = vectors.get((other, r.block, r.so5_c2, r.X, r.P, r.mode))
        # (template) no such mode in the other method (its central subspace is smaller): no matching direction
        cos2[other] = 0.0 if w is None else float((a @ np.array([w.get(k, 0.0) for k in v])) ** 2)
    parts = sorted(((int(k), c * c) for k, c in v.items()), key=lambda kv: -kv[1])
    dom = states.loc[(r.block, r.dominant_index)]
    rows.append(dict(
        **irrep(int(r.so5_c2), int(r.block.split("_")[1][1:])),
        L=r.L, P=r.P, block=r.block, mode=r.mode,
        kind="concentrated" if r.dominant_weight >= run.GATE else "dispersed",
        mode_delta=r.delta, sigma_delta=r.sigma_delta,
        dominant_index=r.dominant_index, dominant_delta=float(dom.delta), dominant_weight=r.dominant_weight,
        dominant_support=dom.support,
        components=";".join(f"#{k}:{float(states.loc[(r.block, k)].delta):.4f}:{w:.3f}" for k, w in parts if w >= 0.02),
        band_min_weight_classify=r.band_min_weight_classify,
        min_cos2_vs_switch=min(cos2.values()),
    ))
out = pd.DataFrame(rows).sort_values(["kind", "dim", "L", "mode_delta"]).reset_index(drop=True)
out.to_csv(R / "mode_levels.csv", index=False, lineterminator="\n")
counts = dict(
    central_modes={m: int((modes.method == m).sum()) for m in run.CLASSIFY},
    concentrated={m: int(((modes.method == m) & (modes.dominant_weight >= run.GATE)).sum()) for m in run.CLASSIFY},
    dispersed_sector_sets_identical=all(
        set(map(tuple, modes[(modes.method == m) & (modes.dominant_weight < run.GATE)][key].values))
        == set(map(tuple, modes[(modes.method == "hard") & (modes.dominant_weight < run.GATE)][key].values))
        for m in run.CLASSIFY),
    min_cos2_vs_switch=float(out.min_cos2_vs_switch.min()),
    min_band_weight=float(out.band_min_weight_classify.min()),
    max_concentrated_mode_minus_raw=float((out[out.kind == "concentrated"].mode_delta
                                           - out[out.kind == "concentrated"].dominant_delta).abs().max()),
    mode_levels_sha256=run.sha(R / "mode_levels.csv"),
)
(R / "mode_levels_summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(counts, indent=2))
print("| 表示 | L | P | 模式 Δ | σ_Δ | 组成（原始能级 Δ: 权重） |")
print("|---|---:|---:|---:|---:|---|")
for r in out[out.kind == "dispersed"].sort_values("mode_delta").itertuples():
    comp = "; ".join(f"{c.split(':')[1]}: {c.split(':')[2]}" for c in r.components.split(";"))
    print(f"| {r.irrep} | {r.L} | {'+' if r.P > 0 else '−'} | {r.mode_delta:.4f} | {r.sigma_delta:.3f} | {comp} |")
