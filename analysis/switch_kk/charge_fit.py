"""Post-freeze: large-charge form fits of the lowest L = 0 rank-q symmetric tensors (q, 0), one size N.

Delta_q = a q^{3/2} + b q^{1/2} + c.  The 3d large-charge EFT predicts c = -0.0937 (universal); the fit
leaves c free as a consistency check.  Single size, no finite-size extrapolation."""

import json
from pathlib import Path

import numpy as np

import run

summary = json.loads((run.ROOT / "results/levels_summary.json").read_text(encoding="utf-8"))
data = summary["lowest_symmetric_tensor_L0"]
q = np.arange(1, 6)
d = np.array([data[f"({k},0)"]["delta"] for k in q])
assert all(data[f"({k},0)"]["verdict"] == "robust" for k in q)
out = {"delta_q": dict(zip(map(int, q), d.tolist()))}
for name, use in (("q1-5", q >= 1), ("q2-5", q >= 2)):
    A = np.stack([q**1.5, q**0.5, np.ones_like(q, dtype=float)], 1)
    coef, *_ = np.linalg.lstsq(A[use], d[use], rcond=None)
    out[f"free_c_{name}"] = dict(a=coef[0], b=coef[1], c=coef[2],
                                 residuals=(d - A @ coef).round(4).tolist())
A = np.stack([q**1.5, q**0.5], 1)
coef, *_ = np.linalg.lstsq(A[q >= 2], d[q >= 2] + 0.0937, rcond=None)
out["fixed_c_q2-5"] = dict(a=coef[0], b=coef[1], c=-0.0937,
                           residuals=(d - A @ coef + 0.0937).round(4).tolist())
path = run.ROOT / "results/charge_fit.json"
path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
print(path.read_text(encoding="utf-8"))
