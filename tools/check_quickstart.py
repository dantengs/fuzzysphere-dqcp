#!/usr/bin/env python3
"""Check a finished N = 5 quick start (configs/quickstart_N5.toml) against its expected properties and reference values.

    python tools/check_quickstart.py <data root>/results/quickstart/N5

Standard library only.  Every line prints the measured value, what is expected and PASS/FAIL; the exit code is 1 if any
check fails.  Reference values come from independent N = 5 runs made when this template was published (they agree with
each other far inside the stated tolerances); signs of matrix elements are arbitrary and never compared.
"""
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REF = {"F": -3.521338602929802e-07, "alpha": 0.4850239285560828, "Delta_T": 2.9999996, "primary_candidates": 171,
       "kk_modes": 639}


def toml_scalars(path):
    """key = number / "string" lines of a flat TOML file (enough for the reports read here)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\s*"?([\w.]+)"?\s*=\s*(.+?)\s*$', line)
        if not m:
            continue
        k, v = m.groups()
        if v.startswith('"'):
            out[k] = v.strip('"')
        else:
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def main(root):
    root = Path(root)
    rows, ok_all = [], True

    def check(name, value, expected, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        rows.append((name, value, expected, "PASS" if ok else "FAIL"))

    rep = toml_scalars(root / "run" / "physics_report.toml")
    check("F(g) = Delta_T - 3 (tuned g)", f"{rep['F']:.3e}", f"|F| < 1e-5; ref {REF['F']:.4e}",
          abs(rep["F"]) < 1e-5 and abs(rep["F"] - REF["F"]) < 1e-9)
    check("alpha = 2/(E_J - E_0)", f"{rep['alpha']:.12f}", f"ref {REF['alpha']:.12f}",
          abs(rep["alpha"] - REF["alpha"]) < 1e-10)

    summ = list(csv.DictReader(open(root / "run" / "summary.tsv", encoding="utf-8"), delimiter="\t"))
    f = lambda r, k: float(r[k])
    J = min(f(r, "delta") for r in summ if (r["sA"], r["sB"], r["L"]) == ("0", "1", "1") and abs(f(r, "C2") - 3) < 1e-6)
    T = min(f(r, "delta") for r in summ if (r["sA"], r["sB"], r["L"]) == ("0", "0", "2") and abs(f(r, "C2")) < 1e-6)
    check("Delta of J (lowest C2=3, (0,1) L1)", f"{J:.10f}", "2 (definition)", abs(J - 2) < 1e-9)
    check("Delta of T (lowest C2=0, (0,0) L2)", f"{T:.7f}", f"3 within 1e-5; ref {REF['Delta_T']}",
          abs(T - 3) < 1e-5 and abs(T - REF["Delta_T"]) < 1e-6)
    hres = max(f(r, "H_res") for r in summ); cres = max(f(r, "C2_res") for r in summ)
    check("max residual |Hv - Ev|", f"{hres:.1e}", "<= 1e-6 (gate), in practice ~1e-10", hres <= 1e-6)
    check("max residual |Cv - c2 v|", f"{cres:.1e}", "<= 1e-6 (gate)", cres <= 1e-6)

    groups = defaultdict(list)
    for r in summ:
        groups[(r["L"], round(f(r, "C2")))].append(((r["sA"], r["sB"]), f(r, "energy")))
    pairs, worst = 0, 0.0
    for items in groups.values():
        for i, (b1, e1) in enumerate(items):
            for b2, e2 in items[i + 1:]:
                if b1 != b2 and abs(e1 - e2) < 1e-6:
                    pairs += 1; worst = max(worst, abs(e1 - e2))
    check("one SO(5) multiplet in several (sA,sB) blocks", f"{pairs} pairs, max |dE| {worst:.1e}", "|dE| <= 1e-10",
          pairs > 0 and worst <= 1e-10)

    ope = toml_scalars(root / "run" / "ope" / "ope.toml")
    d = ope["half_vs_integer_phi_energy_difference"]
    check("phi from half-integer vs integer sector", f"{d:.1e}", "<= 1e-10", abs(d) <= 1e-10)

    par = list(csv.DictReader(open(root / "ope3" / "ope3_phiphi_spin_parity.csv", encoding="utf-8")))
    neg = [abs(float(r["ratio"])) for r in par if r["P"] == "-1" and r["ratio"] not in ("", "NaN")]
    check("phi x phi has no P = -1 content", f"{len(neg)} rows, max {max(neg, default=0):.1e}", "<= 1e-12",
          len(neg) > 0 and max(neg) <= 1e-12)

    cur = toml_scalars(root / "stageb_half" / "raw_current_control.toml")
    check("current in the half sector (Delta=2, P=+1)", cur.get("status"), "pass", cur.get("status") == "pass")

    ps = toml_scalars(root / "parity" / "parity_summary.toml")
    check("parity: max residual, mixed KK modes", f"{ps['max_residual']:.1e}, {int(ps['kk_modes_mixed_t0.25'])}",
          "<= 1e-6, 0", ps["max_residual"] <= 1e-6 and ps["kk_modes_mixed_t0.25"] == 0)
    check("KK modes with parity", int(ps["kk_modes"]), f"ref {REF['kk_modes']}", int(ps["kk_modes"]) == REF["kk_modes"])
    check("primary candidates at t = 0.25", int(rep["primary_candidates"]), f"ref {REF['primary_candidates']}",
          int(rep["primary_candidates"]) == REF["primary_candidates"])

    w = [max(len(str(r[i])) for r in rows) for i in range(3)]
    for r in rows:
        print(f"{r[3]}  {r[0]:<{w[0]}}  {str(r[1]):<{w[1]}}  expected {r[2]}")
    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
