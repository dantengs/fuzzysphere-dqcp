#!/usr/bin/env python3
"""Attach physical parities (bench/parity_run.jl output) to the phi x phi spin towers of bench/ope3_run.jl (v0.2.4).

  python3 ope3_parity_join.py OPE3_DIR PARITY_DIR [OUT_CSV]

* integer-sector raw rows  : parity of the same (0,0) block state  (L, C2, energy)
* integer-sector KK rows   : parity_kk.csv of (0,0) L, threshold, C2, mode
* half-sector rows (1/2,1/2): the half sector holds other SO(5) components of multiplets that also live in
  integer blocks (10 in (0,1), 14 in (0,0), ...) with exactly the same energy; their parity is taken from
  any integer block/target state with the same (L, C2) and |dE| < tol.  Rows without such a partner are
  reported as "unmatched" (a direct half-sector P would need an extra SU(2) rotation, not implemented).
"""
import csv, sys, collections, os

ope3, pdir = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ope3, "ope3_phiphi_spin_parity.csv")
tol = 1e-7

def c2i(x):
    v = float(x); r = round(v)
    return r if abs(v - r) < 1e-4 else None

states = collections.defaultdict(list)                 # (L, C2) -> [(energy, P, block, window)]
for r in csv.DictReader(open(os.path.join(pdir, "parity.csv"))):
    c = c2i(r["C2"])
    if c is None: continue
    states[(int(r["L"]), c)].append((float(r["energy"]), int(r["P"]), "s%s_s%s" % (r["sA"], r["sB"]), r["window"]))
kk = {}
kkf = os.path.join(pdir, "parity_kk.csv")
if os.path.exists(kkf):
    for r in csv.DictReader(open(kkf)):
        if (r["sA"], r["sB"]) == ("0", "0"):
            kk[(int(r["L"]), float(r["threshold"]), int(float(r["C2"])), int(r["mode"]))] = (r["P"], r["P_expectation"])

rows = list(csv.DictReader(open(os.path.join(ope3, "ope3_phiphi_spin.csv"))))
cnt = collections.Counter()
for r in rows:
    L = int(r["spin"]); c = c2i(r["C2"]); P = ""; how = ""; n = 0
    if r["basis"].startswith("kk_t"):
        t = float(r["basis"][4:]); lab = r["state"]                       # L{l}_C2_{c}_m{j}
        j = int(lab.rsplit("_m", 1)[1]); hit = kk.get((L, t, c, j))
        if hit: P, how, n = hit[0], "parity_kk", 1
        else: how = "no_kk_parity"
    elif c is not None:
        e = float(r["energy"])
        hits = [s for s in states.get((L, c), []) if abs(s[0] - e) < tol]
        n = len(hits); ps = sorted(set(h[1] for h in hits))
        if len(ps) == 1: P = str(ps[0]); how = "energy_match:" + ",".join(sorted(set(h[2] + "/" + h[3] for h in hits)))
        elif len(ps) > 1: P = "conflict"; how = "conflict"
        else: how = "unmatched"
    else:
        how = "C2_not_integer"
    r["P"] = P; r["P_source"] = how; r["P_matches"] = n
    cnt[(r["sector"], how.split(":")[0])] += 1
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for k, v in sorted(cnt.items()): print("OPE3_PARITY_JOIN", k[0], k[1], v)
bad = [r for r in rows if r["P"] == "conflict"]
print("OPE3_PARITY_JOIN rows=%d conflicts=%d unmatched_half=%d -> %s" % (len(rows), len(bad),
      sum(1 for r in rows if r["sector"] == "half" and r["P_source"] == "unmatched"), out))
