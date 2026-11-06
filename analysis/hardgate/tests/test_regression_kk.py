"""Gate: recomputed channel K2 eigenvalues reproduce production kk/projections (all N, all channels)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blindpd.io import SizeData
from blindpd.kk import block_kk

def run(cfg):
    report = {}
    for N in cfg["sizes"]:
        sd = SizeData(os.path.join(cfg["kk_export"], f"N{N}"), os.path.join(cfg["physics"], f"N{N}"), N)
        prod = {}
        for p in sd.meta["production_projections"]:
            if abs(p["threshold"] - 0.25) < 1e-12:
                prod[(p["block"], int(round(p["C2"])))] = p
        maxdiff = 0.0; nch = 0; idmismatch = 0; offmax = 0.0
        for key in sd.blocks:
            a, b, L = int(key[1]), int(key[4]), int(key.split("L")[1])
            blk = block_kk(sd, a, b, L)
            for ch in blk["channels"]:
                p = prod.get((key, ch["C2"]))
                if p is None:
                    continue
                if list(ch["guarded"] + 1) != list(p["source_ids"]):
                    idmismatch += 1
                if p["k2_values"] is not None and "k2_values" in ch:
                    d = np.max(np.abs(np.sort(ch["k2_values"]) - np.sort(p["k2_values"])))
                    maxdiff = max(maxdiff, d / max(1.0, np.max(np.abs(p["k2_values"])))); nch += 1
                offmax = max(offmax, ch.get("sector_coupling_max", 0.0))
        report[N] = dict(channels=nch, max_rel_k2_diff=maxdiff, guarded_id_mismatch=idmismatch, max_sector_coupling=offmax)
        print(N, report[N], flush=True)
    return report

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    r = run(cfg)
    ok = all(v["max_rel_k2_diff"] < 1e-9 and v["guarded_id_mismatch"] == 0 for v in r.values())
    print("REGRESSION", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
