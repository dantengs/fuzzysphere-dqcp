"""Pre-registered kappa threshold from the raw KK of a pool of sizes (rule A, blindpd.thresholds valley rule).

    python -X utf8 -B tools/kappa_threshold.py KAPPA_POOL OUT.json [SIZES]      (cwd: analysis/hardgate)

KAPPA_POOL holds kk/N<n>/ (KK export: meta.json + *.f64) and physics/N<n>/parity.csv for every n of the pool;
SIZES is a comma list (default 6,7,8,9,10).  Pool = kappa = K2/(2 Delta_bar) of every KK sector mode with
Delta_bar <= 6.25 of every block; the threshold is the centre of the lowest-count log10 bin in [1e-3, 10**-0.5]
(blindpd/thresholds.py).  No P/D label enters.  To analyse a size with rule C (analysis/switch_kk/) blindly, build
the pool from other sizes and freeze OUT.json (threshold_from_raw.json) before the analysis reads it.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindpd.io import SizeData  # noqa: E402
from blindpd.kk import block_kk  # noqa: E402
from blindpd.thresholds import valley_threshold  # noqa: E402

POOL_SIZES = (6, 7, 8, 9, 10)
DELTA_MAX = 6.25


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def threshold(pool_root, sizes=POOL_SIZES):
    pool = []
    for n in sizes:
        sd = SizeData(str(pool_root / f"kk/N{n}"), str(pool_root / f"physics/N{n}"), n)
        for key in sd.blocks:
            a, b, L = int(key[1]), int(key[4]), int(key.split("L")[1])
            for ch in block_kk(sd, a, b, L)["channels"]:
                for sec in ch["sectors"]:
                    pool.extend(sec["kappa"][sec["Delta_bar"] <= DELTA_MAX])
    value, info = valley_threshold(pool)
    return value, dict(info, pool_sizes=list(sizes), pool_modes=len(pool), source="raw KK only")


def main(pool_root, out, sizes=None):
    sizes = POOL_SIZES if sizes is None else tuple(int(s) for s in sizes.split(","))
    thr, info = threshold(Path(pool_root), sizes)
    Path(out).write_text(json.dumps(clean(dict(threshold=thr, rule=info)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"kappa_c = {thr}  ({info['pool_modes']} modes, bin_left {info['bin_left']}, count {info['bin_count']})")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:4])
