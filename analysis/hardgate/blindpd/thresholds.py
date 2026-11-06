"""Data-defined kappa threshold (fixed rule; the value is computed from our own spectra, no published value is read).

Rule v1: pool kappa = K2/(2 Delta) of all K2 eigenmodes with Delta <= Delta_max over pool_sizes;
histogram log10(kappa) with bin width 0.25 on [-12, 1]; inside search_range pick the bin with the
lowest count (ties -> lowest kappa); threshold = 10**(bin centre). Zero-K2 modes are excluded.
"""
import numpy as np


def valley_threshold(kappas, search_range=(1e-3, 10 ** -0.5), width=0.25):
    k = np.asarray(kappas)
    k = k[k > 0]
    lk = np.log10(k)
    edges = np.arange(-12.0, 1.0 + 1e-9, width)
    h, _ = np.histogram(lk, edges)
    lo, hi = np.log10(search_range[0]), np.log10(search_range[1])
    cand = [(h[i], edges[i]) for i in range(len(h)) if edges[i] >= lo - 1e-9 and edges[i + 1] <= hi + 1e-9]
    cmin = min(c for c, _ in cand)
    left = min(e for c, e in cand if c == cmin)
    thr = 10 ** (left + width / 2)
    return float(thr), dict(bin_left=float(left), bin_count=int(cmin),
                            histogram={f"{e:.2f}": int(c) for e, c in zip(edges[:-1], h) if c})
