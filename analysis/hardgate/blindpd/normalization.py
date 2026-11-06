"""Normalisation ledger: every constant that enters a reported Delta or lambda, per N, with definition and source.

Outputs (written by run.py):
  normalization_constants.csv : one row per (N, quantity)
  cg_table.csv                : SO(5) CG factors used (insertion, sector, rank, X_C2, Z_C2) -> CG_iso, signed CG, status
  convention_factors.csv      : N-independent factors (spin factors c_l, 3j factor, stress factor)
"""
import os, re, json
import numpy as np
import pandas as pd

DEFS = {
    "g": ("coupling V/U at which 2(E_T-E0)/(E_J-E0)=3", "coverage_run.toml"),
    "E0": ("ground-state energy (lowest C2=0 state of (0,0) L0)", "kk/calibration.ckpt"),
    "alpha": ("energy->Delta scale, alpha = 2/(E_J-E0)", "kk/calibration.ckpt"),
    "generator_c1": ("KK generator coefficient of O1 (vacuum-image minimised, |<dS|L|S>|=sqrt(2 Delta_S))", "kk/calibration.ckpt"),
    "generator_c2": ("KK generator coefficient of O2", "kk/calibration.ckpt"),
    "generator_ratio_c2_over_c1": ("ratio fixed by the vacuum-image minimisation", "derived"),
    "Z_phi": ("phi normalisation |<phi|n_Gamma2|0>| (vector density Gamma^2)", "ope3.toml phi_normalization"),
    "Z_T": ("T normalisation |<T|Q_G|0>| (tensor insertion), T phase fixed by Z_T > 0", "ope3.toml Z_T"),
    "Z_S_raw": ("S normalisation <S|O_S|0> = Z_S_raw (|Z_S|=1 after beta calibration)", "ope3.toml Z_S_raw"),
    "beta_1": ("O_S = beta_1 rho'rho + beta_2 Delta'Delta (two conditions: <stress|O_S|0>=0, <S|O_S|0>=+-1)", "ope3.toml beta"),
    "beta_2": ("second coefficient of O_S", "ope3.toml beta"),
    "O_S_vacuum_expectation": ("<0|O_S|0>, subtracted as <O_S>_vac <Z|X>", "ope3.toml vacuum_expectation_Os"),
    "kappa_threshold": ("primary criterion kappa = K2/(2 Delta) <= threshold (pooled N6-10 valley rule)", "pipeline"),
}


def per_N_constants(sds, physics_root, thr, cfg_sizes):
    rows = []
    for N in cfg_sizes:
        sd = sds[N]
        t = open(os.path.join(physics_root, f"N{N}", "ope3.toml")).read()
        cr = open(os.path.join(physics_root, f"N{N}", "coverage_run.toml")).read()
        get = lambda pat, txt: re.search(pat, txt, re.M).group(1)
        beta = [float(x) for x in get(r"^beta = \[([^\]]+)\]", t).split(",")]
        vals = dict(g=float(get(r"^g = (\S+)", cr)), E0=sd.E0, alpha=sd.alpha, generator_c1=float(sd.coeff[0]),
                    generator_c2=float(sd.coeff[1]), generator_ratio_c2_over_c1=float(sd.coeff[1] / sd.coeff[0]),
                    Z_phi=float(get(r"^phi_normalization = (\S+)", t)), Z_T=float(get(r"^Z_T = (\S+)", t)),
                    Z_S_raw=float(get(r"^Z_S_raw = (\S+)", t)), beta_1=beta[0], beta_2=beta[1],
                    O_S_vacuum_expectation=float(get(r"^vacuum_expectation_Os = (\S+)", t)), kappa_threshold=thr)
        for k, v in vals.items():
            d, s = DEFS[k]
            rows.append(dict(N=N, quantity=k, value=v, definition=d, source=s))
    return pd.DataFrame(rows)


def cg_table(rows_prod):
    i = rows_prod
    g = i.groupby(["namespace", "insertion", "sector", "rank", "X_C2", "Z_C2"])
    out = g.agg(cg_iso=("cg_iso", lambda x: ";".join(sorted(set(f"{v:.12g}" for v in x.dropna())))),
                cg_iso_signed=("cg_iso_signed", lambda x: ";".join(sorted(set(f"{v:.12g}" for v in x.dropna())))),
                cg_status=("cg_status", lambda x: ";".join(sorted(set(str(v) for v in x if isinstance(v, str))))),
                cg_key=("cg_key", lambda x: ";".join(sorted(set(str(v) for v in x if isinstance(v, str)))))).reset_index()
    return out


def convention_factors():
    rows = []
    for l in range(0, 5):
        c = np.sqrt(np.prod(np.arange(2 * l + 1, 0, -2, dtype=float)) / np.prod(np.arange(1, l + 1, dtype=float)))
        rows.append(dict(name=f"c_{l}", value=c, definition="spin factor c_l = sqrt((2l+1)!!/l!) (canonical-I STF extraction)"))
    rows += [dict(name="lambda_3j/lambda_iso", value=np.nan, definition="sqrt(dim Z) (permutation-symmetric 3j convention)"),
             dict(name="stress_extra_factor", value=1.0, definition="canonical-I stress normalisation: no extra factor beyond c_2")]
    # v0.5.1: conversions to other published conventions moved to comparison/unblind.py (not used by this package)
    return pd.DataFrame(rows)
