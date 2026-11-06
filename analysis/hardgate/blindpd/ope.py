"""Step 6: re-attach finite-N three-point matrix elements to blind primaries and fit across N.

The per-N normalised coefficients come from the production reduction (fullQ integer sector and
half-integer local sector) in convention 'canonical_I_c_l_no_extra_stress_factor':
  lambda_iso  : unitary SO(5) CG (Wigner-Eckart reduced)  <- primary convention of this pipeline
  lambda_3j   : sqrt(dim Z) * lambda_iso (permutation-symmetric)
No published 'trace' convention is used. External states are re-identified by matching
(N, C2, L, P, Delta) to the blind primary modes; rows whose X or Z is not a blind primary are kept
but excluded from curves.
"""
import numpy as np
import pandas as pd
from .fits import fit_series

COLS = ["N", "namespace", "sector", "route", "insertion", "insertion_rep", "rank", "X_C2", "X_L", "X_P", "X_Delta",
        "X_K2", "X_sigmaE", "Z_C2", "Z_L", "Z_P", "Z_Delta", "Z_K2", "Z_sigmaE", "coefficient_kind",
        "lambda_iso_magnitude", "lambda_iso_component_signed", "lambda_3j_magnitude", "lambda_3j_component_signed",
        "lambda_iso_route", "lambda_3j",
        "normalization_Z", "spin_factor", "cg_iso", "cg_status", "normalization_status", "value_status",
        "parity_status", "physical_publication_allowed", "physical_publication_status", "normalization_convention",
        "raw_row_id"]


def load_rows(path):
    df = pd.read_csv(path, low_memory=False)
    assert not any("paper" in c.lower() for c in COLS), "blind guard: paper-derived column requested"
    return df[[c for c in COLS if c in df.columns]].copy()


def match_states(rows, prim, tol_exact=1e-6, tol_approx=2e-3):
    """prim: DataFrame with N, C2, L, P, Delta, phys_id, track_id."""
    idx = {}
    for k, g in prim.groupby(["N", "C2", "L", "P"]):
        idx[k] = g[["Delta", "phys_id", "track_id"]].values
    out = {}
    for side in ("X", "Z"):
        ids, trk, st, dd = [], [], [], []
        for N, c, L, P, D in zip(rows.N, rows[f"{side}_C2"], rows[f"{side}_L"], rows[f"{side}_P"], rows[f"{side}_Delta"]):
            try:
                key = (int(N), int(round(float(c))), int(L), int(P))
            except (TypeError, ValueError):
                ids.append(None); trk.append(None); st.append("bad_key"); dd.append(np.nan); continue
            arr = idx.get(key)
            if arr is None or not len(arr):
                ids.append(None); trk.append(None); st.append("no_blind_primary_in_sector"); dd.append(np.nan); continue
            dd_all = np.abs(arr[:, 0].astype(float) - float(D))
            j = int(np.argmin(dd_all))
            d = abs(float(arr[j, 0]) - float(D))
            s = "exact_mode" if d < tol_exact else ("approx_mode" if d < tol_approx else "not_a_blind_primary")
            # v0.3.2: an energy match is an identity only if no second candidate lies inside the same tolerance
            # (copies of the same physical state in several SO(4) sectors share one phys_id and are not ambiguous)
            if s != "not_a_blind_primary" and \
                    len(set(arr[dd_all < (tol_exact if s == "exact_mode" else tol_approx), 1])) > 1:
                s = "ambiguous_energy_match"
            ok_ = s in ("exact_mode", "approx_mode")
            ids.append(arr[j, 1] if ok_ else None)
            trk.append(arr[j, 2] if ok_ else None)
            st.append(s); dd.append(d)
        out[side] = (ids, trk, st, dd)
    for side in ("X", "Z"):
        rows[f"{side}_phys_id"], rows[f"{side}_track"], rows[f"{side}_match"], rows[f"{side}_match_dDelta"] = out[side]
    return rows


def coefficient(rows):
    signed = rows.coefficient_kind.eq("signed_unique_invariant")
    # v0.3.1: the signed invariant is the release's lambda_iso_route (S: me/Z_S_raw; diagonal T: signed CG),
    # not lambda_iso_component_signed (= me/|Z|/|CG|, a raw component whose sign differs for S and negative CGs).
    rows["lambda_iso_blind"] = np.where(signed, rows.lambda_iso_route, rows.lambda_iso_magnitude)
    rows["lambda_3j_blind"] = np.where(signed, rows.lambda_3j, rows.lambda_3j_magnitude)
    rows["phase_convention"] = np.where(signed, "signed_unique_invariant", "magnitude_phase_not_fixed")
    return rows


def curves(rows, cfg):
    ok = rows.X_track.notna() & rows.Z_track.notna() & rows.normalization_status.eq("ok") & \
         rows.physical_publication_allowed.astype(str).eq("True") & np.isfinite(rows.lambda_iso_blind.astype(float))
    use = rows[ok].copy()
    out = []
    keys = ["namespace", "route", "rank", "X_track", "Z_track"]
    for k, g in use.groupby(keys):
        perN = g.groupby("N")
        Ns, li, l3, spread = [], [], [], []
        for N, h in perN:
            Ns.append(int(N)); li.append(float(np.median(h.lambda_iso_blind))); l3.append(float(np.median(h.lambda_3j_blind)))
            spread.append(float(h.lambda_iso_blind.max() - h.lambda_iso_blind.min()))
        f_iso = fit_series(Ns, li, cfg); f_3j = fit_series(Ns, l3, cfg)
        r = dict(zip(keys, k))
        r.update(Ns=";".join(map(str, Ns)), nN=len(Ns), phase_convention=g.phase_convention.iloc[0],
                 insertion_rep=g.insertion_rep.iloc[0], max_duplicate_spread=max(spread),
                 match_quality="exact" if (g.X_match.eq("exact_mode") & g.Z_match.eq("exact_mode")).all() else "approx",
                 **{f"lambda_iso_N{n}": v for n, v in zip(Ns, li)},
                 **{f"iso_{a}": b for a, b in f_iso.items()}, **{f"3j_{a}": b for a, b in f_3j.items()})
        if r.get("phase_convention") == "magnitude_phase_not_fixed" and r.get("iso_intercept", 1) < 0:
            r["iso_physical_status"] = "negative_magnitude_intercept_diagnostic_only"
        else:
            r["iso_physical_status"] = "finite_window_estimate"
        out.append(r)
    return pd.DataFrame(out)
