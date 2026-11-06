"""Step 5: finite-size fits (fixed windows, set in the config): main a+b/N on main sizes; reference windows; model diff."""
import numpy as np


def ols(x, y):
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(x) - 2
    if dof <= 0:
        return beta, np.nan
    r = y - X @ beta
    s2 = r @ r / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, float(np.sqrt(cov[0, 0]))


def fit_series(Ns, vals, cfg):
    d = dict(zip(Ns, vals))
    out = dict(fit_status="insufficient_points")
    main = [n for n in cfg["main_fit_sizes"] if n in d and np.isfinite(d[n])]
    if len(main) < 3:
        return out
    x = 1 / np.array(main, float); y = np.array([d[n] for n in main])
    beta, se = ols(x, y)
    out.update(fit_status="ok" if len(main) == len(cfg["main_fit_sizes"]) else "partial_main_window",
               fit_Ns=";".join(map(str, main)), intercept=float(beta[0]), slope=float(beta[1]), SE=se)
    shifts = []
    for w in cfg["window_fit_sizes"]:
        ws = [n for n in w if n in d and np.isfinite(d[n])]
        if len(ws) >= 3:
            b, _ = ols(1 / np.array(ws, float), np.array([d[n] for n in ws]))
            shifts.append(abs(b[0] - beta[0]))
    out["sys_window"] = float(max(shifts)) if shifts else np.nan
    out["err_main"] = float(np.hypot(se, out["sys_window"])) if shifts and np.isfinite(se) else np.nan
    if len(main) >= 4:
        Q = np.vstack([np.ones_like(x), x, x ** 2]).T
        bq, *_ = np.linalg.lstsq(Q, y, rcond=None)
        out["quadratic_intercept"] = float(bq[0]); out["model_diff"] = float(abs(bq[0] - beta[0]))
        # reported separately; err_main deliberately excludes the model difference (convention v1)
        out["err_with_model"] = float(np.hypot(out["err_main"], out["model_diff"])) if np.isfinite(out.get("err_main", np.nan)) else np.nan
        out["model_dominated"] = bool(np.isfinite(out.get("err_main", np.nan)) and out["model_diff"] > out["err_main"])
    return out
