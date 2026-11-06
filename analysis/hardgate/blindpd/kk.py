"""Step 2: K^dagger K per block / Casimir channel / (X,P) sector, and primary subspaces.

Mathematics follows the frozen Tcc kk_classify (v0.2.7):
  K2 = sum_lf (2lf+1)/(2li+1) K^T K,  K = M[t,s] * [E_t < E_s - atol]  (lowering part of the
  calibrated reduced generator), on the guarded source set (E_s < every target frontier - atol).
The blind additions are: sectors split by exchange X and physical parity P, the dimensionless
kappa = K2 / (2 Delta), and a data-defined kappa threshold (see thresholds.py).
"""
import numpy as np


def required_lf(li):
    return [1] if li == 0 else [li - 1, li, li + 1]


def _ids(C, c, tol=1e-5):
    return np.where(np.abs(np.asarray(C) - c) < tol)[0]


def block_kk(sd, a, b, li):
    key = f"s{a}_s{b}_L{li}"
    blk = sd.blocks[key]
    E = np.array(blk["energy"]); C2 = np.array(blk["c2"]); X = np.array(blk["x"])
    P = sd.parity[key]
    edges = [sd.edge_arrays(a, b, li, lf) for lf in required_lf(li) if (a, b, li, lf) in sd.edges]
    missing = [lf for lf in required_lf(li) if (a, b, li, lf) not in sd.edges]
    out = dict(key=key, a=a, b=b, L=li, E=E, C2=C2, X=X, P=P, missing_L=missing, channels=[])
    for e in edges:
        if np.max(np.abs(e["sE"] - E)) > 0:
            raise ValueError(f"edge source energies differ from block {key}")
    for c in sorted(set(int(round(x)) for x in C2)):
        allids = _ids(C2, c)
        if missing:
            guarded = np.array([], int)
        else:
            def ok(i):
                for e in edges:
                    if not (E[i] < e["frontier"] - e["atol"]):
                        return False
                return True
            guarded = np.array([i for i in allids if ok(i)], int)
        ch = dict(C2=c, all_ids=allids, guarded=guarded, excluded=np.setdiff1d(allids, guarded), sectors=[])
        if guarded.size:
            K2 = np.zeros((guarded.size, guarded.size))
            for e in edges:
                ti = _ids(e["tC2"], c)
                if ti.size == 0:
                    continue
                Mc = e["M"][np.ix_(ti, guarded)]
                low = e["tE"][ti][:, None] < E[guarded][None, :] - e["atol"]
                K = Mc * low
                K2 += (2 * e["lf"] + 1) / (2 * li + 1) * (K.T @ K)
            ch["K2"] = K2
            ch["k2_values"] = np.linalg.eigvalsh(K2)
            # sector split by (X, P); check K2 does not couple sectors
            labs = list(zip(X[guarded], P[guarded]))
            secs = sorted(set(labs))
            offmax = 0.0
            for s1 in secs:
                m1 = np.array([l == s1 for l in labs])
                offmax = max(offmax, float(np.max(np.abs(K2[np.ix_(m1, ~m1)]), initial=0.0)))
            ch["sector_coupling_max"] = offmax
            for s in secs:
                m = np.array([l == s for l in labs])
                ids = guarded[m]
                Ks = K2[np.ix_(m, m)]
                w, U = np.linalg.eigh(Ks)
                w = np.where(w < 0, np.maximum(w, -1e-8), w)
                Ebar = np.einsum("ij,i,ij->j", U, E[ids], U)
                Dl = sd.delta(Ebar)
                kappa = np.where(w < 1e-12, 0.0, w / (2 * np.maximum(Dl, 1e-12)))
                ch["sectors"].append(dict(X=int(s[0]), P=int(s[1]), ids=ids, K2=Ks, k2=w, U=U,
                                          Delta_bar=Dl, kappa=kappa))
        out["channels"].append(ch)
    return out


def primary_modes(sd, blk, sec, kappa_thr):
    """Primary subspace {kappa <= thr}, then H diagonalised inside it."""
    E = blk["E"][sec["ids"]]
    keep = sec["kappa"] <= kappa_thr
    Q = sec["U"][:, keep]
    if Q.shape[1] == 0:
        return [], Q
    Hq = Q.T @ np.diag(E) @ Q
    ev, V = np.linalg.eigh(Hq)
    Q = Q @ V
    modes = []
    for j in range(Q.shape[1]):
        q = Q[:, j]
        res = np.linalg.norm(E * q - ev[j] * q)
        w = q ** 2
        k2 = float(q @ sec["K2"] @ q)
        Dj = float(sd.delta(ev[j]))
        modes.append(dict(energy=float(ev[j]), Delta=Dj, sigma_Delta=float(sd.alpha * res), K2=k2,
                          kappa=0.0 if k2 < 1e-12 else k2 / (2 * max(Dj, 1e-12)),
                          dominant_index=int(sec["ids"][np.argmax(w)]), dominant_weight=float(np.max(w)),
                          vec=q))
    return modes, Q
