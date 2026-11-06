"""StageB sectors (v0.3): half-integer segment sector ('half') and extended (1,1) windows ('11ext').

StageBData mimics the SizeData interface so that kk.block_kk / spectrum.analyse_size run unchanged.
The sector is encoded as pair (9,9) for 'half' and (8,8) for '11ext' (block keys s9_s9_L<L>, s8_s8_L<L>).
Physical parity P of every window state is obtained from its segment PH label pseg through the map
(L, C2, pseg) -> P read off the production KK modes (which carry both labels); the map must be single-valued.
"""
import json, os
import numpy as np
import pandas as pd
from .io import SizeData, sha256_file

CODE = {"half": 9, "11ext": 8}


class StageBData(SizeData):
    def __init__(self, export_dir, N, sector):
        self.N = N; self.export_dir = export_dir; self.sector = sector; self._cache = {}
        m = json.load(open(os.path.join(export_dir, "stageB_meta.json"))); self.raw_meta = m
        cal = m["kk_modes"]["L0"]["calibration"]
        self.coeff = np.array(cal["coefficients"], float); self.alpha = float(cal["alpha"]); self.E0 = float(cal["E0"])
        a = CODE[sector]; self.a = a
        # parity map from production modes
        pmap = {}
        for k, km in m["kk_modes"].items():
            for md in km["modes"]:
                key = (int(km["L"]), int(md["c2"]), int(md["pseg"]))
                if key in pmap and pmap[key] != int(md["P"]):
                    raise ValueError(f"parity map not single-valued at {key}")
                pmap[key] = int(md["P"])
        self.pmap = pmap
        # rule check: P = pseg * s(L, C2)?  record the sign table
        # full windows (classification bases) and KK source subsets (blocks)
        self.windows = {}
        for wk, w in m["windows"].items():
            L = int(wk[1:]); E = np.array(w["energy"])
            if np.any(np.diff(E) < 0):
                raise ValueError(f"window {wk} not energy-ordered")
            self.windows[L] = dict(E=E, C2=np.array(w["c2"]), pseg=np.array(w["pseg"], int), dim=int(w["dim"]))
        src = {}
        for e in m["kk_edges"]:
            li = int(e["li"]); sE = np.array(e["source_E"])
            if li in src and (len(src[li]) != len(sE) or np.max(np.abs(src[li] - sE)) > 0):
                raise ValueError(f"inconsistent KK source sets at L{li}")
            src[li] = sE
        self.src_idx, self.blocks, self.parity = {}, {}, {}
        for L, sE in src.items():
            W = self.windows[L]
            idx = np.array([int(np.argmin(np.abs(W["E"] - x))) for x in sE])
            if np.max(np.abs(W["E"][idx] - sE)) > 1e-10 or np.any(np.diff(idx) <= 0):
                raise ValueError(f"KK sources are not a sub-window at L{L}")
            self.src_idx[L] = idx
            self.blocks[f"s{a}_s{a}_L{L}"] = dict(energy=list(sE), c2=list(W["C2"][idx]), x=[0] * len(idx),
                                                  dim=W["dim"], pseg=list(W["pseg"][idx]))
        self.meta = dict(calibration=dict(coefficients=list(self.coeff), alpha=self.alpha, E0=self.E0), blocks=self.blocks)
        self.edges = {}
        for e in m["kk_edges"]:
            li, lf = int(e["li"]), int(e["lf"])
            bkey = f"s{a}_s{a}_L{lf}"
            reused = bkey in self.blocks and len(e["target_E"]) == len(self.blocks[bkey]["energy"]) and \
                np.max(np.abs(np.array(e["target_E"]) - np.array(self.blocks[bkey]["energy"]))) < 1e-12
            self.edges[(a, a, li, lf)] = dict(e, a=a, b=a, key=e["name"],
                                              target_status="reused_run_window" if reused else "stageB_full_window",
                                              target_full=False)
        self.input_hashes = {"stageB_meta.json": sha256_file(os.path.join(export_dir, "stageB_meta.json"))}

    def assign_parity(self, rule):
        """rule(L, c2, pseg) -> P ; applied to every window state and to the KK source subsets."""
        for L, W in self.windows.items():
            W["P"] = np.array([rule(L, int(round(c)), int(p)) for c, p in zip(W["C2"], W["pseg"])], int)
        for key in self.blocks:
            L = int(key.split("L")[1]); self.parity[key] = self.windows[L]["P"][self.src_idx[L]]

    def window_basis(self, L):
        """(E, C2, X, P, frontier) of the full window used as classification basis."""
        W = self.windows[L]
        fr = [e["target_frontier"] for e in self.raw_meta["kk_edges"] if int(e["lf"]) == L and e["target_frontier"] is not None]
        return W["E"], W["C2"], np.zeros(len(W["E"]), int), W["P"], (max(fr) if fr else None)


def parity_rule_from_map(pmap):
    """Find s(L, C2) with P = pseg * s; require consistency over all observed (L, C2, pseg)."""
    s = {}
    for (L, c2, ps), P in pmap.items():
        v = P * ps
        if (L, c2) in s and s[(L, c2)] != v:
            raise ValueError(f"P/pseg relation not a sign at L={L}, C2={c2}")
        s[(L, c2)] = v
    return s


def gauge_check(sdb):
    """Reduced Hermitian conjugacy between forward/reverse KK edges inside one stageB sector (same eigensolve):
    rev.M[t(u), i] = (-1)^(L'-L) sqrt((2L'+1)/(2L+1)) fwd.M[j(i), u]  for sources u (at L) and i (at L')."""
    from .spectrum import conj_factor
    worst, n = 0.0, 0
    for (a, b, L, Lp), e in sdb.edges.items():
        if Lp > 4 or L == Lp or (a, b, Lp, L) not in sdb.edges:
            continue
        F = sdb.edge_arrays(a, b, L, Lp)["M"]; R = sdb.edge_arrays(a, b, Lp, L)["M"]
        jdx = sdb.src_idx[Lp]; tdx = sdb.src_idx[L]
        A = R[np.ix_(tdx, np.arange(len(jdx)))]          # rows: sources at L (as window states), cols: sources at L'
        Bm = F[np.ix_(jdx, np.arange(len(tdx)))].T * conj_factor(L, Lp)
        scale = max(1.0, float(np.max(np.abs(A))))
        worst = max(worst, float(np.max(np.abs(A - Bm))) / scale); n += A.size
    return dict(conjugacy_max_rel_residual=worst, checked_elements=n)


def kk_regression(sdb, thr=0.25):
    """Recompute the production stageB KK modes (K2 <= thr per (C2, P) sector, H diagonalised) and compare
    energies and coefficient vectors with the exported production modes."""
    from .kk import block_kk
    worstE = worstO = 0.0; n = 0; tot = 0
    for L in range(5):
        if f"s{sdb.a}_s{sdb.a}_L{L}" not in sdb.blocks:
            continue
        blk = block_kk(sdb, sdb.a, sdb.a, L); E = blk["E"]; W = sdb.windows[L]; idx = sdb.src_idx[L]
        prod = sdb.raw_meta["kk_modes"][f"L{L}"]["modes"]; tot += len(prod)
        for ch in blk["channels"]:
            for sec in ch["sectors"]:
                Q = sec["U"][:, sec["k2"] <= thr]
                if Q.shape[1] == 0:
                    continue
                ev, V = np.linalg.eigh(Q.T @ np.diag(E[sec["ids"]]) @ Q); Q = Q @ V
                F = np.zeros((len(W["E"]), Q.shape[1])); F[idx[sec["ids"]], :] = Q
                for j in range(F.shape[1]):
                    c = [m for m in prod if m["c2"] == ch["C2"] and m["P"] == sec["P"]]
                    if not c:
                        continue
                    m = min(c, key=lambda m: abs(m["energy"] - ev[j]))
                    worstE = max(worstE, abs(m["energy"] - ev[j]))
                    worstO = max(worstO, 1 - abs(float(np.array(m["coeff"]) @ F[:, j]))); n += 1
    return dict(modes_reproduced=n, production_modes=tot, max_energy_diff=worstE, max_one_minus_overlap=worstO)


def blind_modes_window_coords(sdb, mode_vecs):
    """analyse_size returns mode vectors in KK-source (block) coordinates; OPE edges use full-window coordinates."""
    out = []
    for m in mode_vecs:
        if m is None:
            continue
        L = m["L"]; W = sdb.windows[L]; v = np.zeros(len(W["E"])); v[sdb.src_idx[L]] = m["vec"]
        out.append(dict(m, vec=v))
    return out
