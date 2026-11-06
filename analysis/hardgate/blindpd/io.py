"""Input loading. Pure numpy/pandas; no network, no writes to inputs."""
import hashlib, json, os
import numpy as np
import pandas as pd


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


READ_LOG = {}          # v0.3.2: every binary input actually read -> sha256 (verified against the exporter's bin_sha256)


def read_f64(path, expected_sha=None):
    """Read a little-endian float64 payload; verify its content hash when the exporter recorded one (A5)."""
    h = sha256_file(path)
    if expected_sha is not None and h != expected_sha:
        raise ValueError(f"input binary content hash mismatch: {path}")
    READ_LOG[os.path.abspath(path)] = h
    return np.fromfile(path, dtype="<f8")


def tree_hashes(paths):
    """sha256 of every regular file below each path (file or directory), keyed by absolute path."""
    out = {}
    for p in paths:
        if os.path.isfile(p):
            out[os.path.abspath(p)] = sha256_file(p)
        elif os.path.isdir(p):
            for d, _, fs in os.walk(p):
                for f in sorted(fs):
                    q = os.path.join(d, f); out[os.path.abspath(q)] = sha256_file(q)
        else:
            raise FileNotFoundError(p)
    return out


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


BLOCK_TARGET_STATUS = ("reused_run_window", "block_top_frontier")   # v0.3.5: edge targets that are the source window itself


class SizeData:
    """All exported KK data of one N (one Tcc run directory).

    v0.3.5: a block entry with source_window == "target" (exporter v0.3.5) is a saved target window used as the KK
    source set; its X and P labels come from the parity.csv rows with window == "target"."""

    def __init__(self, export_dir, physics_dir, N):
        self.N = N
        self.export_dir = export_dir
        self.meta = load_json(os.path.join(export_dir, "meta.json"))
        cal = self.meta["calibration"]
        self.coeff = np.array(cal["coefficients"], float)
        self.alpha = float(cal["alpha"])
        self.E0 = float(cal["E0"])
        self.blocks = self.meta["blocks"]
        self.edges = {(e["a"], e["b"], e["li"], e["lf"]): e for e in self.meta["edges"]}
        self._cache = {}
        # physical (PH) parity of raw block states
        par = pd.read_csv(os.path.join(physics_dir, "parity.csv"))
        self.parity = {}
        for (a, b, L, w), g in par.groupby(["sA", "sB", "L", "window"]):
            key = f"s{a}_s{b}_L{L}"
            if key not in self.blocks:
                continue
            blk = self.blocks[key]
            if w != ("target" if blk.get("source_window") == "target" else "block"):
                continue
            E = np.array(blk["energy"])
            g = g.sort_values("index")
            if len(g) != len(E) or np.max(np.abs(g.energy.values - E)) > 1e-9:
                raise ValueError(f"parity.csv does not match block {key} at N={N}")
            if blk.get("source_window") == "target":
                if np.max(np.abs(g.C2.values - np.array(blk["c2"]))) > 1e-5:
                    raise ValueError(f"parity.csv C2 does not match target source window {key} at N={N}")
                blk["x"] = g.X.values.astype(int).tolist()
            self.parity[key] = g.P.values.astype(int)
        nopar = [k for k, b in self.blocks.items() if b.get("source_window") == "target" and k not in self.parity]
        if nopar:
            raise ValueError(f"no window=target parity rows for target source windows {nopar} at N={N}")
        self.gauge_check = self._gauge_check()
        self.input_hashes = {"meta.json": sha256_file(os.path.join(export_dir, "meta.json")),
                             "parity.csv": sha256_file(os.path.join(physics_dir, "parity.csv"))}

    def _gauge_check(self):
        """v0.3.5: every edge whose target window is read in the source window's gauge (block-target statuses, and any
        edge into a target source window) must carry exactly that window's per-column eigenvector SHAs."""
        src = {}
        for (a, b, li, lf), e in self.edges.items():
            if e.get("source_colsha") is not None:
                src.setdefault(f"s{a}_s{b}_L{li}", e["source_colsha"])
        checked = 0
        for (a, b, li, lf), e in self.edges.items():
            key = f"s{a}_s{b}_L{lf}"
            if key not in self.blocks:
                continue
            same_gauge = e["target_status"] in BLOCK_TARGET_STATUS or self.blocks[key].get("source_window") == "target"
            if not same_gauge or e.get("target_colsha") is None or key not in src:
                if same_gauge and self.blocks[key].get("source_window") == "target":
                    raise ValueError(f"target source window {key} lacks column SHAs at N={self.N}")
                continue
            if list(e["target_colsha"]) != list(src[key]):
                raise ValueError(f"edge {e['key']} target columns differ from the {key} source window at N={self.N}")
            checked += 1
        return dict(edges_checked=checked)

    def delta(self, E):
        return self.alpha * (np.asarray(E) - self.E0)

    def edge_arrays(self, a, b, li, lf):
        """Calibrated reduced generator matrix M[t,s] and full image norms per source."""
        k = (a, b, li, lf)
        if k in self._cache:
            return self._cache[k]
        e = self.edges[k]
        nt, ns = e["nt"], e["ns"]
        path = os.path.join(self.export_dir, e["bin"])
        raw = read_f64(path, e.get("bin_sha256"))
        if raw.size != 2 * nt * ns + 4 * ns:
            raise ValueError(f"bad edge payload size {path}")
        O1 = raw[:nt * ns].reshape((nt, ns), order="F")
        O2 = raw[nt * ns:2 * nt * ns].reshape((nt, ns), order="F")
        G = raw[2 * nt * ns:].reshape((2, 2, ns), order="F")
        M = self.coeff[0] * O1 + self.coeff[1] * O2
        full = np.einsum("i,ijs,j->s", self.coeff, G, self.coeff)
        out = dict(M=M, full=full, sE=np.array(e["source_E"]), tE=np.array(e["target_E"]),
                   sC2=np.array(e["source_C2"]), tC2=np.array(e["target_C2"]),
                   frontier=(np.inf if e["target_frontier"] is None else float(e["target_frontier"])), atol=float(e["energy_atol"]),
                   status=e["target_status"], li=li, lf=lf)
        self._cache[k] = out
        return out
