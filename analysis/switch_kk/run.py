"""Blind KK v3 (rule C) for one size N: bounded switch lowering operators with window-exact Gram and commutator."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CODE_FILES = ("run.py", "protocol.md", "test_v3.py")
DELTA_MAX = 6.25
GATE = 0.90
EPS = 1e-8
BIN_WIDTH = 0.25                      # blindpd.thresholds.valley_threshold bin width (log10 kappa)
WIDTHS = {"hard": 0.0, "switch_0.25": 0.25, "switch_0.50": 0.50, "paper_window": None}
CLASSIFY = ("hard", "switch_0.25", "switch_0.50")
METHODS = tuple(WIDTHS)
COMPLETE_MARGIN = max(w for w in WIDTHS.values() if w is not None)
BAND = ("band_lo", "central", "band_hi")
ANCHORS = {"J": ("s0_s1_L1", 1, 2.0), "T": ("s0_s0_L2", 2, 3.0)}
ANCHOR_TOL = 1e-3
CROSS_C2_REL = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------- core maths

def lowering_weight(gap, width, atol):
    """K-share f(omega) of each transition, with the equal-energy (rotation) band removed.

    width None: paper eq. (4.2), f = (1 - omega)/2 (unbounded).  width 0: hard cut.
    width w > 0: f = clip((w - omega)/(2w), 0, 1)."""
    gap = np.asarray(gap, float)
    active = np.abs(gap) > atol
    if width is None:
        f = (1 - gap) / 2
    elif width == 0:
        f = (gap < 0).astype(float)
    else:
        f = np.clip((width - gap) / (2 * width), 0.0, 1.0)
    return f * active, active


def subspaces(delta, gram, cuts):
    """Low-kappa eigen-subspaces of a K^dagger K Gram; kappa = lambda / (2 <Delta>)."""
    values, vectors = np.linalg.eigh(gram)
    if values.min() < -EPS * max(1.0, abs(values).max()):
        raise ValueError("Lowering Gram is not positive")
    kappa = np.maximum(values, 0) / (2 * (vectors**2).T @ delta)
    return {name: vectors[:, kappa <= cut] for name, cut in cuts.items()}


def cut_grid(threshold):
    """Frozen external threshold: central value and the valley bin it was taken from."""
    central, left = float(threshold["threshold"]), float(threshold["rule"]["bin_left"])
    if abs(central / 10 ** (left + BIN_WIDTH / 2) - 1) > 1e-12:
        raise ValueError("Threshold is not the centre of its valley bin")
    return {"strict": 0.01, "band_lo": 10 ** left, "central": central,
            "band_hi": 10 ** (left + BIN_WIDTH), "loose": 0.10}


def support(weights):
    """weights[method][cut] -> robust / low / sensitive plus which axis flips."""
    grid = np.array([[weights[m][c] >= GATE for c in BAND] for m in CLASSIFY])
    label = "robust" if grid.all() else "low" if not grid.any() else "sensitive"
    return dict(support=label,
                cut_flip=bool(any(len(set(row)) > 1 for row in grid)),
                method_flip=bool(any(len(set(col)) > 1 for col in grid.T)))


# ----------------------------------------------------------------------------- inputs

def input_paths(source):
    """KK export of one size N: source/kk/N<N>/{meta.json, edges/*.f64} and source/physics/N<N>/parity.csv.
    The inputs directory must hold exactly one size (one kk/N<N> directory)."""
    sizes = sorted(p for p in (source / "kk").glob("N*") if p.is_dir())
    if len(sizes) != 1:
        raise ValueError(f"Expected exactly one kk/N<N> directory in {source}, found {[p.name for p in sizes]}")
    kk = sizes[0]
    parity = source / "physics" / kk.name / "parity.csv"
    return kk, [kk / "meta.json", parity, *sorted((kk / "edges").glob("*.f64"))]


def freeze(source, threshold, path):
    _, inputs = input_paths(source)
    meta = json.loads(inputs[0].read_text(encoding="utf-8"))
    if not inputs[1].is_file() or len(inputs) - 2 != len(meta["edges"]):
        raise ValueError("Expected parity.csv and one saved .f64 file per KK edge listed in meta.json")
    if path.exists():
        raise FileExistsError(path)
    write_json(path, {
        "code": {name: sha(ROOT / name) for name in CODE_FILES},
        "source_at_freeze": source.name,             # name only: freezes may be committed, absolute paths are private
        "inputs": {p.relative_to(source).as_posix(): sha(p) for p in inputs},
        "threshold_file": {"name": threshold.name, "sha256": sha(threshold)},
        "numpy": np.__version__,
    })


def load(source, threshold_path, lock):
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    kk, paths = input_paths(source)
    if {p.relative_to(source).as_posix(): sha(p) for p in paths} != manifest["inputs"]:
        raise ValueError("Input hashes differ from the freeze")
    if {name: sha(ROOT / name) for name in manifest["code"]} != manifest["code"]:
        raise ValueError("Code, protocol or tests differ from the freeze")
    if sha(threshold_path) != manifest["threshold_file"]["sha256"]:
        raise ValueError("Threshold file differs from the freeze")
    cuts = cut_grid(json.loads(threshold_path.read_text(encoding="utf-8")))
    meta = json.loads(paths[0].read_text(encoding="utf-8"))
    blocks, edges, cal = (meta[key] for key in ("blocks", "edges", "calibration"))
    alpha, e0, coeff = cal["alpha"], cal["E0"], np.array(cal["coefficients"])
    if alpha <= 0 or not np.all(np.isfinite([alpha, e0, *coeff])):
        raise ValueError("Invalid energy or generator calibration")
    parity = {}
    with paths[1].open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["window"] == "block":
                parity.setdefault(f"s{row['sA']}_s{row['sB']}_L{row['L']}", []).append(row)
    indexed, gauges = {}, {}
    for edge in edges:
        key = f"s{edge['a']}_s{edge['b']}_L{edge['li']}"
        nt, ns = edge["nt"], edge["ns"]
        path = (kk / edge["bin"]).resolve()
        if path not in [p.resolve() for p in paths[2:]] or sha(path) != edge["bin_sha256"]:
            raise ValueError(f"Bad or unapproved edge payload {path}")
        gauges.setdefault(key, edge["source_colsha"])
        if (gauges[key] != edge["source_colsha"] or len(gauges[key]) != ns
                or len(edge["target_colsha"]) != nt
                or edge["source_E"] != blocks[key]["energy"]
                or not np.allclose(edge["source_C2"], blocks[key]["c2"], rtol=0, atol=1e-5)):
            raise ValueError(f"Source gauge or eigenvalues disagree for {edge['key']}")
        raw = np.fromfile(path, dtype="<f8")
        if raw.size != 2 * nt * ns + 4 * ns or not np.all(np.isfinite(raw)):
            raise ValueError(f"Invalid raw edge {edge['key']}")
        # layout O1[nt,ns], O2[nt,ns], G[2,2,ns], Float64 LE column-major
        operators = raw[:2 * nt * ns].reshape(2, ns, nt).transpose(0, 2, 1)
        matrix = np.einsum("i,its->ts", coeff, operators)
        gram = raw[2 * nt * ns:].reshape(2, 2, ns, order="F")
        norm = np.einsum("i,ijs,j->s", coeff, gram, coeff)
        if np.any(np.sum(matrix**2, axis=0) > norm + EPS * np.maximum(1, np.abs(norm))):
            raise ValueError(f"Saved image exceeds full image on {edge['key']}")
        delta = alpha * (np.asarray(edge["target_E"]) - e0)
        if np.any(np.diff(delta) < -EPS):
            raise ValueError(f"Unsorted target spectrum on {edge['key']}")
        if edge["target_full"] != (edge["target_frontier"] is None):
            raise ValueError(f"Frontier flag inconsistent on {edge['key']}")
        frontier = np.inf if edge["target_full"] else alpha * (edge["target_frontier"] - e0)
        indexed.setdefault(key, {})[edge["lf"]] = dict(
            edge=edge, target=f"s{edge['a']}_s{edge['b']}_L{edge['lf']}", matrix=matrix, norm=norm,
            delta=delta, frontier=frontier, atol=alpha * edge["energy_atol"])
    for outgoing in indexed.values():
        for item in outgoing.values():
            edge, target = item["edge"], item["target"]
            if edge["target_status"] in ("reused_run_window", "block_top_frontier"):
                if gauges[target] != edge["target_colsha"] or blocks[target]["energy"] != edge["target_E"]:
                    raise ValueError(f"Shared target gauge mismatch on {edge['key']}")
    return blocks, indexed, parity, cuts, manifest, alpha, e0


# ----------------------------------------------------------------------------- analysis

def accumulate(outgoing, ell, c, ids, delta):
    """Grams of every method plus exact norm bookkeeping for one source set."""
    n = len(ids)
    acc = dict(grams={m: np.zeros((n, n)) for m in METHODS},
               p_saved={m: np.zeros(n) for m in CLASSIFY},
               missing=np.zeros(n), equal=np.zeros(n), first=np.zeros(n), first_lb=np.zeros(n),
               cross_c2=0.0)
    for lf, edge in sorted(outgoing.items()):
        same = np.abs(np.asarray(edge["edge"]["target_C2"]) - c) < 1e-5
        columns = edge["matrix"][:, ids]
        # relative to the largest same-Casimir element: eigensolver residuals (~1e-10) are amplified by
        # the ~1e3 generator coefficients, so the absolute scale of this noise is ~1e-7
        scale = max(float(np.max(np.abs(columns[same]), initial=0)), EPS)
        acc["cross_c2"] = max(acc["cross_c2"], float(np.max(np.abs(columns[~same]), initial=0)) / scale)
        m = columns[same]
        gap = edge["delta"][same, None] - delta[ids]
        weight = (2 * lf + 1) / (2 * ell + 1)
        miss = np.maximum(edge["norm"][ids] - np.sum(columns**2, axis=0), 0)
        acc["missing"] += weight * miss
        if np.isfinite(edge["frontier"]):
            acc["first_lb"] += weight * (edge["frontier"] - delta[ids]) * miss
        for method, width in WIDTHS.items():
            f, active = lowering_weight(gap, width, edge["atol"])
            action = f * m
            acc["grams"][method] += weight * (action.T @ action)
            if method in CLASSIFY:
                acc["p_saved"][method] += weight * np.sum(((1 - f) * active * m)**2, axis=0)
        acc["equal"] += weight * np.sum((m * ~active)**2, axis=0)
        acc["first"] += weight * np.sum(gap * active * m**2, axis=0)
    return acc


def fixed_sign(vector):
    return vector if vector[np.argmax(np.abs(vector))] >= 0 else -vector


def analyse(blocks, indexed, parity, cuts, alpha, e0):
    states, modes, sectors = [], [], []
    for key, block in sorted(blocks.items()):
        ell = int(key.rsplit("L", 1)[1])
        outgoing = indexed[key]
        if set(outgoing) != ({1} if ell == 0 else {ell - 1, ell, ell + 1}) or ell > 3:
            raise ValueError(f"Unexpected outgoing edge set for {key}")
        rows = sorted(parity[key], key=lambda row: int(row["index"]))
        energy = np.asarray(block["energy"])
        if (len(rows) != len(energy)
                or not np.allclose([float(r["energy"]) for r in rows], energy, rtol=0, atol=1e-9)
                or not np.allclose([float(r["C2"]) for r in rows], block["c2"], rtol=0, atol=1e-5)):
            raise ValueError(f"Parity rows are not bound to the block spectrum for {key}")
        delta = alpha * (energy - e0)
        c2 = np.rint(block["c2"]).astype(int)
        x = np.asarray(block["x"])
        p = np.array([int(r["P"]) for r in rows])
        frontier = min(item["frontier"] for item in outgoing.values())
        atol = max(item["atol"] for item in outgoing.values())
        if any(0 < d <= DELTA_MAX and abs(cc - raw) > 1e-5 for d, cc, raw in zip(delta, c2, block["c2"])):
            raise ValueError(f"Unresolved SO(5) Casimir for {key}")
        in_scope = (delta >= -EPS) & (delta <= DELTA_MAX)
        eligible = in_scope & (delta > EPS) & (np.abs(p) == 1) & (delta + COMPLETE_MARGIN < frontier - atol)
        result = {}
        for c in sorted(set(c2[eligible])):
            ids = np.flatnonzero(eligible & (c2 == c))
            acc = accumulate(outgoing, ell, c, ids, delta)
            if acc["cross_c2"] > CROSS_C2_REL:
                raise ValueError(f"Generator couples SO(5) Casimir sectors in {key} C2={c}: {acc['cross_c2']}")
            grams = acc["grams"]
            for xx, pp in sorted(set(zip(x[ids], p[ids]))):
                mask = (x[ids] == xx) & (p[ids] == pp)
                selected, positions = ids[mask], np.flatnonzero(mask)
                d = delta[selected]
                leakage = max(float(np.max(np.abs(g[np.ix_(mask, ~mask)]), initial=0))
                              / max(1., float(np.max(np.abs(g)))) for g in grams.values())
                if leakage > 1e-7:
                    raise ValueError(f"Gram couples symmetry sectors in {key} C2={c}: {leakage}")
                sub = {m: g[np.ix_(mask, mask)] for m, g in grams.items()}
                qs = {m: subspaces(d, sub[m], cuts) for m in METHODS}
                fields = dict(block=key, L=ell, so5_c2=int(c), X=int(xx), P=int(pp))
                sector = dict(**fields, dimension=len(selected), leakage=leakage,
                              cross_c2_relative=acc["cross_c2"])
                for cut in cuts:
                    for m in METHODS:
                        sector[f"{m}_rank_{cut}"] = qs[m][cut].shape[1]
                for a, b in (("hard", "switch_0.50"), ("switch_0.25", "switch_0.50"), ("paper_window", "switch_0.50")):
                    cos = np.linalg.svd(qs[a]["central"].T @ qs[b]["central"], compute_uv=False)
                    sector[f"{a}_vs_{b}_central_cos2"] = json.dumps([round(float(v)**2, 12) for v in cos])
                    sector[f"{a}_vs_{b}_central_flag"] = bool(
                        qs[a]["central"].shape[1] != qs[b]["central"].shape[1] or np.any(cos**2 < GATE))
                sectors.append(sector)
                weights = {m: {cut: np.sum(qs[m][cut]**2, axis=1) for cut in cuts} for m in METHODS}
                for j, (idx, pos) in enumerate(zip(selected, positions)):
                    w = {m: {cut: float(weights[m][cut][j]) for cut in cuts} for m in METHODS}
                    row = support(w)
                    paper_central = w["paper_window"]["central"] >= GATE
                    row["paper_window_central_differs_from_switch_0.50"] = bool(
                        paper_central != (w["switch_0.50"]["central"] >= GATE))
                    for m in METHODS:
                        for cut in cuts:
                            row[f"{m}_weight_{cut}"] = w[m][cut]
                    for m in METHODS:
                        row[f"{m}_k2"] = float(sub[m][j, j])
                    for m in CLASSIFY:
                        p_full = acc["p_saved"][m][pos] + acc["missing"][pos]
                        row[f"{m}_comm_minus_delta"] = float((p_full - sub[m][j, j]) / 6 - d[j])
                    row["paper_window_comm_minus_delta_saved"] = float(acc["first"][pos] / 6 - d[j])
                    row["paper_window_comm_minus_delta_lower_bound"] = float(
                        (acc["first"][pos] + acc["first_lb"][pos]) / 6 - d[j])
                    row["lambda_missing_norm2"] = float(acc["missing"][pos])
                    row["equal_energy_lambda_norm2"] = float(acc["equal"][pos])
                    result[idx] = row
                for m in CLASSIFY:
                    q = qs[m]["central"]
                    energy_q, rotation = np.linalg.eigh((q.T * d) @ q)
                    for rank, (mean, vector) in enumerate(zip(energy_q, (q @ rotation).T), 1):
                        vector = fixed_sign(vector)
                        dominant = int(np.argmax(vector**2))
                        mode = dict(method=m, **fields, mode=rank, delta=float(mean),
                                    sigma_delta=float(np.linalg.norm((d - mean) * vector)),
                                    dominant_index=int(rows[selected[dominant]]["index"]),
                                    dominant_weight=float(vector[dominant]**2))
                        for other in METHODS:
                            mode[f"{other}_k2"] = float(vector @ sub[other] @ vector)
                            for cut in cuts:
                                mode[f"{other}_weight_{cut}"] = float(np.linalg.norm(qs[other][cut].T @ vector)**2)
                        mode["band_min_weight_classify"] = min(mode[f"{o}_weight_{c}"] for o in CLASSIFY for c in BAND)
                        mode["vector_json"] = json.dumps({rows[i]["index"]: round(float(v), 12)
                                                         for i, v in zip(selected, vector)})
                        modes.append(mode)
        for idx in np.flatnonzero(in_scope):
            status = "eligible"
            if delta[idx] <= EPS:
                status = "vacuum"
            elif abs(p[idx]) != 1:
                status = "unresolved_parity"
            elif not eligible[idx]:
                status = "frontier_excluded"
            row = dict(block=key, L=ell, raw_index=int(rows[idx]["index"]), so5_c2=int(c2[idx]),
                       X=int(x[idx]), P=int(p[idx]), delta=float(delta[idx]), frontier=float(frontier),
                       status=status)
            states.append((row, result.get(idx, {})))
    columns = list(next(metrics for _, metrics in states if metrics))
    return [dict(**row, **{name: metrics.get(name, "") for name in columns}) for row, metrics in states], modes, sectors


def check_anchors(states):
    out, valid = {}, True
    for name, (block, index, target) in ANCHORS.items():
        row = next((s for s in states if s["block"] == block and s["raw_index"] == index), None)
        ok = (row is not None and row["status"] == "eligible" and abs(row["delta"] - target) < ANCHOR_TOL
              and row["support"] == "robust")
        valid &= ok
        out[name] = dict(block=block, raw_index=index, expected_delta=target, passed=bool(ok),
                         **({} if row is None else {k: row[k] for k in (
                             "delta", "status", "support", "cut_flip", "method_flip",
                             *(f"{m}_weight_central" for m in METHODS))}))
    return dict(method_valid=bool(valid), anchors=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("source", type=Path, help="Blind export inputs directory of one size (kk/N<N>, physics/N<N>)")
    parser.add_argument("threshold", type=Path, help="Frozen external threshold_from_raw.json")
    parser.add_argument("--lock", type=Path, default=ROOT / "freeze.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    source, threshold = args.source.resolve(), args.threshold.resolve()
    if args.action == "freeze":
        freeze(source, threshold, args.lock)
        return
    blocks, indexed, parity, cuts, manifest, alpha, e0 = load(source, threshold, args.lock)
    states, modes, sectors = analyse(blocks, indexed, parity, cuts, alpha, e0)
    anchors = check_anchors(states)
    args.output.mkdir(exist_ok=False)
    for filename, rows in (("states.csv", states), ("modes.csv", modes), ("sectors.csv", sectors)):
        write_csv(args.output / filename, rows)
    write_json(args.output / "anchors.json", {**anchors, "cuts": cuts})
    write_json(args.output / "receipt.json", {
        "freeze_sha256": sha(args.lock),
        "numpy": np.__version__, "numpy_matches_freeze": np.__version__ == manifest["numpy"],
        "cuts": cuts, "method_valid": anchors["method_valid"],
        "outputs": {path.name: sha(path) for path in sorted(args.output.glob("*")) if path.name != "receipt.json"},
        "states": len(states), "modes": len(modes), "sectors": len(sectors),
    })


if __name__ == "__main__":
    main()
