"""Small-N analysis demo: KK export -> kappa threshold -> rule C (bounded switch) with the family check.

    python analysis/tools/demo_small_n.py DATA_ROOT [--size 5] [--pool 6] [--julia julia]

Needs the Tcc outputs of the plan files configs/examples/N05.toml (or configs/quickstart_N5.toml) and N06.toml under
DATA_ROOT, at least the steps coverage and parity (docs/en/REPRODUCE.md, step 4).  Everything is written to
DATA_ROOT/analysis/demo_N<size>/:

    inputs/kk/N<n>/, inputs/physics/N<n>/parity.csv     exports (cluster/export_kk.jl) of the analysed size
    pool/kk/N<m>/, pool/physics/N<m>/parity.csv         exports of the pool sizes (the threshold never sees N<size>)
    threshold_from_raw.json                             kappa_c from the pool (hardgate/tools/kappa_threshold.py)
    switch_kk/                                          a copy of analysis/switch_kk with its freeze.json and results/

The coherent family check is not run: it needs a sign reference for every guarded Delta L = 0 row, which small runs
usually do not provide (its protocol then refuses to run).  A pool of one or two small sizes gives a demonstration
threshold, not the pre-registered value of a production analysis.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent.parent


def sh(cmd, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def find_run(data, n):
    """The pipeline run and the parity output of size n under the data root (reference plans, else quick start)."""
    for run, parity in ((f"results/main/N{n}", f"results/parity/N{n}"),
                        (f"results/quickstart/N{n}/run", f"results/quickstart/N{n}/parity")):
        if (data / run / "kk" / "calibration.ckpt").exists() and (data / parity / "parity.csv").exists():
            return data / run, data / parity / "parity.csv"
    sys.exit(f"no finished run with kk/ and parity.csv for N{n} under {data} (run the coverage and parity steps first)")


def export(julia, data, n, dest):
    run, parity = find_run(data, n)
    kk = dest / "kk" / f"N{n}"
    if not (kk / "meta.json").exists():
        sh([julia, "--startup-file=no", ANALYSIS / "hardgate" / "cluster" / "export_kk.jl", run, kk])
    (dest / "physics" / f"N{n}").mkdir(parents=True, exist_ok=True)
    shutil.copy2(parity, dest / "physics" / f"N{n}" / "parity.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root", type=Path)
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--pool", default="6", help="comma list of sizes for the kappa threshold (must not contain --size)")
    p.add_argument("--julia", default="julia")
    a = p.parse_args()
    data = a.data_root.resolve()
    pool = [int(s) for s in a.pool.split(",")]
    if a.size in pool:
        sys.exit("the pool must not contain the analysed size")
    out = data / "analysis" / f"demo_N{a.size}"
    if (out / "switch_kk").exists():
        sys.exit(f"{out / 'switch_kk'} exists: remove {out} to start again")
    out.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-X", "utf8", "-B"]

    export(a.julia, data, a.size, out / "inputs")
    for n in pool:
        export(a.julia, data, n, out / "pool")
    thr = out / "threshold_from_raw.json"
    sh(py + ["tools/kappa_threshold.py", out / "pool", thr, a.pool], cwd=ANALYSIS / "hardgate")

    code = out / "switch_kk"
    shutil.copytree(ANALYSIS / "switch_kk", code, ignore=shutil.ignore_patterns("results", "freeze.json", "__pycache__"))
    inp = out / "inputs"
    sh(py + ["test_v3.py"], cwd=code)
    sh(py + ["run.py", "freeze", inp, thr], cwd=code)
    sh(py + ["run.py", "run", inp, thr], cwd=code)
    sh(py + ["spectrum.py"], cwd=code)
    sh(py + ["modes_readout.py"], cwd=code)
    fam = code / "family"
    sh(py + ["test_family.py"], cwd=fam)
    sh(py + ["family.py", "freeze", inp, thr], cwd=fam)
    sh(py + ["family.py", "run", inp, thr], cwd=fam)
    sh(py + ["readout.py", inp, thr], cwd=fam)

    anchors = json.loads((code / "results" / "anchors.json").read_text(encoding="utf-8"))
    rec = json.loads((code / "results" / "receipt.json").read_text(encoding="utf-8"))
    print("\nDEMO_DONE")
    print(f"kappa_c = {json.loads(thr.read_text(encoding='utf-8'))['threshold']:.6g} (pool N{a.pool})")
    print(f"method_valid = {anchors['method_valid']}; anchors: " + ", ".join(
        f"{k} Delta={v.get('delta', float('nan')):.6f} {v.get('support')} passed={v['passed']}"
        for k, v in anchors["anchors"].items()))
    print(f"states {rec['states']}, central modes {rec['modes']}, sectors {rec['sectors']}; results in {code / 'results'}")


if __name__ == "__main__":
    main()
