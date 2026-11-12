# Your data: data root, layout, records, inventory

中文：[../zh/DATA_LAYOUT.md](../zh/DATA_LAYOUT.md). Methods: [METHODS.md](METHODS.md).

The repository holds code and documentation only. Every number you compute is written under **your data root**, and
every machine-specific setting lives in a file that git ignores. Nothing in the tracked files names a machine, a user or
a directory of anyone's computer.

## 1. Choosing the data root

`bin/tcc.jl` changes into the data root before it reads the plan, so every relative path of a run file (`run.out`, the
`out` keys of the steps, `kk.cache`, `basis.segment_checkpoint`, `stageb.prelim_top_window`, …) is under it.
Absolute paths in a run file are used as they are. The data root is the first of:

| Source | Example |
|---|---|
| `--data=DIR` on the command line | `bin/tcc.jl configs/examples/N08.toml --data=/scratch/me/fuzzysphere` |
| environment variable `TCC_DATA` | `export TCC_DATA=/scratch/me/fuzzysphere` |
| `[paths] data_root` in `user.toml` (copy of `user.example.toml`; relative = relative to the repository) | `data_root = "D:/fuzzysphere"` |
| the current directory | — |

The run file itself is read from where you give it. The first line of the output, `TCC_DATA_ROOT …`, shows the root used.

Files that hold your settings and are ignored by git: `user.toml`, `analysis/paths.local.json`, `cluster/*.sbatch`
(copies of `cluster/tcc.sbatch.example`). `.gitignore` also excludes `data/`, `results/`, `runs/`, `logs/`, checkpoints and
the analysis `inputs/`, `results/` and threshold files, so a forgotten relative path does not end up in a commit.

## 2. Layout used by the reference plans

```
<data root>/
  kk_cache/                      cached KK generator terms (shared by runs of the same N)
  results/main/N<n>/             pipeline run: manifest.toml, run.toml / coverage_run.toml, blocks/, targets/, kk/, ope/,
                                 summary.tsv, physics_report.toml, physics_primaries.csv
  results/ope3/N<n>/             three-point insertions (ope3.toml, *.csv, plan.toml)
  results/parity/N<n>/           parity.csv, parity_kk.csv, parity_summary.toml
  results/fullq/N<n>/            xy_blocks_raw.csv, run_manifest.toml
  results/stageb/half/N<n>/      Stage B: windows/, kk/, edges/, input_binding.toml, stageB_manifest.toml, rawtarget/
  results/stageb/11/N<n>/
  results/quickstart/N5/         the quick start
  tests/                         test outputs
  inventory/                     tools/data_inventory.jl output
  analysis/                      exports and results of the analyses (docs/en/ANALYSIS.md)
```

Any other layout works; the steps find each other through the paths of the plan.

## 3. What a run records

- `manifest.toml` (pipeline) and `plan.toml` (every other step): the expanded plan, the step options, the code version,
  the git commit (`TCC_GIT_COMMIT`, else read from git) and a digest of the code tree.
- Checkpoints carry the SHA-256 of their segment basis and are bound to N, g and the windows; Stage B binds all of its
  checkpoints to `input_binding.toml`.
- Records inside the data root contain absolute paths of the run directories, and Stage B job records contain the host
  name and the Slurm job id. They are your records: keep them in the data root, do not copy them into the repository.
  Keep an unfinished run at the same absolute path until it is complete (resumption checks the recorded source run).

## 4. Inventory of your data

`tools/data_inventory.jl` builds a table of what was computed (sectors, L ranges, windows, Δ cuts, KK channel counts,
post-processing rows) from the text records above — no checkpoint is loaded.

```bash
cp configs/inventory_index.example.toml configs/inventory_index.toml   # or keep it in your data root
julia tools/data_inventory.jl configs/inventory_index.toml <data root>/inventory/DATA_INVENTORY <data root>
```

The index lists, per N, the output directories relative to the data root (and a free-text `jobs` field for your own job
ids). It writes `DATA_INVENTORY.md` (one section per N) and `DATA_INVENTORY.csv`.

## 5. Coupling g(N)

Each plan gives g as a number. The reference plans carry the tuned values of the original study (F(g) = 0 with
Δ_J = 2, Δ_T = 3; METHODS §4). To tune g yourself, set `g = "tuned"` and a `[tune]` table (`start`, `bounds`,
`maxpoints`; see `configs/coverage_template.toml`), or run the `tune` stage; the selected value and its record are
written into the run directory. Check `F` in `physics_report.toml` of every run.

## 6. Cluster

### 6.1 Environment

`cluster/environment.example.sh` describes a self-contained installation, the setup the original study ran its jobs in:
one Julia depot holds the packages, binary artifacts, the registry and the compile caches; Julia runs offline, so a
job never reads another depot or the network. Copy it to `cluster/environment.sh` (ignored by git), set `TCC_REPO`,
`TCC_DATA` and `JULIA`, and source it in every job.

```bash
cp cluster/environment.example.sh cluster/environment.sh       # edit the paths
source cluster/environment.sh
JULIA_PKG_OFFLINE=false "$JULIA" deps/setup_env.jl               # once, needs internet
cp cluster/validate_env.sbatch.example cluster/validate_env.sbatch && sbatch cluster/validate_env.sbatch
```

* **Cluster without internet.** Build the installation on a Linux x86_64 machine that has internet (WSL works; binary
  artifacts are per platform), with the same Julia version, then copy the repository with its `depot/` to the cluster
  (the commands are in the header of `environment.example.sh`). `env/Manifest.toml` names `deps/FuzzifiED.jl` by a
  relative path, so the copy works at any location.
* **Validation.** `validate_env.sbatch` checks on the node it runs on that everything loads from the one depot
  (`LOAD_ONLY_DEPOT_PASS`), then runs `test/regress.jl 5` and `test/t9_ope.jl` (`REGRESS_PASS`, `OPE_REFERENCE_PASS`).
  Run it once per CPU type: Julia compiles for the CPU it runs on (`-C native`), so each node type builds its own
  compile cache once (about a minute). Given the regression directory of another environment as argument, it also
  compares spectra, C₂ and edges with it; switching environments should change them only at the 1e-13 level.
* **Python.** Steps `parity_join` and the analyses need Python with numpy (and pandas). If the cluster Python lacks them
  and pip is not available, put a directory with the packages on `TCC_PYTHONPATH` (see `environment.example.sh`).

### 6.2 Jobs

`cluster/tcc.sbatch.example` runs one plan or some of its steps: edit the `#SBATCH` lines (partition, account, CPUs,
memory, time) in your copy. `--set=table.key=value` (TOML syntax) changes a key of the plan for one job without editing
the file; the effective plan is written to `<data root>/plans/` and recorded in the outputs. Memory guide: METHODS §2.4
(block dimensions) and §2.5 (residency, Stage B memory limit = 85 % of the allocation).

### 6.3 The largest N: one plan, several jobs

This is how the N12 results of the original study were produced, written with `configs/examples/N12.toml`. Every job
resumes from checkpoints, so a job that is stopped or fails is simply submitted again.

```bash
P=configs/examples/N12.toml
sbatch cluster/tcc.sbatch $P coverage ope3 parity parity_join fullq     # main chain: spectrum, KK, OPE, parity, full-Q
# Stage B (half sector), raw windows split over jobs; the top window is by far the largest
sbatch cluster/tcc.sbatch $P stageb --set=stageb.step=windows --set=stageb.windows=[0,1]
sbatch cluster/tcc.sbatch $P stageb --set=stageb.step=windows --set=stageb.windows=[2]
sbatch cluster/tcc.sbatch $P stageb                                     # KK and OPE on the finished windows
sbatch cluster/tcc.sbatch $P rawtarget                                  # optional: OPE onto the full raw windows
```

* N12 sizes: the segment basis is about 50 GB, the largest blocks need hundreds of GB; the L2 window of the half sector
  (dimension 1.4×10⁸, 183 states) took about two weeks on a node with 1.5 TB, several days of which were the final
  complement check of the older window certificate (the default certificate of this version replaces that check by a
  few extra states, `[stageb] extra_states`); its checkpoint is about 190 GiB. With
  `partial_checkpoints = true` (default) the window job saves every grown step, and `--set=stageb.stop_after_k=k`
  stops it cleanly once k states are saved.
* Exports for the analyses (`analysis/`, docs/en/ANALYSIS.md §2), run where the data are:
  `analysis/hardgate/cluster/export_kk.jl <run.out> <out>`, `export_stageB.jl <stageb out> <out>` and
  `export_ope_edges.jl <fullq out> <out>`.
