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

`cluster/tcc.sbatch.example` runs one plan: set `TCC_REPO`, `TCC_DATA` and `JULIA`, edit the `#SBATCH` lines (partition,
account, CPUs, memory, time) in your copy. Memory guide: METHODS §2.4 (block dimensions) and §2.5 (residency, Stage B
memory limit = 85 % of the allocation).
