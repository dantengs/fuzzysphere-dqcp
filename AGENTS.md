# Instructions for AI coding agents working on this repository

Tcc computes the spectrum, KK-projected primary candidates, OPE coefficients and parities of the SO(5) DQCP model on
the fuzzy sphere (Julia 1.11, FuzzifiED); `analysis/` classifies primaries and descendants (Python). This repository is
a template: it holds code and documentation, the user's data live under the user's own data root.
Humans: see `docs/en/AI_GUIDE.md` (中文 `docs/zh/AI_GUIDE.md`).

## Read first
1. `README.md` (introduction), `docs/en/GUIDE.md` — code map, the plan file, steps, tests; `docs/en/REPRODUCE.md` — the small-N reproduction and its reference numbers.
2. `docs/en/METHODS.md` — every formula, threshold, seed and parameter, with the file that sets it.
3. `docs/en/DATA_LAYOUT.md` — the data root, where outputs go, the inventory.
4. `deps/README.md` — the environment (FuzzifiED 2.0.1 with one replaced file).
Chinese versions are in `docs/zh/`; answer in the language the user writes in.

## Environment and commands
```bash
cp user.example.toml user.toml                                          # once; set [paths] data_root
julia deps/setup_env.jl                                                 # once
julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml   # full chain, N5, a few minutes
python tools/check_quickstart.py <data root>/results/quickstart/N5      # PASS/FAIL against the reference values
julia --project=env --threads=8 test/regress.jl 5 <data root>/tests/regress          # core regression (N = 4, 5, 6)
julia --project=env --threads=8 test/t15_stageb_restart.jl 5 <data root>/tests/t15   # Stage B restart
```
The whole small-N reproduction, including the N6 pool and the rule C analysis of N5
(`analysis/tools/demo_small_n.py`), is `docs/en/REPRODUCE.md`.
On a cluster: `cluster/environment.example.sh` (one depot, offline), `cluster/validate_env.sbatch.example` (run once per
node type), `cluster/tcc.sbatch.example`; `bin/tcc.jl plan.toml step --set=table.key=value` overrides a plan key for one
call (the effective plan goes to `<data root>/plans/`). Job chain of the largest N: `docs/en/DATA_LAYOUT.md` §6.3.
Start with N = 4 or 5 (the N5 full chain takes about 3 minutes with 8 threads on a desktop). Each further N multiplies the
largest block dimension by about 8 (METHODS §2.4: N7 4 280, N9 2.5×10⁵, N11 1.8×10⁷); N ≥ 10 needs a cluster node and
N12 needs hundreds of GB and days. Before starting N ≥ 8, tell the user the block dimensions and ask.

## The user's data and privacy
- Write outputs only under the data root (`--data=`, `TCC_DATA`, `user.toml`), never into the repository.
- Machine paths, host names, account names and cluster settings belong in `user.toml`, `analysis/paths.local.json`
  or `cluster/*.sbatch`, which git ignores. Do not put them into tracked files, commit messages or docs.
- Do not commit data, checkpoints, result tables or logs.

## Rules that protect the physics
- **Never relax a gate** (`src/gates.jl`, `GATE`) or a tolerance to make a run pass. A `GATE_FAIL` is a finding: report it.
- The physics is the same for every N; the only N branch is the C₂ cross-density sign (`src/so5_c2_density_phase.jl`).
  Do not add `if N == …` branches; resource choices go through `src/resources.jl` and must not change results.
- Eigenvectors belong to the segment basis that produced them (SHA recorded in every checkpoint). Never mix checkpoints
  of different bases, g or windows; never rebuild a basis for an existing run.
- Checkpoints are written atomically and reused by identity. Do not delete or edit files inside a run directory to
  "fix" a run; use a new output directory.
- Results are reproducible to rounding, not bitwise across processes: energies, |matrix elements|, C₂ and parities
  agree to ~1e-13 relative; eigenvector **signs** (hence signs of matrix elements) may flip. Compare magnitudes.
- These are finite-N numbers: Δ is scaled by Δ_J = 2, g(N) matched by Δ_T = 3; no extrapolation, and KK low-K²
  states are *candidates*, never certified primaries.
- When you claim something about the code or the data, cite the file and line or the output file and the number.
  If you did not run it, say so.

## Primary/descendant analyses (`analysis/`, Python; `docs/en/ANALYSIS.md`)
- Two rules, one directory each: `hardgate/` (rule A, blindpd, all sizes together) and `switch_kk/` (rule C, one size,
  with `family/` checks).
- **Blind boundary.** When the user runs a blind analysis, never read published P/D labels or values (for example
  arXiv:2306.16435) while developing, running or reviewing it, and do not open `analysis/hardgate/comparison/unblind.py`,
  which contains them. Unblinding is a separate step on a frozen result, and only when the user asks.
- Freeze before running on data: `run.py freeze` / `family.py freeze` / `coherent.py freeze` record the SHA-256 of
  code, protocol, tests and inputs. After a freeze the code, `protocol.md` and thresholds are not edited; a change of
  rule is a new versioned directory with its own freeze and tests.

## Where things are
`src/plan.jl` plan + step runner · `src/pipeline.jl` stages, manifest · `src/runner.jl` spectrum · `src/targets.jl` KK
target windows · `src/kkstage.jl` KK projection · `src/ope.jl`, `src/ope3.jl` OPE · `src/parity.jl` parity ·
`src/stageb.jl` (½,½)/(1,1) sectors · `src/fullq.jl` full-Q OPE · `src/resources.jl` size-based switches ·
`bin/tcc.jl` entry and data root · `configs/examples/` reference plans · `test/` tests · `analysis/` P/D analyses.
