# Fuzzy-sphere SO(5) DQCP template (Tcc v0.3.1)

中文：[docs/zh/README.md](docs/zh/README.md) · Methods: [docs/en/METHODS.md](docs/en/METHODS.md) · Your data and paths:
[docs/en/DATA_LAYOUT.md](docs/en/DATA_LAYOUT.md) · Primary/descendant analyses: [docs/en/ANALYSIS.md](docs/en/ANALYSIS.md) ·
Working with an AI agent: [docs/en/AI_GUIDE.md](docs/en/AI_GUIDE.md)

A template for computing the spectrum, primary candidates (KK projection), OPE coefficients and parities of the SO(5)
deconfined critical point model on the fuzzy sphere, for any number of orbitals N, and for classifying the states into
primaries and descendants afterwards. **There is one code path for all N.** The only N-dependent branch of the physics
is the sign of the C₂ cross-density term for odd and even N (`src/so5_c2_density_phase.jl`). Choices that change only
speed or memory are made from the size of the problem (`src/resources.jl`, METHODS §2.5), never from N.

The repository holds code, documentation and reference plans only. **Your data and paths stay yours:** every output goes
under a data root that you choose, and machine-specific settings live in files that git ignores (`user.toml`,
`analysis/paths.local.json`, `cluster/*.sbatch`).

## Getting started

```bash
cp user.example.toml user.toml                 # set [paths] data_root (default: ./data, ignored by git)
julia deps/setup_env.jl                        # once: builds env/ (Julia 1.11; deps/README.md)
julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml
```

The last command runs the complete chain for N = 5 in a few minutes (< 2 GB) and writes to
`<data root>/results/quickstart/N5/`. The data root is, in this order: `--data=DIR`, the environment variable `TCC_DATA`,
`[paths] data_root` of `user.toml`, the current directory. On a Slurm cluster copy `cluster/tcc.sbatch.example`.
To learn, run and check the code with an AI coding agent (Claude Code, Codex, …), follow
[docs/en/AI_GUIDE.md](docs/en/AI_GUIDE.md); the agent's own instructions are in [AGENTS.md](AGENTS.md).

---

## 1. Code map

| Path | Content |
|---|---|
| `src/model.jl`, `ph.jl`, `seglib.jl`, `exchange.jl` | model, segment spaces (direct / particle–hole half build), segment operator library, segment exchange X |
| `src/resources.jl` | size-based defaults of `ph_half` and `residency` |
| `src/runner.jl`, `matvec.jl` | spectrum of (s_A, s_B, L) blocks, residency modes, H action kernels |
| `src/targets.jl`, `edges.jl`, `kkstage.jl`, `ZhouGenerator.jl`, `kk/KKProjection.jl` | KK target windows, edges, vacuum fit, K² classification |
| `src/tuning.jl` | coupling tuning F(g) = 0 |
| `src/ope.jl`, `ope3.jl` | pipeline OPE; three-point insertions and extended OPE |
| `src/parity.jl` | physical parity |
| `src/stageb.jl` | Stage B — (½,½) and (1,1) sectors to the raw Δ cut, KK, full-Q OPE, raw targets; window certificate, partial checkpoints, preliminary top window |
| `src/fullq.jl` | full-Q OPE of the integer sectors |
| `src/post.jl` | coverage, per-pair targets, ope3 and parity drivers |
| `src/plan.jl` | the plan (`[plan]`) and the step runner |
| `src/pipeline.jl` | pipeline stages, manifest, resumption, provenance |
| `bin/tcc.jl` | the single entry point (data root, steps) |
| `bench/*_run.jl` | command-line forms of single steps |
| `tools/ope3_parity_join.py` | parities of the φ×φ towers |
| `tools/data_inventory.jl` | inventory of your computed sectors and data from the run records |
| `configs/quickstart_N5.toml`, `example_N6.toml` | small runs |
| `configs/examples/N05.toml … N12.toml` | reference plans: what the original study computed for each N |
| `cluster/tcc.sbatch.example` | Slurm job for one plan |
| `test/` | tests (§5) |
| `analysis/` | primary/descendant analyses of the KK output (Python), one directory per rule: `hardgate/` (rule A, all N together) and `switch_kk/` (rule C, one N, with family checks); [docs/en/ANALYSIS.md](docs/en/ANALYSIS.md) |
| `deps/` | environment builder and the one replaced FuzzifiED file |

## 2. One file per calculation: the plan

```toml
[run]
N = 8
g = 0.8903705662882018
out = "results/main/N8"          # relative: under your data root

[plan]
sectors = "all"            # or [[0,0],[0,1],[0,2]]: integer (sA,sB) pairs
lmax = 4                   # blocks L = 0..lmax (KK sources); KK target windows to L = lmax+1
delta_max = 6.2            # target windows complete up to the KK sources with Delta <= 6.2
half = true                # (1/2,1/2) sector in the pipeline OPE
steps = ["coverage", "ope3", "parity", "parity_join", "fullq", "stageb", "rawtarget"]

[spectrum]                 # windows (number of states per block / X sector)
nev = 80
[coverage]                 # grow windows until every block reaches Delta >= 6.0 (or is complete)
delta_cut = 6.0
kmax = 400
[targets]
initial_nev = 80
max_nev = 4000
[ope3]                     # see METHODS section 9
int_ranks = [1, 2, 3, 4]
half_lmax = 3
[fullq]                    # default: every pair of the run; rank_max = 4
[stageb]                   # sectors "half" = (1/2,1/2), "11" = (1,1)
sectors = ["half"]
lmax = 4                   # KK sources and OPE spins 0..4, raw windows 0..5
raw_delta_max = 6.8
```

```bash
julia --project=env --threads=64 bin/tcc.jl N08.toml                    # all plan steps
julia --project=env --threads=64 bin/tcc.jl N08.toml stageb             # one step
julia --project=env --threads=64 bin/tcc.jl N08.toml --data=/scratch/me # another data root
```

* The plan expands into the pipeline keys (`spectrum.blocks` = sectors × L0..lmax in lexicographic order,
  `targets.source_delta_max`, `ope.half`); giving the same key twice with different values is an error.
* **Steps**: `basis tune spectrum targets kk ope summary` (pipeline stages), `coverage`, `targets_pairs`
  (`[targets_pairs] pairs = [[a,b],…]`: target windows of some pairs, for several nodes in parallel), `ope3`, `parity`,
  `parity_join`, `fullq`, `stageb`, `rawtarget`. Output directories default to `<run.out>_<step>`
  (`<run.out>_stageb_<sector>`); `out = …` in a step's table overrides.
* Every step resumes from its checkpoints; every checkpoint is bound to its physical inputs (N, g, basis SHA, windows), so
  a restart never mixes data of different inputs. The pipeline refuses a run directory with other physics keys.
* **Record**: the pipeline stores the plan in `manifest.toml`; every other step writes `plan.toml` (plan, step options,
  code version and commit) into its output directory. `tools/data_inventory.jl` builds your data table from these records.

Variants that are only keys of the plan:

| Calculation | Plan |
|---|---|
| half and (1,1) sectors, spins 0–4 | `[stageb] sectors = ["half", "11"]`, `lmax = 4` |
| half sector, spins 0–3 | `[stageb] lmax = 3` |
| largest N: parallel window jobs | `[stageb] lmax = 1`, `step = "windows"`, `windows = [0, 1]` in one job, `[2]` in another, then `step = "all"` |
| stop a long window job cleanly, resume later | `[stageb] partial_checkpoints = true` (default), `stop_after_k = k` with `step = "windows"` |
| cheap window certificate | `[stageb] extra_states = 4`, `complement_tol = 1e-6`, `final_polish_trigger = 1e-9` (defaults); `0, 1e-10, 0` = the v0.3.0 certificate |
| preliminary KK/OPE from a partial top window | `[stageb] prelim_top_window = "<out>/windows/partial/L2_k081.ckpt"`, `prelim_complement = true` |
| OPE onto the full raw windows | step `rawtarget` (or `[stageb] rawtarget = true`) |
| full-Q OPE of selected sectors | step `fullq`, `[fullq] sectors = [[0,1],[0,2],[0,0]]`, `rank_max = 3` |
| target windows per pair on several nodes | step `targets_pairs` |
| only some sectors / L / pairs in KK | `plan.sectors`, `kk.pairs`, `kk.lmax` |

## 3. Reference plans
`configs/examples/N05.toml … N12.toml` restate what the original study computed for each N (sectors, L, windows, Δ cuts,
post steps, tuned g). Use them as starting points; before N ≥ 8 check the block dimensions (METHODS §2.4) — N10 and up
need a cluster node, N12 hundreds of GB and days.

## 4. Resource switches
See METHODS §2.5: `basis.ph_half` (largest segment sector ≥ 2000 → PH half build), `spectrum.residency` (segment basis
> 16e9 bytes → `pair`), `threads.native_dim_max` (block dimension 2×10⁷), Stage B memory limit (85 % of the Slurm
allocation). None of them changes a result; all can be set explicitly.

## 5. Tests

| Test | Checks |
|---|---|
| `test/regress.jl N out [golden]` | segment operators, spectra with and without X, residency modes, KK/Γ² edges, golden comparison |
| `test/t1_equivalence.jl` … `t10_highscalar.jl` | building blocks: equivalence, exchange solve, runner, hashing, OPE, high scalars |
| `test/t11_extend.jl`, `t12_gates.jl` | window extension; acceptance gates |
| `test/t13_ph.jl N out projectors` | PH half build spans the direct build's multiplet subspaces |
| `test/t14_v028.jl` | half-sector edge identity, upstream digests |
| `test/t15_stageb_restart.jl` | Stage B restarts reuse windows, edges and projections; partly lost outputs are completed |
| `analysis/hardgate/tests/`, `analysis/switch_kk/**/test_*.py` | the analyses on synthetic data |

Give every test an output directory under your data root (for example `<data root>/tests/regress`).
Results are reproducible to rounding, not bitwise across processes (threaded BLAS changes the last bits and hence
eigenvector signs): compare magnitudes. When v0.3.0 was made, the whole chain was run against the earlier per-N code on
N5 and N6; the differences were of the size of two runs of the same code (≤ 2e-11 absolute).

## 6. Credits
See [AUTHORS.md](AUTHORS.md). FuzzifiED is MIT-licensed (`deps/FuzzifiED-2.0.1/LICENSE`). Add a licence of your choice
before you share the repository.
