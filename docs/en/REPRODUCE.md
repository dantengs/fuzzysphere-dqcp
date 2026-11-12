# Reproducing the small-N calculations

中文：[../zh/REPRODUCE.md](../zh/REPRODUCE.md) · Introduction: [../../README.md](../../README.md) · Reference guide:
[GUIDE.md](GUIDE.md) · Methods: [METHODS.md](METHODS.md)

This page takes you from an empty machine to the complete N = 5 calculation, the tests, and a primary/descendant
analysis of N = 5, with the numbers you should get. Every command and every time below was run, in this order, from a
fresh clone with an empty Julia package depot (Windows 11, Julia 1.11.7, Python 3.14, 8 threads).

**What "reproduce" means here.** Energies, Δ, C₂, parities, |matrix elements| and all counts agree to rounding
(relative differences ≲ 1e-10). Eigenvector **signs**, and hence the signs of matrix elements and OPE coefficients, are
arbitrary and may differ between runs; so may the row order of tables written by parallel solves, and timing and memory
columns. Gates in the code stop a run whose residuals or windows are not good enough; nothing is relaxed.

| Step | Command (from the repository root) | Time |
|---|---|---|
| 0 | `cp user.example.toml user.toml`, set `data_root` | — |
| 1 | `julia deps/setup_env.jl` | 1.5 min |
| 2 | `julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml`, then `python tools/check_quickstart.py <data root>/results/quickstart/N5` | 2.5 min |
| 3 | tests (§3) | 5 min |
| 4 | `julia --project=env --threads=8 bin/tcc.jl configs/examples/N06.toml coverage parity` | 2 min |
| 5 | `python analysis/tools/demo_small_n.py <data root> --size 5 --pool 6` | 15 s |

Disk: about 110 MB for all outputs, plus about 150 MB of Julia packages. Memory: below 2 GB for these sizes.

---

## 0. Get the code and choose where the data go

```bash
git clone https://github.com/dantengs/fuzzysphere-dqcp.git
cd fuzzysphere-dqcp
cp user.example.toml user.toml
```

Edit `[paths] data_root` in `user.toml` (default `data`, i.e. `./data`, which git ignores). Every output below is written
under it; `bin/tcc.jl` prints `TCC_DATA_ROOT <dir>` as its first line. Instead of `user.toml` you can pass
`--data=DIR` or set `TCC_DATA` ([DATA_LAYOUT.md](DATA_LAYOUT.md)). On Windows use `Copy-Item` for `cp`; the commands are
otherwise the same in PowerShell.

Install Julia 1.11 (for example with [juliaup](https://github.com/JuliaLang/juliaup): `juliaup add 1.11` and then
`julia +1.11 …`, or make it the default) and Python ≥ 3.10:

```bash
python -m pip install -r analysis/requirements.txt     # numpy, pandas
```

## 1. Build the Julia environment

```bash
julia deps/setup_env.jl
```

It installs FuzzifiED 2.0.1 and KrylovKit 0.10.4 from the General registry into `env/`, replaces the one FuzzifiED file
that Tcc needs (`deps/README.md` says what and why), and compares every source file of both packages with reference
SHA-256 hashes. It must end with

```
FuzzifiED.jl: 48/48 source files identical to production
KrylovKit.jl: 36/39 source files identical to production; expected difference (production-only LockedLanczos, unused by Tcc): src/KrylovKit.jl, src/algorithms.jl, src/eigsolve/lockedlanczos.jl
TCC_ENV_READY  from <repository>:  julia --project=env --threads=8 bin/tcc.jl <run.toml>
```

Any other difference stops the script. The remaining dependencies resolve to the newest compatible releases (in the run
documented here: FuzzifiED_jll 1.3.0+0, SpecialFunctions 2.9.1, WignerSymbols 2.0.0); none of them enters the numbers
beyond rounding. Internet access is needed only for this step.

## 2. The complete N = 5 chain

```bash
julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml
python tools/check_quickstart.py <data root>/results/quickstart/N5
```

The plan (`configs/quickstart_N5.toml`) runs, in order: coverage (spectrum of all 30 blocks (s_A, s_B, L ≤ 4) with the
tuned coupling g = 0.8642902899515241, windows grown to Δ ≥ 6.0; KK target windows; KK projection; pipeline OPE),
`ope3` (three-point insertions), `parity`, `parity_join`, `fullq` (full-Q OPE of the integer sectors), `stageb` for the
(½,½) and (1,1) sectors, and `rawtarget`. The last line of the run is
`TCC_DONE version=0.3.1 stages=basis,kk,ope,spectrum,summary,targets`. Outputs:

| Directory under `<data root>/results/quickstart/N5/` | Content |
|---|---|
| `run/` | `summary.tsv` (every state: block, X, energy, Δ, C₂, residuals), `physics_report.toml` (g, F, α, E₀, KK fit, candidates), `kk/projected_states.csv`, `physics_primaries.csv`, `ope/ope.toml` |
| `ope3/` | three-point coefficients, φ×φ towers, `ope3_phiphi_spin_parity.csv` |
| `parity/` | `parity.csv`, `parity_kk.csv`, `parity_summary.toml` |
| `fullq/` | `xy_blocks_raw.csv` |
| `stageb_half/`, `stageb_11/` | raw spectra, KK modes, OPE components, `rawtarget/` |

`check_quickstart.py` prints one line per check and `ALL PASS`; the run documented here gave

```
PASS  F(g) = Delta_T - 3 (tuned g)                   -3.521e-07                   expected |F| < 1e-5; ref -3.5213e-07
PASS  alpha = 2/(E_J - E_0)                          0.485023928556               expected ref 0.485023928556
PASS  Delta of J (lowest C2=3, (0,1) L1)             2.0000000000                 expected 2 (definition)
PASS  Delta of T (lowest C2=0, (0,0) L2)             2.9999996                    expected 3 within 1e-5; ref 2.9999996
PASS  max residual |Hv - Ev|                         7.0e-14                      expected <= 1e-6 (gate), in practice ~1e-10
PASS  max residual |Cv - c2 v|                       7.6e-11                      expected <= 1e-6 (gate)
PASS  one SO(5) multiplet in several (sA,sB) blocks  316 pairs, max |dE| 3.6e-14  expected |dE| <= 1e-10
PASS  phi from half-integer vs integer sector        2.7e-15                      expected <= 1e-10
PASS  phi x phi has no P = -1 content                99 rows, max 8.5e-15         expected <= 1e-12
PASS  current in the half sector (Delta=2, P=+1)     pass                         expected pass
PASS  parity: max residual, mixed KK modes           5.1e-11, 0                   expected <= 1e-6, 0
PASS  KK modes with parity                           639                          expected ref 639
PASS  primary candidates at t = 0.25                 171                          expected ref 171
ALL PASS
```

What the checks mean: F(g) = Δ_T − 3 must vanish at the tuned coupling (METHODS §4); Δ_J = 2 fixes the energy scale;
residuals are the eigen-equation and Casimir errors that the gates bound; the same SO(5) multiplet appears in several
(s_A, s_B) blocks with the same energy; the φ of the half-integer and integer sectors coincide; φ×φ contains only P = +1;
the conserved current has Δ = 2 and P = +1 in the half sector; every state is a parity eigenstate. The reference values
(F, α, Δ_T, 639 KK modes with parity, 171 primary candidates at t = 0.25) are compared with tolerances far above the
rounding level.

## 3. Tests

```bash
julia --project=env --threads=8 test/regress.jl 5 <data root>/tests/regress
julia --project=env --threads=8 test/t15_stageb_restart.jl 5 <data root>/tests/t15
cd analysis/hardgate && for t in test_units test_harden test_fixes_v041 test_descspan; do python -X utf8 -B tests/$t.py; done; cd ../..
```

Each prints a final verdict; the run documented here gave

```
REGRESS_PASS N=5 v0.3.1                          (test/regress.jl, 101 s)
STAGEB_RESTART_PASS N=5                          (test/t15_stageb_restart.jl, 182 s; after three PASS lines)
UNIT TESTS PASS / HARDEN UNIT TESTS PASS / V0.4.1 FIX TESTS PASS / DESCSPAN UNIT TESTS PASS   (3 s)
```

The analysis tests of rule C (`test_v3.py`, `test_family.py`) also run inside step 5. The other tests in `test/`
(GUIDE §5) need inputs or options described in their headers.

## 4. N = 6, main chain and parity

```bash
julia --project=env --threads=8 bin/tcc.jl configs/examples/N06.toml coverage parity
```

This runs only the steps `coverage` and `parity` of the reference plan `configs/examples/N06.toml` (giving steps on the
command line replaces the plan's step list) and writes `<data root>/results/main/N6` and `<data root>/results/parity/N6`.
Its only use here is as the pool of the κ threshold in step 5: the threshold for an analysed size must come from other
sizes. In the run documented here the N6 pool alone gives κ_c = 0.0421697: the same valley bin, hence the same value (the
threshold is a bin centre), as the pre-registered N6–10 pool of the original study (0.04217). To run the whole N = 6 plan, omit the step names.

## 5. Primary/descendant analysis of N = 5 (rule C)

```bash
python analysis/tools/demo_small_n.py <data root> --size 5 --pool 6     # add --julia <path> if julia is not on PATH
```

The script ([ANALYSIS.md](ANALYSIS.md), `analysis/tools/demo_small_n.py`) exports the KK edges of N5 and N6
(`analysis/hardgate/cluster/export_kk.jl`), copies their `parity.csv`, computes κ_c from the N6 pool
(`hardgate/tools/kappa_threshold.py`), copies the rule into `<data root>/analysis/demo_N5/switch_kk/`, and there runs the
unit tests, `run.py freeze`, `run.py run`, `spectrum.py`, `modes_readout.py` and the incoherent family check
(`family/`: freeze, run, readout). It ends with

```
DEMO_DONE
kappa_c = 0.0421697 (pool N6)
method_valid = True; anchors: J Delta=2.000000 robust passed=True, T Delta=3.000000 robust passed=True
states 285, central modes 194, sectors 78; results in <data root>/analysis/demo_N5/switch_kk/results
```

Of the 265 eligible states of the (0,x) blocks, 37 are robust primary candidates, 18 sensitive and 210 low (`states.csv`); the J and T anchors are robust. The incoherent family check labels, with the hard split, 51 states primary, 71 descendant, 135 mixed and 8 unresolved (`family/results/state_families.csv`). These are the numbers of this demonstration threshold, not results of a production analysis.

Where to look: `switch_kk/results/states.csv` (per state: status, the three methods' weights, `support` =
robust / sensitive / low), `levels.csv` (SO(5) levels with irreps), `mode_levels.csv` (central low-K†K directions),
`family/results/state_families.csv` and `summary.json` (family, level and label of every state; slot counts).

Limits of this demonstration: a one-size pool gives a demonstration threshold, not the pre-registered value of a
production analysis (which pooled N = 6–10). The coherent family check (`family/coherent/`) is not run: it needs a sign
reference for every guarded ΔL = 0 row, which the N5 export does not provide, and its protocol then refuses to run.
Rule A (`analysis/hardgate`, all sizes together) also needs the combined per-N OPE rows, which this template has no tool
to collect (ANALYSIS §4).

## 6. Larger N

The same commands work for every reference plan `configs/examples/N07.toml … N12.toml`. Before N ≥ 8 look at the block
dimensions (METHODS §2.4: the largest block is 4 280 at N7, 3.2×10⁴ at N8, 2.5×10⁵ at N9, 2.0×10⁶ at N10); N ≥ 10 needs
a cluster node (`cluster/tcc.sbatch.example`), and N12 hundreds of GB and days.

## 7. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `setup_env.jl` reports `NOT AS IN PRODUCTION` | a package source differs from the reference; delete `env/Manifest.toml` and `deps/FuzzifiED.jl/` and run it again with Julia 1.11 |
| `parity_join` cannot start Python | set `TCC_PYTHON` to your Python, or `[parity_join] python = "…"` in the plan |
| a run stops with `GATE_FAIL` | a residual, window or independent check failed: report it, do not relax the gate (METHODS §3.4) |
| a run was interrupted | run the same command again: every step resumes from its checkpoints |
| you want to start over | delete the output directory under the data root (never edit files inside a run directory) |
| `demo_small_n.py` says the demo directory exists | delete `<data root>/analysis/demo_N5` |
