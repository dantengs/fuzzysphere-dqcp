# Using and checking Tcc with an AI assistant — quick start

中文：[../zh/AI_GUIDE.md](../zh/AI_GUIDE.md). For people who want to run this code, understand it or check it with the help
of an AI coding agent (Claude Code, Codex, or similar). About 30 minutes from nothing to a full N = 5 calculation that you
have checked yourself.

---

## 1. What you need
- Julia **1.11** (validated with 1.11.7), git, Python 3 (only for one small join step), 8 CPU threads and 4 GB of memory for N ≤ 6.
- An AI coding agent that can read files and run commands in a terminal. Both Claude Code and Codex read the instruction
  file in the repository root automatically (`CLAUDE.md` / `AGENTS.md`); that file tells the agent what the code is, how to
  run it, and which rules protect the physics (never relax a gate, never mix checkpoints, finite-N numbers only).

## 2. Set up (once)
```bash
git clone <your copy of this repository> fuzzysphere && cd fuzzysphere
cp user.example.toml user.toml        # set [paths] data_root: where your outputs go
julia deps/setup_env.jl
```
`setup_env.jl` installs FuzzifiED 2.0.1 and KrylovKit 0.10.4, replaces the one FuzzifiED file that Tcc needs
(`deps/README.md` says what and why) and checks every package source file against the reference hashes.
It ends with `TCC_ENV_READY`. Then start your agent in this directory. Outputs of `bin/tcc.jl` go under your data
root ([DATA_LAYOUT.md](DATA_LAYOUT.md)), never into the repository.

## 3. First conversation: let the AI learn the project
Copy this as the first message:

> Read AGENTS.md, README.md, docs/en/GUIDE.md, docs/en/METHODS.md and docs/en/DATA_LAYOUT.md. Do not run anything yet. Explain in one page:
> what physical system this code studies, what the steps of `bin/tcc.jl` compute and in which order, which outputs hold
> the final numbers, and which quantity is tuned to fix the coupling g. Cite the file for every statement.

Check the answer against GUIDE §2 and METHODS §4–§7 yourself. If the agent states something without a file, ask for it.

## 4. Run a complete calculation (N = 5, a few minutes)
> Run `julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml`. When it finishes, list the output files and, for each, say in one line what it contains.

Main outputs under `<data root>/results/quickstart/N5/`:

| File | Content |
|---|---|
| `run/summary.tsv` | every computed state: block (s_A, s_B, L), exchange sector, energy, Δ, C₂, residuals |
| `run/physics_report.toml` | g, F(g), E₀, α, KK fit, primary-candidate count, pipeline OPE |
| `run/kk/projected_states.csv`, `run/physics_primaries.csv` | KK-projected primary candidates with K² |
| `ope3/*.csv`, `ope3/ope3.toml` | three-point coefficients and φ×φ towers (with parity in `ope3_phiphi_spin_parity.csv`) |
| `parity/parity.csv`, `parity_kk.csv` | physical parity of every state and KK mode |
| `fullq/xy_blocks_raw.csv` | full-Q OPE components of the integer sectors |
| `stageb_half/`, `stageb_11/` | the (½,½) and (1,1) sectors: raw spectrum, KK modes, OPE components, raw-target edges |

## 5. Check the results — ask for these, with numbers
Each check has an expected outcome you can verify in the file the agent names. `python tools/check_quickstart.py
<data root>/results/quickstart/N5` runs all of them at once ([REPRODUCE.md](REPRODUCE.md) §2); ask the agent to show where
each number comes from rather than only running the script.

| # | Ask the agent to show | Expected | N5 quick start gave (reference) |
|---|---|---|---|
| 1 | `F` in `run/physics_report.toml` | \|F\| < 1e-5 (the tuned g gives Δ_T = 3) | −3.5e-7 |
| 2 | Δ of the lowest C₂ = 3 state of (0,1) L1 and of the lowest C₂ = 0 state of (0,0) L2 in `summary.tsv` | exactly 2 (by definition) and ≈ 3 | 2.0 and 2.9999996 |
| 3 | the largest `H_res` and `C2_res` in `summary.tsv` | ≤ 1e-6 (the code stops otherwise), in practice ~1e-10 or smaller | 6.6e-14, 1.8e-10 |
| 4 | the same SO(5) multiplet in different (s_A, s_B) blocks: equal (L, C₂, energy) | energies agree to ~1e-12 or better | 316 pairs (counted as in `check_quickstart.py`), max \|ΔE\| 3.6e-14 |
| 5 | `half_vs_integer_phi_energy_difference` in `run/ope/ope.toml` | ~1e-14: the half-integer and integer sectors give the same φ | 8.0e-15 |
| 6 | column `ratio` of all P = −1 rows in `ope3/ope3_phiphi_spin_parity.csv` | ~0 (≤ 1e-12): φ×φ contains only P = +1 | 99 rows, max 9.2e-15 |
| 7 | `status` in `stageb_half/raw_current_control.toml` | `pass`: the current has Δ = 2, P = +1 in the half sector | pass |
| 8 | `max_residual` and `kk_modes_mixed_t0.25` in `parity/parity_summary.toml` | ≤ 1e-6, and 0 mixed modes | 6.4e-11, 0 |
| 9 | the primary-candidate count at t = 0.25 in `run/physics_report.toml` | stable between reruns | 171 candidates |

Signs of matrix elements and OPE coefficients may differ between runs (eigenvector phases); everything else should
agree to rounding.

Then the tests:
> Run `julia --project=env --threads=8 test/regress.jl 5 <data root>/tests/regress` and `test/t15_stageb_restart.jl 5
> <data root>/tests/t15`. Report each PASS/FAIL line.

## 6. Check the code — useful review requests
> Compare `src/model.jl: build_terms` with METHODS §1.2–1.3 term by term: interaction, pairing, C₂, and the only
> N-dependent branch. Report any difference with file and line.

> Find every place where a tolerance or gate is applied (`src/gates.jl`, `window_gate`, `GateError`). For each, say what
> it protects and whether any code path can skip it.

> Take `src/stageb.jl` and explain how a restart decides to reuse a window, an edge or a projection. Could a checkpoint
> computed with another g or basis be reused? Show the identity fields.

> Independently of Tcc, write a short exact-diagonalisation check for N = 4 with FuzzifiED (`GetDenIntTerms`,
> `GetPairIntTerms`, full Fock space with the same quantum numbers) and compare the lowest energies of (0,0) L0 with
> `summary.tsv`. Do not modify `src/`.

Independent checks of this last kind (a second program, not the same code) are the strongest evidence; they were done
for N ≤ 6 when the code was written.

## 7. Plan your own calculation
> I want sectors (0,0) and (0,1) up to L = 3 at N = 7 with Δ coverage 6.0. Write the plan file from
> `configs/examples/N07.toml`, tell me the largest block dimension (METHODS §2.4) and the expected memory, and do not
> start it before I confirm.

Plans are described in GUIDE §2; every step records its plan in `plan.toml` in its output directory.

## 8. How to keep the AI honest
- **Ask for evidence**: file and line for claims about code, file and value for claims about results, and whether it
  actually ran the command.
- **Small N first**; agree on time and memory before N ≥ 8.
- **Never accept a relaxed threshold** as a fix. A gate failure is a result to report.
- **Signs are arbitrary** between runs (eigenvector phase); compare magnitudes. Energies, C₂ and parities must agree.
- **Do not let it edit or delete run directories**; reruns go to new output directories.
- **Finite N**: numbers are raw finite-size estimates; asking for "the critical exponent" requires an extrapolation
  this code does not do.

## 9. Limits
v0.3.x was regressed against the earlier per-N code on N5 and N6 only; the N12 reference plan covers fewer sectors.
See METHODS §14.
