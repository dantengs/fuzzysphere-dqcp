# Primary/descendant analysis of the Tcc outputs (`analysis/`)

Chinese version: [../zh/ANALYSIS.md](../zh/ANALYSIS.md). Commands: [analysis/README.md](../../analysis/README.md).

The KK step of Tcc (METHODS §7) gives, for every guarded eigenstate, the matrix elements M_ts of the lowering
generator to the states of neighbouring L. Which eigenstates are primaries, and which descendants, is decided
afterwards from these exported matrices, in Python. The template contains two rules, one directory each, with code,
written rules (`protocol.md` / `docs/`), tests on synthetic data and a freeze mechanism.

## 1. The rules

Both split the generator by the energy change ω = Δ_t − Δ_s into a lowering part K and a raising part P, after removing
the equal-energy band |ω| ≤ atol (the angular momentum L^μ, which commutes with D). The primary criterion is
κ = λ(K†K)/(2Δ̄), a small eigenvalue of K†K in a sector of fixed (SO(5) C₂, L, P).

| | Rule A — hard gate (`hardgate/`, blindpd 0.5.1) | Rule C — bounded switch (`switch_kk/`, v3) |
|---|---|---|
| K | K_ts = M_ts for ω < 0 | K_ts = f_w(ω) M_ts, f_w = clip((w − ω)/2w, 0, 1), w ∈ {0 (= A), 0.25, 0.5}; the linear weight (1 − ω)/2 of arXiv:2611.03817 eq. (4.2) is reported for reference only |
| threshold | κ_c from the valley of the log₁₀ κ histogram of a pool of sizes (`blindpd/thresholds.py`, `tools/kappa_threshold.py`) | the same κ_c, read from a frozen `threshold_from_raw.json`; stability band = its valley bin |
| per-state decision | raw-level weight ≥ 0.90 in the low-κ subspace; descendant span option B up to level 7; tracking across N (jump ≤ 0.3) | robust / sensitive / low over 3 methods × 3 thresholds |
| families | descendant span (`blindpd/descspan.py`, rules `docs/RULES_descendant_span_v1.md`) | `family/`: incoherent parent propagation of the K image; `family/coherent/`: signed P maps generate the descendant subspaces level by level, Löwdin decomposition per sector |
| sizes | all computed sizes together (fits, tracking) | one size per run |
| anchors | — | J (s0_s1_L1 #1, Δ = 2) and T (s0_s0_L2 #2, Δ = 3) must be robust, else the run is reported as a failure |

Rule C with w = 0 is rule A's split. Within the saved windows C is exact: a source is used only if every outgoing
frontier lies above Δ_s + 0.5, so every target outside the window has ω > w and f = 0. The linear weight is not exact in
this sense and lets a primary's own P transitions leak into K when the finite-size descendant gap differs from 1; this
is why the bounded switch is used.

## 2. Inputs

| Input | Made by | Used by |
|---|---|---|
| `kk/N<n>/` (meta.json + edges/*.f64) | `analysis/hardgate/cluster/export_kk.jl <run.out> <out>` on the machine that holds the run | A, C, κ threshold |
| `physics/N<n>/` (parity.csv, ope3.toml, …) | copy from the parity and ope3 step outputs | A, C |
| `ope/N<n>/` | `export_ope_edges.jl <fullq out> <out>` | A |
| Stage B exports, image norms, raw-target exports | `export_stageB.jl` (`export_stageB.sbatch`) | A (optional `stageB` key) |
| combined per-N OPE rows (`ope_per_N_rows`) | your own collection of the OPE tables | A |
| `threshold_from_raw.json` | `hardgate/tools/kappa_threshold.py POOL OUT [SIZES]` | C (frozen input) |

The exporters only read the run directories. Keep the exports under your data root (for example
`<data root>/analysis/inputs/`); `.gitignore` excludes `inputs/` and `results/` directories under `analysis/`.

## 3. Workflow

A complete small-N run of rule C (N5 analysed, N6 as the κ pool) is one command, `analysis/tools/demo_small_n.py`;
see [REPRODUCE.md](REPRODUCE.md) §5. The steps it performs are:

1. **Rule A.** Copy `analysis/paths.example.json` to `analysis/paths.local.json` and set your roots; fill the template
   with `analysis/tools/localize.py hardgate/config/formal.template.json paths.local.json <base>/config/formal.json`; set
   `sizes`, `main_fit_sizes`, `window_fit_sizes` and the `stageB` sizes to what you computed; run
   `python -X utf8 -B -m blindpd.formal <base>/config/formal.json <empty out>` from `analysis/hardgate`.
2. **Rule C for one size.** Build `threshold_from_raw.json` from other sizes, freeze it, then in `analysis/switch_kk`:
   `python test_v3.py`, `python run.py freeze <inputs> <threshold>`, `python run.py run <inputs> <threshold>`,
   `python spectrum.py`, `python modes_readout.py`; then `family/` and `family/coherent/` (freeze, run, readout). The
   inputs directory holds exactly one `kk/N<n>`.
3. **Blind boundary.** For a blind analysis do not look at published P/D labels or values while developing, freezing or
   running, and do not open `hardgate/comparison/unblind.py` (it contains published values). Compare with the literature
   only after the result is frozen, as a separate step.

A freeze records the SHA-256 of code, protocol, tests and inputs (input paths relative, the input directory by name
only). After a freeze, rules and thresholds are not changed; a change of rule is a new versioned directory with its own
freeze. If a program stops on a programming error before it writes any result, only that error may be fixed; keep the
first freeze (for example in `audit/preflight/`) and freeze again.

## 4. Limits
- Rule A needs the combined per-N OPE rows and, with `stageB`, image norms; this template has no tool that collects
  them. Without them only the parts of rule A that use the KK export can be run (`blindpd.run` reads the OPE rows).
- Rule C is a single-size analysis; it does not extrapolate. States above the coverage of the exported blocks
  (for example L = 4 targets) are counted as unresolved / rank deficits, never guessed.
- The rule-A config keys `L_max`, `null_singular_value` and `bin_width_log10` are partly hard-coded in blindpd.
- `CODE_MANIFEST_v0.5.1.sha256` and `FROZEN_RULES.sha256` describe the files of this template copy; check them with
  `analysis/tools/manifest.py check`.
