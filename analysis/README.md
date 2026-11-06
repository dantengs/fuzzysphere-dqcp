# analysis/ — primary/descendant analyses (Python)

Rules, inputs and limits: [docs/en/ANALYSIS.md](../docs/en/ANALYSIS.md) (中文 [docs/zh/ANALYSIS.md](../docs/zh/ANALYSIS.md)).
Python ≥ 3.10 with numpy, pandas, scipy. Run with `python -X utf8 -B` (UTF-8 I/O, no `__pycache__` next to frozen code).

| Directory | Rule | Entry points |
|---|---|---|
| `hardgate/` | A: hard gate, all sizes together (blindpd 0.5.1) | `python -m blindpd.formal CONFIG OUT`, `tools/kappa_threshold.py`, `cluster/export_*.jl`, `tests/` |
| `switch_kk/` | C: bounded switch, one size; `family/` and `family/coherent/` checks | `run.py`, `spectrum.py`, `modes_readout.py`, `charge_fit.py`, `family/family.py`, `family/readout.py`, `family/coherent/coherent.py`, `family/coherent/readout.py` |
| `tools/` | `localize.py` (fill `${NAME}` roots of a template config), `manifest.py` (write/check SHA-256 manifests) | |

```bash
# your paths (ignored by git)
cp analysis/paths.example.json analysis/paths.local.json

# rule A (cwd analysis/hardgate)
python -X utf8 -B tests/test_units.py
python -X utf8 -B tools/kappa_threshold.py <pool> <data root>/analysis/threshold_from_raw.json 6,7,8,9,10
python -X utf8 -B ../tools/localize.py config/formal.template.json ../paths.local.json <base>/config/formal.json
python -X utf8 -B -m blindpd.formal <base>/config/formal.json <empty out dir>

# rule C for one size (cwd analysis/switch_kk; <inputs> holds kk/N<n>/ and physics/N<n>/parity.csv)
python -X utf8 -B test_v3.py
python -X utf8 -B run.py freeze <inputs> <threshold_from_raw.json>
python -X utf8 -B run.py run <inputs> <threshold_from_raw.json>
python -X utf8 -B spectrum.py && python -X utf8 -B modes_readout.py
cd family && python -X utf8 -B test_family.py && python -X utf8 -B family.py freeze <inputs> <thr> && python -X utf8 -B family.py run <inputs> <thr> && python -X utf8 -B readout.py <inputs> <thr>
cd coherent && python -X utf8 -B test_coherent.py && python -X utf8 -B coherent.py freeze <inputs> <thr> && python -X utf8 -B coherent.py run <inputs> <thr> && python -X utf8 -B readout.py
```

Results are written to `results/` next to each script (ignored by git); freezes (`freeze.json`) record only hashes and
the input directory name, so you may commit them to your own repository if you want a record of the frozen rules.
`analysis/.gitattributes` keeps the bytes of these files unchanged on checkout, so their SHA-256 stays valid.

Blind boundary: `hardgate/comparison/unblind.py` contains published values — do not open it during a blind analysis.
