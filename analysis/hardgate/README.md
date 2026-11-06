# Rule A — hard gate (blindpd 0.5.1)

Primary/descendant classification of the KK output of all sizes together: hard K/P split, a κ threshold taken from
the valley of the log₁₀ κ histogram of a pool of sizes, raw-level weight gate 0.90, descendant span option B
(`config/descendant_span_v1.json`, rules in `docs/RULES_descendant_span_v1.md`), tracking across N, exact OPE
recontraction, harden checks and S3/S4 resolution. Overview of all rules: `docs/en/ANALYSIS.md` of the repository.

```bash
# from analysis/hardgate (Python ≥ 3.10, numpy, pandas, scipy)
python -X utf8 -B tests/test_units.py            # and the other tests/
python -X utf8 -B tools/kappa_threshold.py KAPPA_POOL threshold_from_raw.json
python ../tools/localize.py config/formal.template.json ../paths.local.json <base>/config/formal.json
python -X utf8 -B -m blindpd.formal <base>/config/formal.json <empty out dir>
```

- `cluster/export_kk.jl`, `export_ope_edges.jl`, `export_stageB.jl` turn Tcc run directories into the portable inputs
  (run them where the data are; they only read).
- `comparison/unblind.py` compares a frozen result with published tables and contains published values. Keep it out of
  any blind analysis; run it only as a separate step after the result is frozen.
- `CODE_MANIFEST_v0.5.1.sha256` lists the SHA-256 of the code, configs and docs of this copy
  (`python ../tools/manifest.py check hardgate/CODE_MANIFEST_v0.5.1.sha256` from `analysis/`);
  `FROZEN_RULES.sha256` the frozen rule files. Change a rule only in a new versioned directory.

中文：规则 A（硬切）。一次处理所有尺寸：硬切 K/P，κ 门槛取若干尺寸 log₁₀ κ 直方图的谷值，原始能级权重门槛 0.90，
descendant span 选项 B，跨 N 追踪，精确 OPE 重收缩，harden 检查与 S3/S4 消解。命令见上。`comparison/unblind.py` 含文献数值，
盲分析期间不要打开，只在结果冻结后单独运行。
