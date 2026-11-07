# Tcc 参考手册：代码结构、plan 文件、step、测试

English: [../en/GUIDE.md](../en/GUIDE.md) · 简介：[../../README.md](../../README.md) · 小 N 复现：[REPRODUCE.md](REPRODUCE.md) · 方法：[METHODS.md](METHODS.md) · 你的数据：[DATA_LAYOUT.md](DATA_LAYOUT.md)。学术名词保留英文原词。

## 1. 代码结构

| 路径 | 内容 |
|---|---|
| `src/model.jl`、`ph.jl`、`seglib.jl`、`exchange.jl` | 模型、segment space（直接 / particle–hole 半构造）、segment 算符库、segment exchange X |
| `src/resources.jl` | `ph_half` 与 `residency` 按规模的默认值 |
| `src/runner.jl`、`matvec.jl` | (s_A, s_B, L) block 的 spectrum、residency 模式、H 作用 kernel |
| `src/targets.jl`、`edges.jl`、`kkstage.jl`、`ZhouGenerator.jl`、`kk/KKProjection.jl` | KK target window、edge、真空拟合、K² 分类 |
| `src/tuning.jl` | coupling tuning F(g) = 0 |
| `src/ope.jl`、`ope3.jl` | pipeline OPE；三点 insertion 与扩展 OPE |
| `src/parity.jl` | 物理 parity |
| `src/stageb.jl` | Stage B：(½,½) 与 (1,1) sector 到 raw Δ 截断、KK、full-Q OPE、raw target；window 证书、partial checkpoint、preliminary top window |
| `src/fullq.jl` | integer sector 的 full-Q OPE |
| `src/post.jl` | coverage、逐 pair target、ope3 与 parity 驱动 |
| `src/plan.jl` | plan（`[plan]`）与 step runner |
| `src/pipeline.jl` | pipeline stage、manifest、续算、来源记录 |
| `bin/tcc.jl` | 唯一入口（数据根目录、step） |
| `bench/*_run.jl` | 单个 step 的命令行形式 |
| `tools/ope3_parity_join.py` | φ×φ tower 的 parity |
| `tools/data_inventory.jl` | 由 run 记录生成你的数据清单 |
| `tools/check_quickstart.py` | N5 quick start 的 PASS/FAIL 检查 |
| `analysis/tools/demo_small_n.py` | 小 N 分析演示：KK 导出、κ 门槛、规则 C、家族检验 |
| `configs/quickstart_N5.toml`、`example_N6.toml` | 小规模 run |
| `configs/examples/N05.toml … N12.toml` | 参考 plan：原始研究在每个 N 算了什么 |
| `cluster/tcc.sbatch.example` | 一个 plan 的 Slurm 作业 |
| `test/` | 测试（§5） |
| `analysis/` | KK 输出的 primary/descendant 分析（Python），每条规则一个目录：`hardgate/`（规则 A，所有 N 一起）、`switch_kk/`（规则 C，单个 N，含家族检验）；见 [ANALYSIS.md](ANALYSIS.md) |
| `deps/` | 环境构建脚本与替换的那一个 FuzzifiED 文件 |

## 2. 一次计算一个文件：plan

```toml
[run]
N = 8
g = 0.8903705662882018
out = "results/main/N8"          # 相对路径：在数据根目录下

[plan]
sectors = "all"            # 或 [[0,0],[0,1],[0,2]]：integer (sA,sB) pair
lmax = 4                   # block L = 0..lmax（KK source）；KK target window 到 L = lmax+1
delta_max = 6.2            # target window 覆盖到 Delta <= 6.2 的 KK source
half = true                # pipeline OPE 中包括 (1/2,1/2) sector
steps = ["coverage", "ope3", "parity", "parity_join", "fullq", "stageb", "rawtarget"]

[spectrum]                 # window（每个 block / X sector 的态数）
nev = 80
[coverage]                 # 扩窗直到每个 block 到 Delta >= 6.0（或已完整）
delta_cut = 6.0
kmax = 400
[targets]
initial_nev = 80
max_nev = 4000
[ope3]                     # 见 METHODS 第 9 节
int_ranks = [1, 2, 3, 4]
half_lmax = 3
[fullq]                    # 默认：run 的全部 pair；rank_max = 4
[stageb]                   # sector "half" = (1/2,1/2)，"11" = (1,1)
sectors = ["half"]
lmax = 4                   # KK source 与 OPE spin 0..4，raw window 0..5
raw_delta_max = 6.8
```

```bash
julia --project=env --threads=64 bin/tcc.jl N08.toml                    # plan 的全部 step
julia --project=env --threads=64 bin/tcc.jl N08.toml stageb             # 单个 step
julia --project=env --threads=64 bin/tcc.jl N08.toml --data=/scratch/me # 换一个数据根目录
```

* plan 展开成 pipeline 的键（`spectrum.blocks` = sectors × L0..lmax，按字典序；`targets.source_delta_max`；`ope.half`）；同一个键
  给出两个不同的值会报错。
* **step**：`basis tune spectrum targets kk ope summary`（pipeline stage）、`coverage`、`targets_pairs`（`[targets_pairs] pairs =
  [[a,b],…]`：部分 pair 的 target window，可在多个节点并行）、`ope3`、`parity`、`parity_join`、`fullq`、`stageb`、`rawtarget`。
  输出目录默认 `<run.out>_<step>`（`<run.out>_stageb_<sector>`）；step 表里的 `out = …` 可覆盖。
* 每个 step 都从自己的 checkpoint 续算；每个 checkpoint 都绑定物理输入（N、g、basis SHA、window），重启不会混用不同输入的数据。
  pipeline 拒绝物理键不同的 run 目录。
* **记录**：pipeline 把 plan 存进 `manifest.toml`；其余 step 在输出目录写 `plan.toml`（plan、step 选项、代码版本与 commit）。
  `tools/data_inventory.jl` 由这些记录生成你的数据表。

只靠 plan 键就能表达的变体：

| 计算 | plan |
|---|---|
| half 与 (1,1) sector，spin 0–4 | `[stageb] sectors = ["half", "11"]`，`lmax = 4` |
| half sector，spin 0–3 | `[stageb] lmax = 3` |
| 最大的 N：并行 window 作业 | `[stageb] lmax = 1`、`step = "windows"`，一个作业 `windows = [0, 1]`，另一个 `[2]`，最后 `step = "all"` |
| 长 window 作业干净停止、以后续算 | `[stageb] partial_checkpoints = true`（默认），`step = "windows"` 时 `stop_after_k = k` |
| 低成本 window 证书 | `[stageb] extra_states = 4`、`complement_tol = 1e-6`、`final_polish_trigger = 1e-9`（默认）；`0, 1e-10, 0` = v0.3.0 的证书 |
| 由部分 top window 得到初步 KK/OPE | `[stageb] prelim_top_window = "<out>/windows/partial/L2_k081.ckpt"`、`prelim_complement = true` |
| OPE 投到完整 raw window | step `rawtarget`（或 `[stageb] rawtarget = true`） |
| 选定 sector 的 full-Q OPE | step `fullq`，`[fullq] sectors = [[0,1],[0,2],[0,0]]`、`rank_max = 3` |
| 多节点逐 pair target window | step `targets_pairs` |
| KK 只做部分 sector / L / pair | `plan.sectors`、`kk.pairs`、`kk.lmax` |

## 3. 参考 plan
`configs/examples/N05.toml … N12.toml` 重述了原始研究在每个 N 的计算（sector、L、window、Δ 截断、后处理 step、tuning 后的 g）。
可作为起点；N ≥ 8 之前先看 block 维数（METHODS §2.4）——N10 起需要集群节点，N12 需要数百 GB 内存和数天。

## 4. 资源开关
见 METHODS §2.5：`basis.ph_half`（最大 segment sector ≥ 2000 → PH 半构造）、`spectrum.residency`（segment basis > 16e9 字节 → `pair`）、
`threads.native_dim_max`（block 维数 2×10⁷）、Stage B 内存上限（Slurm 分配的 85 %）。都不改变结果，都可显式设定。

## 5. 测试

| 测试 | 检查内容 |
|---|---|
| `tools/check_quickstart.py <out>/N5` | 按预期性质与参考值检查 N5 quick start（[REPRODUCE.md](REPRODUCE.md) §2） |
| `test/regress.jl N out [golden]` | segment 算符、带/不带 X 的 spectrum、residency 模式、KK/Γ² edge、golden 对照 |
| `test/t1_equivalence.jl` … `t10_highscalar.jl` | 基本组件：等价性、exchange 求解、runner、哈希、OPE、高 scalar |
| `test/t11_extend.jl`、`t12_gates.jl` | 扩窗；验收 gate |
| `test/t13_ph.jl N out projectors` | PH 半构造张成与直接构造相同的 multiplet 子空间 |
| `test/t14_v028.jl` | half sector edge 身份、上游 digest |
| `test/t15_stageb_restart.jl` | Stage B 重启复用 window、edge、projection；部分丢失的输出会补齐 |
| `analysis/hardgate/tests/`、`analysis/switch_kk/**/test_*.py` | 分析代码的合成数据测试 |

测试输出目录请放在数据根目录下（例如 `<数据根目录>/tests/regress`）。结果在舍入误差内可重现，但跨进程不是逐位相同（多线程 BLAS
改变末位，从而改变 eigenvector 符号）：比较绝对值。制作 v0.3.0 时，在 N5、N6 上把完整链与早期按 N 的代码对照过，差异与同一代码
两次运行之间的差异同量级（绝对值 ≤ 2e-11）。
