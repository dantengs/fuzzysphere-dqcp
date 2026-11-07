# Tcc 输出的 primary/descendant 分析（`analysis/`）

English: [../en/ANALYSIS.md](../en/ANALYSIS.md)。命令：[analysis/README.md](../../analysis/README.md)。学术名词保留英文原词。

Tcc 的 KK 步骤（METHODS §7）对每个 guarded eigenstate 给出降生成元到相邻 L 各态的矩阵元 M_ts。哪些本征态是 primary、哪些是
descendant，之后用 Python 从这些导出矩阵判定。模板含两条规则，每条一个目录，包括代码、成文规则（`protocol.md` / `docs/`）、
合成数据测试和冻结机制。

## 1. 规则

两条规则都先去掉等能带 |ω| ≤ atol（角动量 L^μ，与 D 对易），再按能量变化 ω = Δ_t − Δ_s 把生成元拆成降部分 K 与升部分 P。
primary 判据是 κ = λ(K†K)/(2Δ̄)，即固定 (SO(5) C₂, L, P) 扇区内 K†K 的小本征值。

| | 规则 A — 硬切（`hardgate/`，blindpd 0.5.1） | 规则 C — 有界开关（`switch_kk/`，v3） |
|---|---|---|
| K | ω < 0 时 K_ts = M_ts | K_ts = f_w(ω) M_ts，f_w = clip((w − ω)/2w, 0, 1)，w ∈ {0（= A）, 0.25, 0.5}；arXiv:2611.03817 式 (4.2) 的线性权重 (1 − ω)/2 只作参考 |
| 门槛 | 若干尺寸 log₁₀ κ 直方图的谷值 κ_c（`blindpd/thresholds.py`、`tools/kappa_threshold.py`） | 同一 κ_c，读自冻结的 `threshold_from_raw.json`；稳定带为其谷值 bin |
| 逐态判定 | 低 κ 子空间中原始能级权重 ≥ 0.90；descendant span 选项 B，到第 7 级；跨 N 追踪（跳变 ≤ 0.3） | 3 种方法 × 3 个门槛下的 robust / sensitive / low |
| 家族 | descendant span（`blindpd/descspan.py`，规则 `docs/RULES_descendant_span_v1.md`） | `family/`：K 像的非相干父态传播；`family/coherent/`：带符号的 P 映射逐级生成后裔子空间，按扇区做 Löwdin 分解 |
| 尺寸 | 所有已算尺寸一起（拟合、追踪） | 每次一个尺寸 |
| 锚点 | — | J（s0_s1_L1 #1，Δ = 2）与 T（s0_s0_L2 #2，Δ = 3）必须为 robust，否则如实报告失败 |

w = 0 的规则 C 就是规则 A 的拆分。规则 C 在保存的 window 内是精确的：只使用所有出边 frontier 都高于 Δ_s + 0.5 的 source，于是
window 外的 target 都有 ω > w、f = 0。线性权重没有这种精确性；有限尺寸下后裔能隙偏离 1 时，primary 自身的 P 跃迁会漏进 K，
这是采用有界开关的原因。

## 2. 输入

| 输入 | 生成方式 | 使用者 |
|---|---|---|
| `kk/N<n>/`（meta.json + edges/*.f64） | 在存放 run 的机器上运行 `analysis/hardgate/cluster/export_kk.jl <run.out> <out>` | A、C、κ 门槛 |
| `physics/N<n>/`（parity.csv、ope3.toml 等） | 从 parity、ope3 step 的输出复制 | A、C |
| `ope/N<n>/` | `export_ope_edges.jl <fullq out> <out>` | A |
| Stage B 导出、image norm、raw-target 导出 | `export_stageB.jl`（`export_stageB.sbatch`） | A（可选的 `stageB` 键） |
| 合并的逐 N OPE 行（`ope_per_N_rows`） | 你自己汇总的 OPE 表 | A |
| `threshold_from_raw.json` | `hardgate/tools/kappa_threshold.py POOL OUT [SIZES]` | C（冻结输入） |

导出脚本只读 run 目录。导出结果放在你的数据根目录下（例如 `<数据根目录>/analysis/inputs/`）；`.gitignore` 排除了 `analysis/` 下的
`inputs/` 与 `results/`。

## 3. 流程

规则 C 的完整小 N 运行（分析 N5，N6 作 κ pool）只需一条命令 `analysis/tools/demo_small_n.py`，见 [REPRODUCE.md](REPRODUCE.md) §5。
它执行的步骤如下：

1. **规则 A。** 把 `analysis/paths.example.json` 复制为 `analysis/paths.local.json` 并填入你的根目录；用
   `analysis/tools/localize.py hardgate/config/formal.template.json paths.local.json <base>/config/formal.json` 填好模板；按你算过的
   尺寸设置 `sizes`、`main_fit_sizes`、`window_fit_sizes` 和 `stageB` 的尺寸；在 `analysis/hardgate` 下运行
   `python -X utf8 -B -m blindpd.formal <base>/config/formal.json <空的输出目录>`。
2. **单个尺寸的规则 C。** 用其他尺寸生成 `threshold_from_raw.json` 并冻结，然后在 `analysis/switch_kk` 下：`python test_v3.py`、
   `python run.py freeze <inputs> <threshold>`、`python run.py run <inputs> <threshold>`、`python spectrum.py`、
   `python modes_readout.py`；再做 `family/` 和 `family/coherent/`（freeze、run、readout）。inputs 目录里恰好有一个 `kk/N<n>`。
3. **盲性边界。** 做盲分析时，在开发、冻结和运行期间不看文献中的 P/D 标签或数值，也不打开 `hardgate/comparison/unblind.py`
   （其中含文献数值）。只在结果冻结后，作为单独一步与文献比较。

冻结会记录代码、protocol、测试和输入的 SHA-256（输入路径记相对路径，输入目录只记名字）。冻结后不再修改规则和门槛；改规则
须另建带版本号的目录并单独冻结。若程序在写出任何结果之前因编程错误中断，只能修该错误；首次冻结另存（例如 `audit/preflight/`），
再重新冻结。

## 4. 限制
- 规则 A 需要合并的逐 N OPE 行，用 `stageB` 时还需要 image norm；本模板没有汇总它们的工具。缺少时只能运行规则 A 中使用 KK
  导出的部分（`blindpd.run` 会读取 OPE 行）。
- 规则 C 是单尺寸分析，不做外推。超出导出 block 覆盖范围的态（例如 L = 4 的 target）记为未解析或秩缺额，不做猜测。
- 规则 A 的 config 键 `L_max`、`null_singular_value`、`bin_width_log10` 在 blindpd 中部分写死。
- `CODE_MANIFEST_v0.5.1.sha256` 与 `FROZEN_RULES.sha256` 描述本模板副本中的文件；用 `analysis/tools/manifest.py check` 核对。
