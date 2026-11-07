# 借助 AI 使用和检查 Tcc —— 快速上手

English: [../en/AI_GUIDE.md](../en/AI_GUIDE.md)。写给想借助 AI coding agent（Claude Code、Codex 等）运行、理解或检查这套代码的人。
从零开始到跑完并亲自核对一次完整的 N = 5 计算，大约 30 分钟。本文与英文版逐节对应，学术名词保留英文原词。

---

## 1. 需要什么
- Julia **1.11**（已用 1.11.7 验证）、git、Python 3（只在一个小的 join step 用到）；N ≤ 6 需要 8 个 CPU 线程、4 GB 内存。
- 一个能读文件、能在终端里执行命令的 AI coding agent。Claude Code 和 Codex 都会自动读取仓库根目录的说明文件
  （`CLAUDE.md` / `AGENTS.md`）：里面写了代码是什么、怎么运行，以及保护物理结果的规则（不许放宽 gate、不许混用 checkpoint、
  只给有限 N 的数）。

## 2. 安装（一次）
```bash
git clone <你的仓库副本> fuzzysphere && cd fuzzysphere
cp user.example.toml user.toml        # 设置 [paths] data_root：输出写到哪里
julia deps/setup_env.jl
```
`setup_env.jl` 安装 FuzzifiED 2.0.1 和 KrylovKit 0.10.4，替换 Tcc 需要的那一个 FuzzifiED 文件（替换了什么、为什么，见
`deps/README.md`），并逐个核对两个包的源文件与参考哈希。最后打印 `TCC_ENV_READY`。然后在这个目录里启动 agent。
`bin/tcc.jl` 的输出写到你的数据根目录下（[DATA_LAYOUT.md](DATA_LAYOUT.md)），不写进仓库。

## 3. 第一轮对话：让 AI 先读懂项目
把下面这段作为第一条消息发给它：

> 读 AGENTS.md、README.md、docs/zh/GUIDE.md、docs/zh/METHODS.md 和 docs/zh/DATA_LAYOUT.md，先不要运行任何东西。用一页纸说明：这套代码研究什么物理
> 系统；`bin/tcc.jl` 的各个 step 算什么、按什么顺序；最终数值在哪些输出文件里；用哪个量 tuning 来确定 coupling g。
> 每句话都注明出处文件。

自己对照 GUIDE §2 和 METHODS §4–§7 检查它的回答。它若说了某件事却没给出处，追问。

## 4. 跑一次完整计算（N = 5，几分钟）
> 运行 `julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml`。跑完后列出输出文件，每个用一行说明内容。

主要输出在 `<数据根目录>/results/quickstart/N5/` 下：

| 文件 | 内容 |
|---|---|
| `run/summary.tsv` | 每个算出的态：block (s_A, s_B, L)、exchange sector、能量、Δ、C₂、残差 |
| `run/physics_report.toml` | g、F(g)、E₀、α、KK 拟合、primary candidate 个数、pipeline OPE |
| `run/kk/projected_states.csv`、`run/physics_primaries.csv` | KK projection 后的 primary candidate 及其 K² |
| `ope3/*.csv`、`ope3/ope3.toml` | 三点系数和 φ×φ tower（parity 在 `ope3_phiphi_spin_parity.csv`） |
| `parity/parity.csv`、`parity_kk.csv` | 每个态和每个 KK mode 的物理 parity |
| `fullq/xy_blocks_raw.csv` | integer sector 的 full-Q OPE 分量 |
| `stageb_half/`、`stageb_11/` | (½,½) 和 (1,1) sector：原始 spectrum、KK mode、OPE 分量、raw-target edge |

## 5. 检查结果 —— 让它给出具体数值
每一项都有可以在 agent 指出的文件里自己核实的预期结果。`python tools/check_quickstart.py <数据根目录>/results/quickstart/N5`
一次跑完全部检查（[REPRODUCE.md](REPRODUCE.md) §2）；让 agent 指出每个数的来源，而不只是运行脚本。

| # | 让 agent 给出 | 预期 | N5 quick start 实测（参考） |
|---|---|---|---|
| 1 | `run/physics_report.toml` 里的 `F` | \|F\| < 1e-5（tuning 后的 g 使 Δ_T = 3） | −3.5e-7 |
| 2 | `summary.tsv` 里 (0,1) L1 最低 C₂ = 3 态和 (0,0) L2 最低 C₂ = 0 态的 Δ | 前者恰为 2（定义），后者 ≈ 3 | 2.0 和 2.9999996 |
| 3 | `summary.tsv` 里最大的 `H_res` 和 `C2_res` | ≤ 1e-6（否则代码会停），实际 ~1e-10 或更小 | 6.6e-14、1.8e-10 |
| 4 | 同一个 SO(5) multiplet 在不同 (s_A, s_B) block 中出现：(L, C₂, 能量) 相同 | 能量一致到 ~1e-12 或更好 | 316 对（按 `check_quickstart.py` 的计数方式），最大 \|ΔE\| 3.6e-14 |
| 5 | `run/ope/ope.toml` 里的 `half_vs_integer_phi_energy_difference` | ~1e-14：half-integer 和 integer sector 给出同一个 φ | 8.0e-15 |
| 6 | `ope3/ope3_phiphi_spin_parity.csv` 中所有 P = −1 行的 `ratio` 列 | ~0（≤ 1e-12）：φ×φ 只含 P = +1 | 99 行，最大 9.2e-15 |
| 7 | `stageb_half/raw_current_control.toml` 里的 `status` | `pass`：current 在 half sector 中 Δ = 2、P = +1 | pass |
| 8 | `parity/parity_summary.toml` 里的 `max_residual` 和 `kk_modes_mixed_t0.25` | ≤ 1e-6，mixed mode 为 0 | 6.4e-11、0 |
| 9 | `run/physics_report.toml` 里 t = 0.25 的 primary candidate 个数 | 重跑之间不变 | 171 个 candidate |

矩阵元和 OPE 系数的符号在不同运行之间可能不同（eigenvector 相位任意）；其余量应在舍入误差内一致。

然后跑测试：
> 运行 `julia --project=env --threads=8 test/regress.jl 5 <数据根目录>/tests/regress` 和 `test/t15_stageb_restart.jl 5
> <数据根目录>/tests/t15`，报告每一行 PASS/FAIL。

## 6. 检查代码 —— 好用的审查请求
> 把 `src/model.jl: build_terms` 与 METHODS §1.2–1.3 逐项对照：interaction、pairing、C₂，以及唯一与 N 有关的分支。
> 任何不一致都给出文件和行号。

> 找出所有施加 tolerance 或 gate 的地方（`src/gates.jl`、`window_gate`、`GateError`）。逐个说明它保护什么，以及是否有
> 代码路径能绕过它。

> 读 `src/stageb.jl`，解释重启时如何决定复用一个 window、一个 edge 或一个 projection。用另一个 g 或另一个 basis 算出的
> checkpoint 有没有可能被复用？列出身份字段。

> 不依赖 Tcc，用 FuzzifiED（`GetDenIntTerms`、`GetPairIntTerms`，相同量子数下的完整 Fock 空间）为 N = 4 写一个简短的
> exact diagonalisation 检查，把 (0,0) L0 的最低能量与 `summary.tsv` 比较。不要修改 `src/`。

最后这类独立检查（另写一个程序，而不是用同一份代码）是最有力的证据；编写代码时对 N ≤ 6 做过这类检查。

## 7. 规划自己的计算
> 我想在 N = 7 算 (0,0) 和 (0,1) sector，L 到 3，Δ coverage 6.0。以 `configs/examples/N07.toml` 为基础写 plan 文件，
> 告诉我最大的 block 维数（METHODS §2.4）和预计内存，我确认之前不要开始。

plan 的写法见 GUIDE §2；每个 step 都会在输出目录的 `plan.toml` 里记录自己的 plan。

## 8. 怎样让 AI 保持诚实
- **要证据**：关于代码的说法要文件和行号，关于结果的说法要文件和数值，还要说明它是否真的运行了命令。
- **先小 N**；N ≥ 8 之前先商定时间和内存。
- **绝不接受放宽阈值**作为修复。gate 失败本身就是要报告的结果。
- **符号任意**：不同运行之间 eigenvector 相位可能不同，比较绝对值。能量、C₂ 和 parity 必须一致。
- **不要让它修改或删除 run 目录**；重跑写到新的输出目录。
- **有限 N**：这些数是原始的有限尺寸估计；要"临界指数"需要外推，这套代码不做外推。

## 9. 局限
v0.3.x 只在 N5、N6 上与早期按 N 的代码做过回归对照；N12 参考 plan 覆盖的 sector 较少。另见 METHODS §14。
