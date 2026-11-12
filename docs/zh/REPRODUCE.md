# 小 N 计算的复现

English: [../en/REPRODUCE.md](../en/REPRODUCE.md) · 简介：[../../README.md](../../README.md) · 参考手册：[GUIDE.md](GUIDE.md) ·
方法：[METHODS.md](METHODS.md)。学术名词保留英文原词。

本页从一台空机器出发，带你跑完 N = 5 的完整计算、测试，以及 N = 5 的 primary/descendant 分析，并给出应当得到的数值。下面的
每条命令和每个用时都按这个顺序实际跑过：全新 clone，空的 Julia 包目录（Windows 11、Julia 1.11.7、Python 3.14、8 线程）。

**这里“复现”的含义。** 能量、Δ、C₂、parity、矩阵元绝对值和各种计数在舍入误差内一致（相对差 ≲ 1e-10）。eigenvector 的
**符号**任意，因此矩阵元和 OPE 系数的符号在不同运行之间可能不同；并行求解写出的表格行序、计时和内存列也可能不同。代码中的
gate 会让残差或窗口不合格的 run 停下来，不会放宽任何门槛。

| 步骤 | 命令（在仓库根目录） | 用时 |
|---|---|---|
| 0 | `cp user.example.toml user.toml`，设置 `data_root` | — |
| 1 | `julia deps/setup_env.jl` | 1.5 分钟 |
| 2 | `julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml`，再 `python tools/check_quickstart.py <数据根目录>/results/quickstart/N5` | 2.5 分钟 |
| 3 | 测试（§3） | 5 分钟 |
| 4 | `julia --project=env --threads=8 bin/tcc.jl configs/examples/N06.toml coverage parity` | 2 分钟 |
| 5 | `python analysis/tools/demo_small_n.py <数据根目录> --size 5 --pool 6` | 15 秒 |

磁盘：全部输出约 110 MB，另有约 150 MB 的 Julia 包。内存：这些尺寸都低于 2 GB。

---

## 0. 获取代码，选定数据位置

```bash
git clone https://github.com/dantengs/fuzzysphere-dqcp.git
cd fuzzysphere-dqcp
cp user.example.toml user.toml
```

编辑 `user.toml` 的 `[paths] data_root`（默认 `data`，即 `./data`，已被 git 忽略）。下面所有输出都写在它下面；`bin/tcc.jl`
第一行会打印 `TCC_DATA_ROOT <目录>`。也可以不用 `user.toml`，改用 `--data=DIR` 或环境变量 `TCC_DATA`（[DATA_LAYOUT.md](DATA_LAYOUT.md)）。
Windows 上把 `cp` 换成 `Copy-Item`，其余命令在 PowerShell 中相同。

安装 Julia 1.11（例如用 [juliaup](https://github.com/JuliaLang/juliaup)：`juliaup add 1.11`，再用 `julia +1.11 …` 或设为默认）
和 Python ≥ 3.10：

```bash
python -m pip install -r analysis/requirements.txt     # numpy、pandas
```

## 1. 建立 Julia 环境

```bash
julia deps/setup_env.jl
```

它从 General registry 把 FuzzifiED 2.0.1 和 KrylovKit 0.10.4 装进 `env/`，替换 Tcc 需要的那一个 FuzzifiED 文件（替换了什么、
为什么，见 `deps/README.md`），并把两个包的每个源文件与参考 SHA-256 比对。结尾必须是

```
FuzzifiED.jl: 48/48 source files identical to production
KrylovKit.jl: 36/39 source files identical to production; expected difference (production-only LockedLanczos, unused by Tcc): src/KrylovKit.jl, src/algorithms.jl, src/eigsolve/lockedlanczos.jl
TCC_ENV_READY  from <仓库目录>:  julia --project=env --threads=8 bin/tcc.jl <run.toml>
```

任何其他差异都会让脚本停止。其余依赖解析到最新的兼容版本（本次记录的运行中：FuzzifiED_jll 1.3.0+0、SpecialFunctions 2.9.1、
WignerSymbols 2.0.0），它们对数值的影响不超过舍入。只有这一步需要联网。

## 2. N = 5 的完整链

```bash
julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml
python tools/check_quickstart.py <数据根目录>/results/quickstart/N5
```

plan（`configs/quickstart_N5.toml`）依次运行：coverage（用 tuning 后的 coupling g = 0.8642902899515241 求全部 30 个 block
(s_A, s_B, L ≤ 4) 的 spectrum，窗口扩到 Δ ≥ 6.0；KK target window；KK projection；pipeline OPE）、`ope3`（三点 insertion）、
`parity`、`parity_join`、`fullq`（integer sector 的 full-Q OPE）、(½,½) 与 (1,1) sector 的 `stageb`，以及 `rawtarget`。最后一行是
`TCC_DONE version=0.3.1 stages=basis,kk,ope,spectrum,summary,targets`。输出：

| `<数据根目录>/results/quickstart/N5/` 下的目录 | 内容 |
|---|---|
| `run/` | `summary.tsv`（每个态：block、X、能量、Δ、C₂、残差）、`physics_report.toml`（g、F、α、E₀、KK 拟合、candidate 数）、`kk/projected_states.csv`、`physics_primaries.csv`、`ope/ope.toml` |
| `ope3/` | 三点系数、φ×φ tower、`ope3_phiphi_spin_parity.csv` |
| `parity/` | `parity.csv`、`parity_kk.csv`、`parity_summary.toml` |
| `fullq/` | `xy_blocks_raw.csv` |
| `stageb_half/`、`stageb_11/` | 原始 spectrum、KK mode、OPE 分量、`rawtarget/` |

`check_quickstart.py` 每项检查打印一行，最后是 `ALL PASS`；本次记录的运行输出为

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

各项检查的含义：tuning 后的 coupling 应使 F(g) = Δ_T − 3 为零（METHODS §4）；Δ_J = 2 确定能量标度；残差是 gate 所限制的本征方程
误差与 Casimir 误差；同一个 SO(5) multiplet 出现在多个 (s_A, s_B) block 中且能量相同；half-integer 与 integer sector 的 φ 一致；
φ×φ 只含 P = +1；守恒流在 half sector 中 Δ = 2、P = +1；每个态都是 parity 本征态。参考值（F、α、Δ_T、带 parity 的 639 个 KK mode、
t = 0.25 下的 171 个 primary candidate）的比较容差远大于舍入量级。

## 3. 测试

```bash
julia --project=env --threads=8 test/regress.jl 5 <数据根目录>/tests/regress
julia --project=env --threads=8 test/t15_stageb_restart.jl 5 <数据根目录>/tests/t15
cd analysis/hardgate && for t in test_units test_harden test_fixes_v041 test_descspan; do python -X utf8 -B tests/$t.py; done; cd ../..
```

每个测试最后打印结论；本次记录的运行为

```
REGRESS_PASS N=5 v0.3.1                          （test/regress.jl，101 s）
STAGEB_RESTART_PASS N=5                          （test/t15_stageb_restart.jl，182 s；前面有三行 PASS）
UNIT TESTS PASS / HARDEN UNIT TESTS PASS / V0.4.1 FIX TESTS PASS / DESCSPAN UNIT TESTS PASS   （3 s）
```

规则 C 的分析测试（`test_v3.py`、`test_family.py`）在第 5 步中运行。`test/` 中的其他测试（GUIDE §5）需要的输入或选项写在各自的文件头。

## 4. N = 6 的主链与 parity

```bash
julia --project=env --threads=8 bin/tcc.jl configs/examples/N06.toml coverage parity
```

只运行参考 plan `configs/examples/N06.toml` 中的 `coverage` 与 `parity` 两个 step（命令行给出的 step 会替换 plan 的 step 列表），
输出 `<数据根目录>/results/main/N6` 与 `<数据根目录>/results/parity/N6`。它在这里只作第 5 步 κ 门槛的 pool：被分析尺寸的门槛必须来自
其他尺寸。本次记录的运行中，仅用 N6 的 pool 就得到 κ_c = 0.0421697：与原始研究中预登记的 N6–10 pool 落在同一个谷值 bin，因而数值相同
（门槛取 bin 中心，0.04217）。 要跑完整的 N = 6 plan，去掉命令中的 step 名即可。

## 5. N = 5 的 primary/descendant 分析（规则 C）

```bash
python analysis/tools/demo_small_n.py <数据根目录> --size 5 --pool 6     # julia 不在 PATH 上时加 --julia <路径>
```

脚本（[ANALYSIS.md](ANALYSIS.md)、`analysis/tools/demo_small_n.py`）导出 N5 和 N6 的 KK edge（`analysis/hardgate/cluster/export_kk.jl`），
复制它们的 `parity.csv`，由 N6 pool 计算 κ_c（`hardgate/tools/kappa_threshold.py`），把规则复制到
`<数据根目录>/analysis/demo_N5/switch_kk/`，并在那里依次运行单元测试、`run.py freeze`、`run.py run`、`spectrum.py`、`modes_readout.py`
和非相干家族检验（`family/`：freeze、run、readout）。结尾为

```
DEMO_DONE
kappa_c = 0.0421697 (pool N6)
method_valid = True; anchors: J Delta=2.000000 robust passed=True, T Delta=3.000000 robust passed=True
states 285, central modes 194, sectors 78; results in <data root>/analysis/demo_N5/switch_kk/results
```

(0,x) block 中 265 个 eligible 态里，37 个是 robust 的 primary candidate，18 个 sensitive，210 个 low（`states.csv`）；锚点 J 与 T 都是 robust。非相干家族检验在硬切下给出 51 个 primary、71 个 descendant、135 个 mixed、8 个 unresolved（`family/results/state_families.csv`）。这些是演示门槛下的数字，不是正式分析的结果。

看哪些文件：`switch_kk/results/states.csv`（每个态的状态、三种方法的权重、`support` = robust / sensitive / low）、`levels.csv`
（带 irrep 的 SO(5) 能级）、`mode_levels.csv`（中央低 K†K 方向）、`family/results/state_families.csv` 与 `summary.json`
（每个态的家族、级和标签；槽计数）。

本演示的限制：单一尺寸的 pool 只给出演示用的门槛，不是正式分析中预登记的值（正式分析用 N = 6–10 的 pool）。相干家族检验
（`family/coherent/`）没有运行：它要求每个 guarded ΔL = 0 行都有符号参照，N5 的导出不满足，按其 protocol 会拒绝运行。规则 A
（`analysis/hardgate`，所有尺寸一起）还需要合并的逐 N OPE 行，本模板没有汇总它们的工具（ANALYSIS §4）。

## 6. 更大的 N

同样的命令适用于每个参考 plan `configs/examples/N07.toml … N12.toml`。N ≥ 8 之前先看 block 维数（METHODS §2.4：最大 block 在 N7 为
4 280，N8 为 3.2×10⁴，N9 为 2.5×10⁵，N10 为 2.0×10⁶）；N ≥ 10 需要集群节点（`cluster/tcc.sbatch.example`），N12 需要数百 GB 内存和数天。

## 7. 出了问题怎么办

| 现象 | 原因与处理 |
|---|---|
| `setup_env.jl` 报 `NOT AS IN PRODUCTION` | 某个包的源文件与参考不同；删除 `env/Manifest.toml` 和 `deps/FuzzifiED.jl/`，用 Julia 1.11 重新运行 |
| `parity_join` 启动不了 Python | 把 `TCC_PYTHON` 设为你的 Python，或在 plan 中写 `[parity_join] python = "…"` |
| run 以 `GATE_FAIL` 停止 | 某个残差、窗口或独立检查没有通过：如实报告，不要放宽 gate（METHODS §3.4） |
| run 中断了 | 重新运行同一条命令：每个 step 都从自己的 checkpoint 续算 |
| 想从头再来 | 删除数据根目录下对应的输出目录（不要修改 run 目录里的文件） |
| `demo_small_n.py` 提示演示目录已存在 | 删除 `<数据根目录>/analysis/demo_N5` |
