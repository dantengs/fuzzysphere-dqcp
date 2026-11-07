# fuzzysphere-dqcp-template

**English** · [中文](#中文简介)

A complete, reusable computation template for the **SO(5) deconfined quantum critical point (DQCP) model on the fuzzy
sphere**. For any number of orbitals N it computes the finite-size spectrum, the primary candidates by KK projection,
OPE coefficients and physical parities (Julia, code name *Tcc*), and then classifies the states into primaries and
descendants (Python, `analysis/`). Everything needed to rerun the small-N calculations from scratch is in this
repository; your own data stay in a directory you choose.

- **One code path for every N.** The only N-dependent branch of the physics is the odd/even-N sign of the C₂
  cross-density term. Choices that affect only speed or memory are made from the problem size, never from N.
- **One plan file per calculation.** Sectors, spins, energy windows and post-processing steps are keys of a TOML file;
  every step resumes from checkpoints that are bound to their physical inputs.
- **Hard gates.** Residuals, Casimir purity, window completeness and independent checks stop a run instead of being
  relaxed.
- **Your data, your paths.** All outputs go under a data root that you set (`--data`, `TCC_DATA` or `user.toml`); the
  tracked files contain no machine paths or accounts.
- **Bilingual documentation** (English and Chinese) of every formula, threshold and parameter.

## Requirements

| | |
|---|---|
| Julia | 1.11 (tested 1.11.7; e.g. `juliaup add 1.11`) |
| Python | ≥ 3.10 with numpy and pandas (`pip install -r analysis/requirements.txt`) for the analyses and one join step |
| Machine | small N: 8 threads, 4 GB memory, about 300 MB of disk (outputs and Julia packages); internet access once, to install the Julia packages |
| OS | the reproduction below was checked on Windows 11; earlier versions of this code ran the production calculations on Linux clusters (macOS untested) |

## Quick start

```bash
git clone https://github.com/dantengs/fuzzysphere-dqcp-template.git && cd fuzzysphere-dqcp-template
cp user.example.toml user.toml      # set [paths] data_root: where all outputs go (default ./data, ignored by git)
julia deps/setup_env.jl             # once: installs and checks the Julia environment (about 2 minutes)
julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml
python tools/check_quickstart.py data/results/quickstart/N5   # use your data_root; prints PASS/FAIL per check
```

The fourth command runs the whole chain for N = 5 (spectrum, KK projection, OPE, three-point insertions, parity,
full-Q OPE, the (½,½) and (1,1) sectors) in about 2.5 minutes on 8 threads.

## Reproducing the small-N results

[docs/en/REPRODUCE.md](docs/en/REPRODUCE.md) walks through the whole reproduction with the expected numbers:

| Step | What it reproduces | Time (8 threads) |
|---|---|---|
| 1. Environment | FuzzifiED 2.0.1 with its one replaced file, KrylovKit 0.10.4, checked file by file | 1.5 min |
| 2. N = 5 full chain + `check_quickstart.py` | tuned coupling, spectrum, KK candidates, OPE, parity, Stage B | 2.5 min |
| 3. Tests | core regression, Stage B restart, analysis unit tests | 5 min |
| 4. N = 6 main chain + parity | the κ-threshold pool for step 5 | 2 min |
| 5. Primary/descendant analysis of N = 5 | KK export, κ threshold, rule C, family check | 15 s |

Larger N use the same commands with the reference plans `configs/examples/N07.toml … N12.toml`; N ≥ 10 needs a
cluster node (METHODS §2.4 lists the block dimensions).

## Documentation

| Document | Content |
|---|---|
| [REPRODUCE](docs/en/REPRODUCE.md) | step-by-step small-N reproduction and reference numbers |
| [GUIDE](docs/en/GUIDE.md) | code map, plan files, steps, resource switches, tests |
| [METHODS](docs/en/METHODS.md) | model, algorithms, every threshold, seed and parameter, limitations |
| [DATA_LAYOUT](docs/en/DATA_LAYOUT.md) | data root, directory layout, run records, inventory, coupling tuning, cluster jobs |
| [ANALYSIS](docs/en/ANALYSIS.md) | primary/descendant rules A and C, inputs, freezing, blind analysis |
| [AI_GUIDE](docs/en/AI_GUIDE.md) | using and checking the code with an AI coding agent ([AGENTS.md](AGENTS.md) is the agent's own instruction file) |
| [deps/README](deps/README.md) | the Julia environment |

Each has a Chinese version under `docs/zh/`.

## Licence and credits

MIT licence ([LICENSE](LICENSE)); credits in [AUTHORS.md](AUTHORS.md). FuzzifiED is MIT-licensed as well; the one
replaced file and its licence are in `deps/FuzzifiED-2.0.1/`. If you use this template, please also cite FuzzifiED and
the fuzzy-sphere SO(5) DQCP literature on which the model and the tuning criterion are based (for example
arXiv:2306.16435).

---

## 中文简介

**fuzzy sphere 上 SO(5) deconfined quantum critical point（DQCP）模型**的完整、可复用计算模板。对任意 orbital 数 N，计算
有限尺寸 spectrum、KK projection 得到的 primary candidate、OPE 系数和物理 parity（Julia，代码名 *Tcc*），再把各态分为
primary 与 descendant（Python，`analysis/`）。从零重跑小 N 计算所需的一切都在本仓库中；你自己的数据放在你选定的目录里。

- **所有 N 用同一条代码路径。** 物理上唯一与 N 有关的分支是奇偶 N 下 C₂ cross-density 项的符号；只影响速度或内存的选择按问题规模决定，从不按 N。
- **一次计算一个 plan 文件。** sector、spin、能量窗口和后处理步骤都是 TOML 文件的键；每个 step 都从绑定物理输入的 checkpoint 续算。
- **硬门槛。** 残差、Casimir 纯度、窗口完整性和独立检查不通过时 run 直接停止，不会放宽。
- **数据和路径归你自己。** 所有输出写到你设定的数据根目录（`--data`、`TCC_DATA` 或 `user.toml`）；被跟踪的文件中没有任何机器路径或账号。
- **中英双语文档**，覆盖每个公式、门槛和参数。

### 环境要求

| | |
|---|---|
| Julia | 1.11（已用 1.11.7 测试；例如 `juliaup add 1.11`） |
| Python | ≥ 3.10，需 numpy、pandas（`pip install -r analysis/requirements.txt`），用于分析和一个 join 步骤 |
| 机器 | 小 N：8 线程、4 GB 内存、约 300 MB 磁盘（输出与 Julia 包）；首次安装 Julia 包需要联网 |
| 系统 | 下面的复现在 Windows 11 上核对过；本代码的早期版本在 Linux 集群上运行过生产计算（macOS 未测试） |

### 快速上手

```bash
git clone https://github.com/dantengs/fuzzysphere-dqcp-template.git && cd fuzzysphere-dqcp-template
cp user.example.toml user.toml      # 设置 [paths] data_root：所有输出写到这里（默认 ./data，已被 git 忽略）
julia deps/setup_env.jl             # 一次：安装并核对 Julia 环境（约 2 分钟）
julia --project=env --threads=8 bin/tcc.jl configs/quickstart_N5.toml
python tools/check_quickstart.py data/results/quickstart/N5   # 换成你的 data_root；逐项打印 PASS/FAIL
```

第四条命令用 8 线程约 2.5 分钟跑完 N = 5 的完整链（spectrum、KK projection、OPE、三点 insertion、parity、full-Q OPE、
(½,½) 与 (1,1) sector）。

### 复现小 N 结果

[docs/zh/REPRODUCE.md](docs/zh/REPRODUCE.md) 逐步给出完整复现流程和预期数值：

| 步骤 | 复现内容 | 用时（8 线程） |
|---|---|---|
| 1. 环境 | FuzzifiED 2.0.1（替换一个文件）、KrylovKit 0.10.4，逐文件核对 | 1.5 分钟 |
| 2. N = 5 完整链 + `check_quickstart.py` | tuning 后的 coupling、spectrum、KK candidate、OPE、parity、Stage B | 2.5 分钟 |
| 3. 测试 | 核心回归、Stage B 重启、分析单元测试 | 5 分钟 |
| 4. N = 6 主链 + parity | 第 5 步 κ 门槛所用的 pool | 2 分钟 |
| 5. N = 5 的 primary/descendant 分析 | KK 导出、κ 门槛、规则 C、家族检验 | 15 秒 |

更大的 N 用同样的命令和参考 plan `configs/examples/N07.toml … N12.toml`；N ≥ 10 需要集群节点（block 维数见 METHODS §2.4）。

### 文档

| 文档 | 内容 |
|---|---|
| [REPRODUCE](docs/zh/REPRODUCE.md) | 逐步的小 N 复现与参考数值 |
| [GUIDE](docs/zh/GUIDE.md) | 代码结构、plan 文件、step、资源开关、测试 |
| [METHODS](docs/zh/METHODS.md) | 模型、算法、全部门槛、种子与参数、限制 |
| [DATA_LAYOUT](docs/zh/DATA_LAYOUT.md) | 数据根目录、目录布局、run 记录、数据清单、coupling tuning、集群作业 |
| [ANALYSIS](docs/zh/ANALYSIS.md) | primary/descendant 规则 A 与 C、输入、冻结、盲分析 |
| [AI_GUIDE](docs/zh/AI_GUIDE.md) | 借助 AI coding agent 使用和检查代码（[AGENTS.md](AGENTS.md) 是 agent 自己的说明文件） |
| [deps/README](deps/README.md) | Julia 环境 |

### 许可证与署名

MIT 许可证（[LICENSE](LICENSE)）；署名见 [AUTHORS.md](AUTHORS.md)。FuzzifiED 同为 MIT 许可，替换的那一个文件及其许可证在
`deps/FuzzifiED-2.0.1/`。使用本模板时，请同时引用 FuzzifiED 以及模型与 tuning 判据所依据的 fuzzy sphere SO(5) DQCP 文献
（例如 arXiv:2306.16435）。
