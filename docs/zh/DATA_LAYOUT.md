# 你的数据：数据根目录、目录布局、记录与数据清单

English: [../en/DATA_LAYOUT.md](../en/DATA_LAYOUT.md)。方法：[METHODS.md](METHODS.md)。学术名词保留英文原词。

仓库里只有代码和文档。你算出的每个数都写到**你的数据根目录**下，所有与机器相关的设置都放在 git 忽略的文件里。被跟踪的文件中
不出现任何机器名、用户名或某台电脑上的目录。

## 1. 选择数据根目录

`bin/tcc.jl` 在读取 plan 之前先进入数据根目录，所以 run 文件里的每个相对路径（`run.out`、各 step 的 `out`、`kk.cache`、
`basis.segment_checkpoint`、`stageb.prelim_top_window` 等）都在它下面。run 文件里的绝对路径按原样使用。数据根目录依次取：

| 来源 | 例子 |
|---|---|
| 命令行 `--data=DIR` | `bin/tcc.jl configs/examples/N08.toml --data=/scratch/me/fuzzysphere` |
| 环境变量 `TCC_DATA` | `export TCC_DATA=/scratch/me/fuzzysphere` |
| `user.toml`（复制自 `user.example.toml`）的 `[paths] data_root`；相对路径相对于仓库根目录 | `data_root = "D:/fuzzysphere"` |
| 当前目录 | — |

run 文件本身按你给出的位置读取。输出的第一行 `TCC_DATA_ROOT …` 显示实际使用的根目录。

保存你的设置、被 git 忽略的文件：`user.toml`、`analysis/paths.local.json`、`cluster/*.sbatch`（复制自
`cluster/tcc.sbatch.example`）。`.gitignore` 还排除了 `data/`、`results/`、`runs/`、`logs/`、checkpoint，以及分析的 `inputs/`、
`results/` 和阈值文件，忘记改相对路径时数据也不会进入提交。

## 2. 参考 plan 使用的布局

```
<数据根目录>/
  kk_cache/                      KK 生成元项的缓存（同一 N 的 run 共用）
  results/main/N<n>/             pipeline run：manifest.toml、run.toml / coverage_run.toml、blocks/、targets/、kk/、ope/、
                                 summary.tsv、physics_report.toml、physics_primaries.csv
  results/ope3/N<n>/             三点 insertion（ope3.toml、*.csv、plan.toml）
  results/parity/N<n>/           parity.csv、parity_kk.csv、parity_summary.toml
  results/fullq/N<n>/            xy_blocks_raw.csv、run_manifest.toml
  results/stageb/half/N<n>/      Stage B：windows/、kk/、edges/、input_binding.toml、stageB_manifest.toml、rawtarget/
  results/stageb/11/N<n>/
  results/quickstart/N5/         quick start
  tests/                         测试输出
  inventory/                     tools/data_inventory.jl 的输出
  analysis/                      分析的导出与结果（ANALYSIS.md）
```

也可以用别的布局；各 step 通过 plan 中的路径找到彼此。

## 3. run 记录了什么

- `manifest.toml`（pipeline）和 `plan.toml`（其他 step）：展开后的 plan、step 选项、代码版本、git commit（`TCC_GIT_COMMIT`，
  否则从 git 读取）和代码树 digest。
- checkpoint 带有其 segment basis 的 SHA-256，并绑定 N、g 和 window；Stage B 的所有 checkpoint 都绑定到 `input_binding.toml`。
- 数据根目录里的记录含有 run 目录的绝对路径，Stage B 作业记录含有主机名和 Slurm 作业号。这些是你自己的记录：留在数据根目录，
  不要复制进仓库。未完成的 run 在完成前保持同一绝对路径（续算会核对记录的 source run）。

## 4. 你的数据清单

`tools/data_inventory.jl` 从上述文本记录生成计算清单（sector、L 范围、window、Δ 截断、KK channel 计数、后处理行数），不加载
checkpoint。

```bash
cp configs/inventory_index.example.toml configs/inventory_index.toml   # 也可以放在数据根目录
julia tools/data_inventory.jl configs/inventory_index.toml <数据根目录>/inventory/DATA_INVENTORY <数据根目录>
```

索引按 N 列出相对于数据根目录的输出目录（`jobs` 为自由文本，可填你自己的作业号）。输出 `DATA_INVENTORY.md`（每个 N 一节）和
`DATA_INVENTORY.csv`。

## 5. Coupling g(N)

每个 plan 以数值给出 g。参考 plan 带有原始研究 tuning 得到的值（F(g) = 0，Δ_J = 2、Δ_T = 3；METHODS §4）。要自己 tuning，设
`g = "tuned"` 并给出 `[tune]` 表（`start`、`bounds`、`maxpoints`；见 `configs/coverage_template.toml`），或运行 `tune` stage；选定值
及其记录写入 run 目录。每个 run 都检查 `physics_report.toml` 中的 `F`。

## 6. 集群

### 6.1 环境

`cluster/environment.example.sh` 描述一套自包含安装，原始研究的作业就是在这样的环境里运行的。环境的组成：
- 一个 Julia depot 存放全部内容：包、二进制 artifact、registry 和编译缓存；
- Julia 离线运行，作业不会读取别的 depot，也不会联网。

使用方法：复制为 `cluster/environment.sh`（git 不跟踪），设置 `TCC_REPO`、`TCC_DATA`、`JULIA`，每个作业开头 source 它。

```bash
cp cluster/environment.example.sh cluster/environment.sh       # 修改路径
source cluster/environment.sh
JULIA_PKG_OFFLINE=false "$JULIA" deps/setup_env.jl               # 一次，需要联网
cp cluster/validate_env.sbatch.example cluster/validate_env.sbatch && sbatch cluster/validate_env.sbatch
```

* **集群不能联网时**，在一台能联网的 Linux x86_64 机器上建好安装（WSL 也可以；二进制 artifact 区分平台），Julia 版本要相同，
  然后把仓库连同 `depot/` 一起拷到集群，命令写在 `environment.example.sh` 开头。`env/Manifest.toml` 用相对路径指向
  `deps/FuzzifiED.jl`，所以拷到任何位置都能用。
* **验证。** `validate_env.sbatch` 在它运行的节点上依次检查三件事：
  - 所有包都从这一个 depot 加载（`LOAD_ONLY_DEPOT_PASS`）；
  - `test/regress.jl 5` 通过（`REGRESS_PASS`）；
  - `test/t9_ope.jl` 通过（`OPE_REFERENCE_PASS`）。

  每种 CPU 跑一次：Julia 按所在 CPU 编译（`-C native`），每种节点第一次运行时会生成自己的编译缓存，约 1 分钟。参数里给出另一套
  环境的回归目录时，还会与它对照谱、C₂ 和边；换环境后，这些量的差异应在 1e-13 量级。
* **Python。** step `parity_join` 和分析代码需要带 numpy（以及 pandas）的 Python。集群 Python 没有这些包、又不能用 pip 时，
  把装有这些包的目录写进 `TCC_PYTHONPATH`（见 `environment.example.sh`）。

### 6.2 作业

`cluster/tcc.sbatch.example` 运行一个 plan 或其中几个 step；在你的副本里修改 `#SBATCH` 行（partition、account、CPU、内存、时间）。
`--set=table.key=value`（TOML 语法）只对这一次作业修改 plan 的一个键，不用改文件。实际使用的 plan 写到 `<数据根目录>/plans/`，
并记在输出里。内存估计见 METHODS §2.4（block 维数）和 §2.5（residency、Stage B 内存上限 = 分配的 85 %）。

### 6.3 最大的 N：一个 plan，多个作业

原始研究的 N12 结果就是这样算出来的，下面用 `configs/examples/N12.toml` 写出。每个作业都从检查点续算，中断或失败的作业重新提交即可。

```bash
P=configs/examples/N12.toml
sbatch cluster/tcc.sbatch $P coverage ope3 parity parity_join fullq     # 主链：谱、KK、OPE、宇称、full-Q
# Stage B（半整数扇区）：原始窗口分到几个作业里算，最高的那个窗口远比其他大
sbatch cluster/tcc.sbatch $P stageb --set=stageb.step=windows --set=stageb.windows=[0,1]
sbatch cluster/tcc.sbatch $P stageb --set=stageb.step=windows --set=stageb.windows=[2]
sbatch cluster/tcc.sbatch $P stageb                                     # 窗口完成后算 KK 与 OPE
sbatch cluster/tcc.sbatch $P rawtarget                                  # 可选：投到完整原始窗口的 OPE
```

* **N12 的规模。**
  - segment basis 约 50 GB，最大的 block 需要数百 GB 内存。
  - 半整数扇区的 L2 窗口（维数 1.4×10⁸，183 个态）在一台 1.5 TB 内存的节点上算了约两周，检查点约 190 GiB。
    其中好几天花在旧版窗口证书最后的补空间检验上；本版本的默认证书改为多求几个态（`[stageb] extra_states`），不再做这项检验。
  - `partial_checkpoints = true`（默认）时，窗口作业每扩一次窗口就存一次盘；`--set=stageb.stop_after_k=k` 让它在存够 k 个态后正常退出。
* **分析用的导出**（`analysis/`，见 docs/zh/ANALYSIS.md §2）要在数据所在的机器上运行：
  - `analysis/hardgate/cluster/export_kk.jl <run.out> <out>`
  - `export_stageB.jl <stageb out> <out>`
  - `export_ope_edges.jl <fullq out> <out>`
