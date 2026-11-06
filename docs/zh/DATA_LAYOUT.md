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

`cluster/tcc.sbatch.example` 运行一个 plan：设置 `TCC_REPO`、`TCC_DATA`、`JULIA`，在你的副本里修改 `#SBATCH` 行（partition、
account、CPU、内存、时间）。内存估计见 METHODS §2.4（block 维数）和 §2.5（residency、Stage B 内存上限 = 分配的 85 %）。
