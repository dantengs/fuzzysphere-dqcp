# KK 盲分析 v3（规则 C）：有界开关降算符

只对已导出的 KK 矩阵做本机后处理，一次分析一个尺寸 N。输入白名单：

- `inputs/kk/N<N>/`：`meta.json`（原始块、边和定标）与 `edges/*.f64`（每条 KK 边一个二进制文件，`meta.json` 中逐条登记 SHA-256）；
- `inputs/physics/N<N>/parity.csv`；
- 外部预登记阈值文件 `threshold_from_raw.json`，由 `../hardgate/tools/kappa_threshold.py` 从**其他尺寸**的原始 KK 按 `blindpd.thresholds` 的谷值规则得到，不含被分析尺寸的数据，也不含任何 P/D 标签。

历史逐态标签、投影、OPE 表和文献数值都不进入计算。

只分析 s_A = 0 的块 (0,0)、(0,1)、(0,2)：在这些块里 SO(5) 表示由 C₂ 与 λ₂ = s_B 唯一确定（`spectrum.irrep`），s_A > 0 的块只是同一批 multiplet 的副本。导出中其余的块和边被忽略。相邻的 target sector 若为空（Tcc 不为它写 KK 边），导出在 `meta.json` 的 `empty_targets` 中列出（`export_kk.jl` v0.3.6），该方向不计入所需的出边集合。

## 降算符族

生成元按能量变化 \(\omega=\Delta_t-\Delta_s\) 拆分。先令 \(|\omega|\le\mathrm{atol}\) 的等能元为零：这一带对应角动量 \(L^\mu\)，它与 \(D\) 对易，既不属于 \(K\) 也不属于 \(P\)。其余元素取 \(K_{ts}=f_w(\omega)M_{ts}\)、\(P_{ts}=(1-f_w(\omega))M_{ts}\)：

- **hard**（\(w=0\)）：\(f=\mathbf 1_{\omega<0}\)，即 Tcc 的硬切（规则 A 的拆分）。
- **switch_w**（\(w=0.25,0.50\)）：\(f_w=\mathrm{clip}\big((w-\omega)/(2w),0,1\big)\)。\(\omega\le-w\) 全归 \(K\)，\(\omega\ge w\) 全归 \(P\)，中间线性过渡，\(f_w(0)=1/2\)。

\(w=1/2\) 是中点规则：一次跃迁离 \(\pm1\) 比离 0 近时按符号确定归属，更接近 0 的跃迁视为模糊并平滑分配。\(w=1/4\) 是更窄的对照。在严格 CFT 谱（\(\omega=\pm1\)）上三者一致。

**paper_window** 为线性权重 \(f=(1-\omega)/2\)（arXiv:2611.03817 式 (4.2)），用全部已保存目标计算，只作参考，不进入分类；它是全空间值的下界。线性权重无界，有限尺寸下后裔能隙偏离 1 时 primary 自身的 P 跃迁会漏进 \(K^\dagger K\)，这是采用有界开关的原因。

## 覆盖与精确性

源态要求 \(0<\Delta_s\le6.25\)、物理宇称确定，且所有出边 frontier \(>\Delta_s+0.50+\mathrm{atol}\)。窗口外目标都满足 \(\omega>w\)，在三种分类方法下 \(f=0\)，因此 \(K\) 及其 Gram 在窗口内是**精确的**，不是截断近似。

对易子诊断 \(r=(\|P|s\rangle\|^2-\|K|s\rangle\|^2)/6-\Delta_s\) 中，\(\|P\|^2\) 由已保存非等能部分加上逐源态的完整像范数缺口组成，三种分类方法都精确。paper_window 报告窗口内一阶矩，以及“缺口全落在 frontier”给出的下界。

Gram 的组装、跨宇称耦合检查、SO(5) Casimir 分块和 \((2L_t+1)/(2L_s+1)\) 多重度权重与 `../hardgate/blindpd/kk.py` 相同。跨 SO(5) Casimir 的生成元元素相对整条边的最大元素须 \(\le10^{-6}\)（本征态残差经生成元系数放大后的数值噪声量级；与 Tcc 的 KK 门槛同一约定）。N12 原版相对同扇区同 Casimir 的最大元素；当选择定则使该边没有同 Casimir 的 target 时，这个分母为零或本身是噪声，小 N 时会误报，所以模板改用整条边。两种约定在 N12 上给出相同的分类，只有诊断列 `cross_c2_relative` 不同。

## 门槛

\(\kappa=\lambda_{K^\dagger K}/(2\bar\Delta)\)。中央门槛取外部文件的 \(\kappa_c\)，稳定带取同一规则的谷值 bin \([10^{\ell},10^{\ell+0.25}]\)（\(\ell\) 为文件中的 `rule.bin_left`，\(\kappa_c\) 必须是该 bin 的中心）。另报告 0.01 和 0.10。逐态的子空间权重 \(\ge0.90\) 记为支持。

- **robust**：三种分类方法在带内三个门槛（左端、中央、右端）下全部支持。
- **low**：九个组合全部不支持。
- **sensitive**：其余情况，并标出翻转来自门槛（cut_flip）还是方法（method_flip）。

low 只表示缺乏 primary 支持，不等于 descendant 标签；家族归属需要另外的证据（`family/`）。

## 有效性门

J（s0_s1_L1 原始索引 1，定标 \(\Delta=2\)）和 T（s0_s0_L2 原始索引 2，\(\Delta=3\)）已用于定标，不属于盲目标。预登记要求：两者 \(\Delta\) 与 2、3 的偏差在 \(10^{-3}\) 内，状态为 eligible，且判为 robust。任一不满足，本轮方法即判为失败并如实报告，不调整 \(w\) 或门槛。T 在 \(\Delta\approx3\) 是 g 调参的复现，不是独立检验。

## 输出与冻结

1. `python run.py freeze <inputs> <threshold_from_raw.json>` 写 `freeze.json`：代码、本协议、测试和全部输入的 SHA-256（按相对路径记录，便于迁移复跑；只记输入目录名，不记绝对路径）。冻结前先跑 `python test_v3.py`（合成数据，随代码一起冻结）。
2. `python run.py run <inputs> <threshold_from_raw.json>` 核对冻结后写 `results/`：`states.csv`、`modes.csv`（三种方法在中央门槛下重对角化）、`sectors.csv`（维数、秩和主角余弦）、`anchors.json`、`receipt.json`。

冻结后不得修改本协议、代码或阈值；改规则须另建带版本号的目录并重新冻结。若程序在写出任何结果之前因编程错误中断，修复只能针对该错误，首次冻结及其代码另存（例如 `audit/preflight/`），重新冻结并在报告中说明。
