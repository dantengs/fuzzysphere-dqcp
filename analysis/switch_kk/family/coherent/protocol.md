# 相干家族检验（规则 C 之后，一个尺寸 N）：冻结规则

非相干版本（上一级目录 [../protocol.md](../protocol.md)）沿 \(K\) 像向下找父态，只用平方矩阵元。本版本反过来，从 primary 出发用带符号的 \(P\) 像逐级生成后裔子空间，再把每个本征态投影到这些子空间上。父态之间的干涉因此自动计入。两个版本使用同一批输入，方向相反，可以互相核对。

## 输入

- 与非相干版本相同，先核对哈希：
  - v3 的 KK 输入，经 `run.load` 按 v3 `freeze.json` 核对；
  - v3 的 `results/states.csv`、`results/modes.csv`，按 v3 `results/receipt.json` 核对。
- 上游代码 `../family.py`、`../../spectrum.py` 的哈希一并冻结。
- 不读取文献数值、旧 P/D 标签或旧家族复核表，也不读取非相干版本的结果。

## 数据性质（冻结前用探测脚本检查，不是结果）

冻结前应先确认导出数据满足下列性质（只读 KK 输入，检查关闭时构造映射并记录残差）；不满足时不能冻结：

- 两个方向都在块规范内的 ΔL=±1 边对：\(A^{a\to b}_{ts}=-\sqrt{(2L_a+1)/(2L_b+1)}\,B^{b\to a}_{st}\)，相对误差 \(<10^{-12}\)。
- 块规范内的 ΔL=0 边：矩阵对称，相对误差 \(<10^{-12}\)。
- 非块规范窗口（`guarded_lowest_window`）每个目标块只有一套规范（`target_colsha` 相同），它的每一行与反向块规范边对应的列严格成比例（\(|\cos|=1\)），只差一个符号 \(\sigma_t\)；有多个来源的块，各处给出的符号一致。
- 只有非块规范的 ΔL=0 边对齐符号后对称。这一项不参与定符号，是独立核对。
- 所有 ΔL=±1 的块对都至少有一个方向在块规范内。
## 生成元映射

- 记 \(A^{b\to c}\) 为源块 \(b\) 到目标块 \(c\) 的边矩阵（行是 \(c\) 的态，列是 \(b\) 的态）。物理归一的映射为 \(\Phi_{c\leftarrow b}=\sqrt{(2L_c+1)/(2L_b+1)}\,A^{b\to c}\)：对归一多重态 \(x\)，\(\|\Phi x\|^2\) 即该通道的物理跃迁强度。
- 取法：
  - ΔL=±1：若 \(b\to c\) 在块规范内，直接用；否则用反向边 \(\Phi_{c\leftarrow b}=-(B^{c\to b})^{\mathsf T}\)。两者都有时要求 \(\max|\Phi_{\rm direct}-\Phi_{\rm reverse}|/\max|\Phi|\le10^{-9}\)，否则失败。
  - ΔL=0：用自身边。若为非块规范窗口，先对齐行符号。对齐用同一规范下的其他目标为 \(c\) 的边和它们的块规范反向边，按 \(A_{j,:}=\sigma_t\,c_{ab}\,B_{:,t}\) 定 \(\sigma_t\)：
    - 残差 \(\max|A_{j,:}-\sigma_t c_{ab}B_{:,t}|/\max|A|\le10^{-6}\)；两边都小于 \(10^{-9}\max|A|\) 的行不带符号信息，跳过（该行若在 ΔL=0 边上非零则失败）；
    - 多个来源必须一致；
    - 对齐后 \(\max|\Phi-\Phi^{\mathsf T}|/\max|\Phi|\le10^{-8}\)。
    - 任何一条不满足即失败。
  - 目标不在导出块中的边（最高导出 L 之上的目标窗口，代码中记为 `dropped_L4`）舍去。经这些块的路径因此缺失，按秩缺额记录。
- \(P\) 部分：\(\hat P^{(m)}_{c\leftarrow b}=\Phi_{c\leftarrow b}\odot[1-f_m(\omega)]\odot\mathbb 1(|\omega|>\text{atol})\)，其中 \(\omega=\Delta_t-\Delta_s\)，\(f_m\) 与 v3 相同（hard、switch_0.25、switch_0.50）。

## 后裔子空间

- 家族源头是每种方法中央低 \(K^\dagger K\) 子空间的全部模式向量 \(v_F\)（v3 `modes.csv`，带符号）。
- 第 0 级：\(Q_F(0,b_F)=v_F\)。
- 第 \(n\) 级，目标块 \(c\)：
  1. 把所有相邻块 \(b\)（\(|L_b-L_c|\le1\)）的像拼接起来：\(Y=[\hat P_{c\leftarrow b}Q_F(n-1,b)]_b\)。
  2. 限制到家族的 (C₂, X) 扇区、宇称为 \(P\) 的行，做奇异值分解。
  3. 预期秩 \(m=m(n,\ell_F,L_c,P/P_F)\)，守恒流家族扣除 null 模，计数规则同非相干版本。
  4. 取前 \(k=\min(m,\#\{\sigma_i>10^{-8}\})\) 个左奇异向量作为 \(Q_F(n,c,P)\)。\(k<m\) 记为秩缺额。
- 只建 \(j\le3\)、\(\Delta_F+n\le6.25\) 的级。完整槽的定义同非相干版本：\(\Delta_F+n+0.25\le\) 块 \(c\) 首个未进入分析的 Δ。
- 记录的诊断：
  - 各槽奇异值；
  - 冗余比 \(\sigma_{m+1}/\sigma_m\)：多出的路径向量应当近似线性相关；
  - 禁止槽（\(m=0\)）的 \(\sigma_1\) 与同家族同级允许槽最大 \(\sigma_1\) 之比；
  - 子空间的 Rayleigh 能量 \(\langle D\rangle\)。

## 分解

- 扇区为 (块, C₂, X, P)。每个扇区的并集 \(Z\) 包含：
  - 该扇区内所有家族的第 0 级向量；
  - 落在该扇区的全部完整槽基。
- 对 \(Z\) 做 Löwdin 对称正交化：\(\tilde Z=Z O^{-1/2}\)，\(O=Z^{\mathsf T}Z\)。本征值 \(\le10^{-10}\) 的方向舍去，并记录 \(O\) 的最小本征值。
- 本征态 \(t\) 在 (F, n) 上的权重 \(\rho_t(F,n)=\sum_{k\in(F,n)}\tilde Z_{tk}^2\)，覆盖度 \(c_t=\sum_{F,n}\rho_t\)。

## 逐态标签（每种方法）

GATE = 0.90，依次判定：

| 标签 | 条件 |
|---|---|
| primary | \(\sum_F\rho_t(F,0)\ge\) GATE |
| uncovered | \(c_t<\) GATE |
| descendant | 最大的 \(\rho_t(F,n)\)（\(n\ge1\)）\(\ge\) GATE |
| family_mixed_level | 同一家族 \(n\ge1\) 各级之和 \(\ge\) GATE |
| mixed | 其余情况，记录前两位 |

robust 指三种方法给出相同的标签、家族和级。

## 检验

- **覆盖（漏判 primary）**：检验区为扇区内 \(\Delta_t+0.5\le\min(\text{cover}_c-0.25,\,6.25)\) 的态。
  - 记录检验区内 \(c_t\) 的最小值；
  - 在检验区态上计算残差矩阵 \(R=I-\tilde Z_T\tilde Z_T^{\mathsf T}\) 的最大本征值。接近 1 说明有一个方向不属于任何家族，即候选漏判 primary。
- **重复计数（误判 primary）**：\(O\) 的最小本征值，以及不同家族子空间间的最大主角余弦平方。接近 1 说明两个家族生成了同一方向，即某个“primary”其实是后裔。
- **能量**：完整槽 \(\langle D\rangle-\Delta_F\) 按级汇总；descendant 标签态的 \(\Delta_t-\Delta_F\) 按级汇总。
- **代数关系**：完整槽的冗余比、禁止槽比值。CFT 中两者为 0，分别对应 \([P_\mu,P_\nu]=0\) 和守恒流的 null 态。
- 报告阈值：以上比值或本征值 \(\ge0.5\) 记为异常，最小本征值 \(<0.5\) 记为异常。

## 失败处理

出现以下情况即如实报告失败，不调整规则：

- 符号对齐或映射一致性检查不通过；
- 匹配不唯一；
- 权重为负或覆盖度超过 \(1+10^{-9}\)。

若程序在写出任何结果之前因编程错误中断，修复只能针对该错误，不得改动本文件的规则或阈值。首次冻结及其代码另存（例如 `audit/preflight/`），然后重新冻结并在报告中说明。

## 运行

`python test_coherent.py`；`python coherent.py freeze <inputs> <threshold_from_raw.json>`；`python coherent.py run <inputs> <threshold_from_raw.json>`；`python readout.py`。需要先完成 v3 的 `run` 和非相干 `family.py run`。
