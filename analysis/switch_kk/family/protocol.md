# low 态家族检验（规则 C 之后，一个尺寸 N）：冻结规则

本检验在 v3 冻结结果之后进行，不改动 v3 的 primary 判定。它回答两个问题：每个非 primary 态的 \(K\) 像落在哪个 primary 家族、哪一级；以及各家族在各 (级, 自旋, 宇称) 上的占据数是否符合共形表示论的预期。

## 输入

- 输入只用两类数据，均先核对哈希：
  - v3 的 KK 输入，即全部 KK 边、`meta.json` 和 `parity.csv`，经 `run.load` 按 v3 `freeze.json` 核对；
  - v3 的 `results/states.csv` 和 `results/modes.csv`，按 `results/receipt.json` 核对。
- 不读取文献数值、旧 P/D 标签或旧家族复核表。

## 数据性质（冻结前检查，不是结果）

- 加权平方矩阵元对称：\(|\Lambda_{ts}|^2(2L_t+1)=|\Lambda_{st}|^2(2L_s+1)\)。
- 选择定则：X 守恒，SO(5) Casimir 守恒；\(\Delta L=\pm1\) 跃迁保持物理宇称 P，\(\Delta L=0\) 跃迁翻转 P。等能带（角动量分量）除外，违反项的加权平方都小于 \(10^{-6}\)。
- 非块规范的目标窗口（`guarded_lowest_window`）按能量（\(|\delta\Delta|<10^{-6}\)）和 \(C_2\)（\(<10^{-4}\)）唯一匹配到块内原始能级。只用平方矩阵元，因此不需要符号规范。

## 家族

- 每种分类方法 \(m\in\){hard, switch_0.25, switch_0.50}，各取其中央低 \(K^\dagger K\) 子空间的全部模式作为家族源头，即 v3 `modes.csv` 中该方法的集中方向和分散方向（`../modes_readout.py`）。
- 态 \(t\) 对家族 \(F\) 的 0 级权重 \(a_{t,F}\) 等于模式向量在 \(t\) 上分量的平方。primary 权重 \(w_t=\sum_F a_{t,F}\)。
- 守恒流家族：主导原始能级为 J（s0_s1_L1 #1）或 T（s0_s0_L2 #2）的家族，其表示中扣除 null 模。

## 父态分布与递推

- 态 \(t\) 的 \(K\) 像只取向下跃迁（\(\omega<-\)atol），权重为 \(f_m(\omega)^2|\Lambda_{st}|^2(2L_s+1)/(2L_t+1)\)，其中 \(f_m\) 与 v3 相同。这里不计向上的小 \(\omega\) 部分：它在家族意义上不能充当父态。
- 归一化后得到父态分布 \(\pi_t(s)\)。下列父态计入“未解析”：
  - 落在未导出块的父态（目标窗口在最高导出 L 之上，代码中记为 `L4`）；
  - frontier 不足而未进入 v3 分析的父态；
  - 无法匹配的父态；
  - 真空。
- 按 Δ 从低到高递推：
  - \(\rho_t(F,0)=a_{t,F}\)；
  - \(\rho_t(F,n+1)=(1-w_t)\sum_s\pi_t(s)\rho_s(F,n)\)；
  - 未解析权重 \(u_t=(1-w_t)[\pi_t(\text{unres})+\sum_s\pi_t(s)u_s]\)。
- 于是 \(\sum_{F,n}\rho_t+u_t=1\)。这是非相干传播：父态间的干涉忽略不计。

## 逐态标签（每种方法）

GATE = 0.90。

| 标签 | 条件 |
|---|---|
| primary | \(w_t\ge\) GATE |
| unresolved | \(u_t\ge1-\)GATE |
| descendant | 最大的 \(\rho_t(F,n)\)（\(n\ge1\)）\(\ge\) GATE |
| family_mixed_level | 同一家族各级之和 \(\ge\) GATE，但单级不到 |
| mixed | 其余情况，记录前两位 |

robust 指三种方法给出相同的标签、家族和级。

## 计数检验

- 对家族 \(F\)（自旋 \(\ell\)、宇称 \(P_F\)、模式能量 \(\Delta_F\)），第 \(n\) 级自旋 \(j\)、宇称 \(P\) 的预期数为
  \[
  m(n,j,P)=\#\{k\equiv n\ (\mathrm{mod}\ 2),\ 0\le k\le n:\ |k-\ell|\le j\le k+\ell,\ P_F(-1)^{k+\ell-j}=P\}.
  \]
- 守恒流家族扣除 \(m_{\ell-1}(n-1,j,P)\)。
- 只统计 \(j\le3\)、\(\Delta_F+n\le6.25\) 的槽。
- 观测占据 \(O=\sum_t\rho_t(F,n)\)，对该家族所在块 s0_s{λ₂}_L{j} 中宇称为 \(P\) 的态求和。
- 完整槽：\(\Delta_F+n+0.25\le\) 该块首个未进入分析的 Δ（v3 `levels_summary.json` 的 `first_unclassified_delta_by_block`）。其余为边缘槽，只列出，不计入主统计。
- 异常槽：完整槽中 \(|O-m|\ge0.5\)。占据过多提示漏判的 primary 或错误归属，不足提示后裔落到别处。
- 禁止占据：所有 \(m=0\) 的 (F,n,j,P) 上的 O 之和，应接近 0。

## 其他诊断

- **能量**：descendant 标签的 \(\Delta_t-\Delta_F\) 按级汇总。
- **一级范数**：对每个集中 primary 的主导原始能级，hard 拆分下各通道 \(j\) 的
  \[
  N_j=\tfrac{2j+1}{2\ell+1}\Big[\sum_{\omega>\text{atol}}|\Lambda|^2+\text{窗外缺失}\Big]
  \]
  与 CFT 值 \(\tfrac{2j+1}{2\ell+1}\,2(\Delta+\lambda_j)\) 比较，其中 \(\lambda_{\ell+1}=\ell\)，\(\lambda_\ell=-1\)，\(\lambda_{\ell-1}=-(\ell+1)\)。三通道之和即 \(6\Delta\)。

## 失败处理

若递推中出现负权重、\(\sum\rho+u\) 偏离 1 超过 \(10^{-9}\)，或匹配不唯一，则本轮如实报告失败，不调整规则。

若程序在写出任何结果之前因编程错误中断，修复只能针对该错误，不得改动本文件的规则或阈值。首次冻结及其代码另存（例如 `audit/preflight/`），然后重新冻结并在报告中说明。

## 运行

`python test_family.py`；`python family.py freeze <inputs> <threshold_from_raw.json>`；`python family.py run <inputs> <threshold_from_raw.json>`；`python readout.py <inputs> <threshold_from_raw.json>`。需要先完成 v3 的 `run`（`../results/`）。
