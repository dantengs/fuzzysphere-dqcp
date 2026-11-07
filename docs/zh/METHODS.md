# Fuzzy sphere SO(5) DQCP：方法与全部可调参数（Tcc v0.3.1）

English: [../en/METHODS.md](../en/METHODS.md)。代码结构、plan 文件与测试：[GUIDE.md](GUIDE.md)；复现：[REPRODUCE.md](REPRODUCE.md)；数据放在哪里：
[DATA_LAYOUT.md](DATA_LAYOUT.md)。本文与英文版逐节对应，学术名词保留英文原词。

下文每个数值都给出代码默认值，或标明“原始研究”的取值（即本模板所源自的 N = 5–12 计算所用的值），以及设定它的文件
（路径相对于仓库根目录）。参考 plan `configs/examples/N05.toml … N12.toml` 重述了这些计算。

---

## 0. 软件

| 项 | 值 |
|---|---|
| Julia | 1.11.7 |
| FuzzifiED | 2.0.1（registry 版本），**替换了一个文件** `src/so3lver/core/comp_operator.jl`（多线程组装、带 `num_th`、无锁 H·v；算符相同，只是求和顺序不同）；FuzzifiED_jll 1.3.0+0。由 `deps/setup_env.jl` 构建，见 `deps/README.md` |
| KrylovKit | 0.10.4（registry 版本；原始研究所用副本多一个 Tcc 不调用的 `LockedLanczos`） |
| 其他 | WignerSymbols 2.0.0、SpecialFunctions 2.9.0（以 `Manifest.toml` 为准） |
| 浮点 | `FuzzifiED.ElementType = Float64`；Hamiltonian 实对称，所有 eigenvector 为实向量 |
| 线程 | `OMP_NUM_THREADS=1`；Julia 线程、BLAS 线程、分块宽度、residency 只影响速度 |

---

## 1. 模型

### 1.1 Landau level 与 orbital
- 球面穿过 2s 个 flux quantum，取 lowest Landau level，s = (N − 1)/2；每个 flavour 有 N 个 orbital，m = −s…s。
- 四个 flavour，half filling，N_e = 2N。
- flavour 分成两个 **segment**：A = (f₁, f₂)，B = (f₃, f₄)；segment 内 pairing 为 Δ_seg = c_{f₁}c_{f₂}（矩阵 `[0 1; 0 0]`）。
  每个 segment 有 SU(2) flavour symmetry，SU(2)_A × SU(2)_B ⊂ Sp(2) ≅ Spin(5)。
- 所有算符写成"segment 内项 + segment 间耦合"（FuzzifiED `SO3lver`，2+2 segment）。

### 1.2 Hamiltonian
H(g) = H_den − g · H_pair（`src/model.jl: build_terms, set_g!`），两项都只含 V₀ pseudopotential（V₀ = 1）：
- **H_den**：segment 内 `SingleSegCouple(2, p, GetDenIntTerms(N, 2), [0,0,0])`（p = A、B），加 segment 间
  `2 · CoupleDecomps([n, n], ConvPsPot(RecoupleAngMom(s, [1]))…)`，n = `GetDensityMod(N, 2, I₂)`；合起来是四 flavour 总密度的
  V₀ contact interaction。
- **H_pair**：segment 内 `GetPairIntTerms(N, 2, [0 1; 0 0])`，加 segment 间 `CoupleDecomps([Δ′, Δ], ConvPsPot(s, [1])…,
  [2 −2; 0 0; 0 0])` 及其反序项，Δ = `GetPairingMod(N, 2, [0 1; 0 0])`；合起来是 (Δ_A + Δ_B)†(Δ_A + Δ_B) 的 V₀ channel。
- 与 g 无关的模板 `nn + dd` 只组装一次；`set_g!` 只改写 channel 系数（density channel 不变，pairing channel 乘 −g）；
  所有 channel 系数必须为实数（虚部 < 1e-11）。
- U_{m₁m₂m₃m₄} = Σ_l V_l (2N − 2l + 1)(s s N−l; m₁ m₂ −m₁−m₂)(s s N−l; m₄ m₃ −m₃−m₄)（FuzzifiED `GetIntMatrix`）。

### 1.3 SO(5) Casimir
`src/model.jl: build_terms`。segment 内 `GetC2Terms(N, 2, :SU) + (n_f − 2)/4 · GetPolTerms(N, 2)`（n_f = 4）；segment 间
density 耦合对所有 rank l = 0…2s 乘 `cross_density_coefficient(N)`：奇数 N 为 −½，偶数 N 为 +½（`src/so5_c2_density_phase.jl`；
这个符号抵消钉住版本 `GetDensityMod` 给 mode 附加的 (i)^{2m} 相位）；segment 间 pairing 耦合对偶数 l 取 −1。
归一化 C₂(5) = 2、C₂(10) = 3、C₂(14) = 5。irrep 识别（Dynkin label ↔ C₂）见 `src/kkstage.jl: so5_types`。

**这个奇偶相位是代码中物理上唯一随 N 变化的分支。**

### 1.4 Coupling g(N)
每个 N 由 finite-size 条件 F(g) = 0 固定（Δ_J = 2 时 Δ_T = 3），见 §4；N = 5–12 的参考值在 `configs/examples/`。
这是有限 N 的匹配，不是 critical point；代码输出中也这样声明。

---

## 2. Hilbert space 与对称 sector

### 2.1 Segment space
`src/model.jl: segment_setup, build_segment_spaces`，底层是 FuzzifiED `BuildSegSpaces`。
- 每个 segment 的量子数：粒子数 n_e（0…2N）、2L_z（`GetLz2QNDiag(N, 2)`）、flavour S^z = 0（`GetFlavQNDiag(N, 2, [1, −1])`）；
  每个 n_e 只保留 |L_z| 最低的 sector，`sec_pt = [n_e, ((N+1)n_e) % 2, 0]`。
- multiplet：联合对角化 L²（`GetLzLpTerms`）与 segment SU(2) Casimir（`GetC2Terms(N, 2, :SU)`），`l2c2_ratio = √2`；
  segment spin S_seg ∈ {0, 1, 2}（`c2_rng = S(S+1)`）。
- **gauge**：简并 multiplet 内的基由 `eigen()` 任意给出，两次构建不同；eigenvector 只在产生它的那份 basis 中有意义。每个
  `eig.ckpt` 记录所用 segment basis 的 SHA-256，后续每一步都核对。
- **particle–hole 半量构建**（`src/ph.jl`，`basis.ph_half`）：只对角化 n_e ≤ N，n_e > N 由
  P c†_{m,f}P⁻¹ = (−1)^{s−m} ε_f c_{−m,f̄} 生成。与直接构建张成相同的 multiplet 子空间（只差 multiplet gauge，故 SHA 不同）；
  N4–N12：|U′U − I| ≤ 1.3e-12，spectrum 差 ≤ 2.7e-13。默认 "auto" 见 §2.5。

### 2.2 复合 block (s_A, s_B, L)
`BuildCompSpace([sp_A, sp_B], [2N, 0, 0], 2L)`：总粒子数 2N、总 flavour S^z = 0、总角动量 L 的 highest weight。block 记作
`s{a}_s{b}_L{L}`，a ≤ b 为两个 segment 的 SU(2) spin；每个 SO(5) multiplet 在若干 (s_A, s_B) block 中都有副本（N12：(1,1) 与
(0,x) 副本 |ΔE| ≤ 6.9e-9，parity 184/184 一致）。**integer sector** 是六个 pair (a, b)，0 ≤ a ≤ b ≤ 2；**half-integer sector
(½,½)** 每个 segment 的 n_e 为奇数、segment spin ½、Q = [2N, 0, 2]（用于 pipeline OPE、ope3 与 Stage B，§9–§11）。

### 2.3 Segment exchange X
`src/exchange.jl`。对角 block (s, s, L) 上的 segment exchange（FuzzifiED `PermFirstSecondSegs`，带 SO(3) recoupling 符号；
每个 segment 粒子数为偶数，故 fermion 符号为 +1）经检查是带符号置换且 X² = 1。X = ±1 在长度约一半的约化空间中分别求解，
e_k = (|i⟩ + c_k|j⟩)/√2（不动点取 |i⟩）。非对角 block 记 X = 0。

### 2.4 Block 维数（Lanczos 维数）

| N | (0,0) L0，X=+1 / −1 | (0,1) L1 | 最大 block | block 数 |
|---|---|---|---|---|
| 5 | 18 / 18 | 62 | (0,1) L3：96 | 30 |
| 6 | 71 / 52 | 315 | (0,1) L4：598 | 30 |
| 7 | 293 / 293 | 1 898 | (0,1) L4：4 280 | 30 |
| 8 | 1 716 / 1 582 | 12 762 | (0,1) L4：31 893 | 30 |
| 9 | 10 964 / 10 964 | 94 320 | (0,1) L4：249 340 | 30 |
| 10 | 80 990 / 79 926 | 748 172 | (0,1) L4：2 049 387 | 30 |
| 11 | 640 086 / 640 086 | 6 295 212 | (0,1) L4：17 636 814 | 30 |
| 12 | 5 408 157 / 5 398 483 | 55 612 834 | (0,1) L3：126 012 436 | 12 |

### 2.5 资源开关（v0.3.0：按规模选择，从不按 N）
`src/resources.jl`。都不改变任何物理数值；每一对算法都已互相验证过。

| 开关 | "auto" 规则 | 重现的原始研究选择 |
|---|---|---|
| `basis.ph_half` | 最大的 segment sector（BuildSegSpaces 做稠密对角化的那个）≥ 2000 个 configuration 时用 PH 半量构建 | N9：1070 → 直接；N10：3656、N11：10 762、N12：37 834 → PH（与原来的"N ≥ 10"相同） |
| `spectrum.residency` | 存储的 segment basis 超过 16e9 字节时用 `pair`（最多 2 个 segment space 常驻），否则 `all` | N11 4.6e9 → all，N12 4.9e10 → pair |
| `threads.native_dim_max` | 逐 block：维数 2×10⁷ 以下用 FuzzifiED 自带的 H 作用，以上用 Tcc 的列分块作用 | 不变 |
| `stageb.memory_limit_bytes` | Slurm 分配内存的 85 %；会超出的求解步在分配前被拒绝 | 与原始研究每个 Stage B 作业的设置相同 |

都可以强制指定（`true`/`false`、`all`/`pair`/`minimal`、具体数值）。

---

## 3. 对角化

### 3.1 求解器（`src/runner.jl: lowest`）
- n ≤ 400：稠密，逐列作用得到矩阵，`eigen(Symmetric((M + M′)/2))`。
- 否则 KrylovKit Lanczos，`eigsolve(f, x₀, k, :SR, Lanczos(krylovdim, tol, maxiter=500))`，x₀ = normalize(randn(MersenneTwister(seed), n))；
  k ≤ 90 时 Krylov 维数 min(max(2k, k+10), 100)，否则 k + max(30, ⌊k/2⌋)；`spectrum.tol` = 1e-10；只保留最低的 k 个收敛对。
- 固定种子：block 1000a + 100b + 10L + (σ+1)（σ 为 X sector；deflation 扩展 + 7919）；target window 71231 + 100a + 10b + l + 7(σ+1)
  （deflation + 104729）；target 独立复核 99871 + 100a + 10b + l（+13σ）；tune 5052026 + 100s_A + 10s_B + l（最终求解 + 7919）；
  pipeline half sector 424242 + k；Stage B 88000 + 100L + k，complement check 77100 + L（prior）与 99000 + L（最终）。
- **可重现性**：种子固定时结果在舍入误差内可重现；多线程 BLAS 使最后几位、进而 eigenvector 的任意符号在进程间不同（matrix
  element 可能变号；绝对值、能量、C₂、parity 一致到 ~1e-13）。比较不同 run 时比较绝对值。

### 3.2 Window 与 coverage
- 每个 block、每个 X sector 求最低 `nev` 个态（`block_nev`、`parity_nev` 按 block / sector 覆盖）。
- **coverage step**（`src/post.jl: run_coverage`，`[coverage] delta_cut, kmax`）：第一次求解后，凡 window 不是整个 sector 且顶端
  低于 Δ_cut = 6.0 的 sector，扩大到 k = min(dim, k_max, max(n + 8, ⌈1.6n⌉))，k_max = 400，`spectrum.extend = true`（已存向量逐位
  保留，只用 deflation 求新态），直到全部到 Δ ≥ 6.0 或已完整（最多 50 轮）。原始研究：N5、N6 各一轮，其余为零轮。
- **deflation**（`lowest_deflated`）：g = QHQ + σP，P = V₀V₀′，σ = 10 max|E_ref| + 100；新态对旧空间正交化、QR、Rayleigh–Ritz；
  若新态低于旧窗口顶端 − 1e-8 则报错（说明旧窗口不是完整的最低前缀）。

### 3.3 C₂ polishing（`src/targets.jl: polish_c2!`）
C₂ 残差超过触发值（block 1e-6，target 1e-7）的列，与同一 X sector 内能量差 1e-3 以内的所有态一起在 C₂ 中重新对角化；C₂ 相同
（≤ 1e-6）的组再在 H 中做 Rayleigh–Ritz。冻结列从不改动。target window 先在 C₂ 中分开数值简并簇（≤ 1e-7，`resolve_c2!`）。

### 3.4 验收门槛（`src/gates.jl`，硬性，从不放宽）
‖Hv − Ev‖、‖Cv − c₂v‖、列范数误差、‖V′V − 1‖_F ≤ 1e-6；C₂ 与整数之差 ≤ 1e-5；能量有限且升序；target fresh 独立复核
|ΔE| ≤ 1e-7、子空间最小奇异值 ≥ 1 − 1e-5。每一列带 SHA-256，作为增量复用中的身份。实际残差远低于门槛（N12 的 φ、J、T：~1e-13）。

---

## 4. Coupling tuning（`src/tuning.jl`）
F(g) = 2(E_T − E₀)/(E_J − E₀) − 3，E₀ 为 (0,0) L0 中 C₂ = 0 的最低态（X = +1），E_J 为 (0,1) L1 中 C₂ = 3 的最低态，E_T 为 (0,0) L2
中 C₂ = 0 的最低态。每个 g 点用 k = 8、12、24、tol 1e-9 求三个锚点态（门槛：残差 ≤ 1e-7，目标 C₂ 偏差 ≤ 1e-5，目标下方所有态的
C₂ 已分辨，窗口末端能隙 > 1e-6 或已是整个 sector）。求根（`find_root`）：从 `start` 出发，依次取 ±0.005、±0.015、±0.035、
±0.075 和 `bounds` = [0.90, 1.12] 两端，直到 F 变号；在括号内做带保护的割线法（退化时二分）；|F| ≤ ftol = 1e-5 但括号宽于
xtol = 1e-4 时，在最优点两侧 ±0.49 xtol 各取一点。接受条件：与最近另一 g 点的锚点重叠 ≥ 0.95；换种子的独立最终求解仍有
|F| ≤ ftol，能量一致到 1e-7。起点只决定从哪一侧逼近括号。原始研究的参考值见 `configs/examples/N*.toml` 的 `run.g`。判据只用模型自身的谱（条件的形式沿用 arXiv:2306.16435），
不声称 g* 是临界点。

## 5. 能量标定
Δ = α(E − E₀)，α = 2/(E_J − E₀)（Δ_J = 2；在 g* 处 Δ_T ≈ 3）。summary 与 KK fit 用同一个 α，各自独立算出。没有 finite-size 修正，
没有外推。

## 6. KK target window（`src/targets.jl: extend_targets`）
每个 source block L_i 的 KK 步骤需要 L_f ∈ {L_i − 1, L_i, L_i + 1}（L_i = 0 时只要 L_f = 1）的完整最低窗口：即 L = 0…source_lmax + 1
的 target window，source_lmax = max(`spectrum.lmax`, block 的最大 L)。
- 所需能量：相邻 source 的最高能量，截到 E₀ + `source_delta_max`/α（原始研究 6.2；高于它的 source 记为 `partial_source_coverage`）。
- frontier：整个 sector 为 +∞；最后一个能隙大于 `guard`（1e-5）时为窗口顶；落在简并簇中为 −∞。frontier > need + guard 即完整。
  已满足此条件且过门槛的 run 窗口直接复用。
- 从头扩窗（`target_growth = "fresh"`）：首个 k 由本 block 的谱估计（×1.15 + 8，或按 ∝ (E − E₁)^{1.5} 外推，×1.1 + 8）；失败后
  k ← max(k + 8, ⌈1.5k⌉, 外推值)，上限 `max_nev`；到上限仍不完整则报 `INCOMPLETE_TARGET_WINDOW`（从不静默截断）。
- 独立复核（默认 `complement`）：窗口正交补中最低 2 个态（换种子）；margin < −1e-7 为 `GATE_FAIL`，−1e-7…guard 则扩窗。
- 原始研究：`initial_nev` = run 的 nev，`max_nev` 4000（N5–11）、400（N12），`target_parity = "both"`。

## 7. KK projection（primary candidate；`src/kkstage.jl`、`src/ZhouGenerator.jl`）
- generator 候选：O₁ = (n†n) 与 O₂ = (Δ†Δ) 的 l = 1 分量（`GetDensityObs`、`GetPairingObs(…, mom_incr=false)`，`ObsNormRadSq = N`；
  pairing 取 m → −m 并补 i^{2m}；segment 间耦合乘 1/(N√(4π))，density 再乘 2，pairing 取两种序；奇数 rank 时 O₂ 取负号）；
  K = c₁O₁ + c₂O₂，只取 lowering 部分。
- **vacuum fit**（`kk_calibration!`、`fit_vacuum_generator`）：参考态 v（vacuum）、s（(0,0) L0 的第二个 C₂ = 0 态）、∂S（L1 中 C₂ = 0、
  能量最接近 E_S + 1/α 的态）；在约束 Σ_p c_p⟨∂S|O_p|S⟩ = √(2Δ_S) 下最小化 vacuum leakage c′Gc（G 的本征分解截断 rtol 1e-11）。
- **edge**：overlap T′O_pS 与 image Gram ⟨O_pS_j, O_qS_j⟩；generator 不得改变 C₂（≤ 1e-6‖M‖）；energy_atol = max(1e-8, 10(r_S + r_T))。
- **guarded source**：所需 L_f 都存在，且对每条 edge 有 E_i < frontier_f − energy_atol。
- 每个整数 C₂ channel：K² = Σ_{L_f} (2L_f + 1)/(2L_i + 1) K′K（只含 lowering 元素）；检查 captured ≤ image norm（1e-7）、K² ≥ −1e-8；
  K² ≤ threshold 的本征子空间在能量中对角化 → projected mode（能量、混合残差）。
- threshold [0.05, 0.1, 0.25, 0.5]；报告与 OPE basis 用 0.25。channel 状态为 `finite_saved_window_complete_K_action`、
  `partial_source_coverage` 或 `incomplete`；从不标 `CFT_primary_certified`。

## 8. Pipeline OPE（`src/ope.jl`）
Γ² insertion n_l = n_A − n_B（`GetDensityObs(N, 2, I₂; norm_r2 = N)`、`FilterL2`），edge [[0,0,0],[2,0,2]] 在 (0,0) 上；
n = |⟨φ|Γ²₀|0⟩|；f_φφS、f_φφT14（×√5/2）、f_φφstress（×√(15/8)）分别在 raw、L0 projected、L0+L2 projected basis 上计算；φ×φ singlet
与 T14 tower。half sector (½,½)（`ope.half`）：L0（C₂ 2、5）与 L1（C₂ 3）窗口，k = 6、10、16、24，f_φφJ = √3|⟨J|Γ²₁|φ⟩|/n，
f_φφT_half = √2|⟨φ|Γ²₀|T⟩|/n，带号共轭门槛 ≤ 1e-7。

## 9. Three-point insertion 与扩展 OPE（`src/ope3.jl`，step `ope3`）
选项：`src_dmax` 5.0、`pair_dmax` 6.5、`int_ranks` [1,2,3,4]、`half_lmax` 3、`half_nev` 24、`threshold` 0.25、`image_bytes`（只影响速度）、
`qg_gate`（N ≤ 7）、`half_only`。
insertion：任意 rank 的 Γ²；singlet O_S = b_ρ(ρ†ρ) + b_Δ(Δ†Δ)（rank 0 与 2，由 ⟨stress|O_S^{(2)}|0⟩ = 0 与 ⟨S|O_S|0⟩ = Z_S = ±1 确定，
S 的符号由 ⟨φ|n₀|S⟩ > 0 固定）；tensor Q_G，G = (−Q₁₁ + 4Q₂₂ − Q₃₃ − Q₄₄ − Q₅₅)/√20，精确 Fierz 形式（`ObsNormRadSq = 1`），
Z_T = |⟨T|Q_G|0⟩|。φ×φ → spin l：λ = c_l · CG · |⟨O_l|n_l|φ⟩|/|⟨φ|n₀|0⟩|，c_l = √((2l+1)!!/l!)。门槛：hermiticity < 1e-7，⟨0|Q_G|0⟩ 与
⟨S|Q_G|S⟩ 为零（< 1e-8 Z_T），half sector selection rule。half sector edge 绑定 window checkpoint SHA 与 g（v0.2.8，审计 A3）。

## 10. Physical parity P（`src/parity.jl`，step `parity`）
P 是 arXiv:2306.16435 的 particle–hole 变换 c → Jc†（i → −i）；在 2+2 segment 下等于两个 segment 内 PH 映射 P_seg 之积（multiplet
|n_e, l, α⟩ → |2N − n_e, l, β⟩，正交矩阵 U^{(l)} 由存储的代表分量算出，误差 ≤ 1e-8）。P_phys = ⟨v|P_seg|v⟩ · ⟨vac|P_seg|vac⟩ · (−1)^L
（P_seg 含 m → −m）。每个存储的态必须是纯 P_seg 本征态（残差 < 1e-6）。KK mode：⟨P⟩ = Σ q_i² P_i（||⟨P⟩| − 1| ≥ 1e-6 记为 `mixed`）。
φφ tower 的 P 由 `tools/ope3_parity_join.py` 给出（step `parity_join`）。N10 重现 2306.16435 Table VI 全部 129 行的 parity。

## 11. Stage B：(½,½) 与 (1,1) sector 算到 raw Δ 截断（`src/stageb.jl`，step `stageb`）
v0.3.0 由三个早期按 N 写的脚本合并而来（小 N 的 half 与 (1,1)；spin 0–3 的 half；最大 N 的省内存版本，带并行 window 作业）。
选项在 `[stageb]` 中。
1. **raw window** L = 0…lmax + 1（`sector = "half"`：segment spin ½，Q = [2N, 0, 2]，basis 取自 ope3；`"11"`：run 的 (1,1) basis），
   每个 L 逐步扩大（24 → ×1.5，上限 `max_nev` = 384），直到 frontier 高于 E₀ + `raw_delta_max`/α（6.8）且 complement check
   （2 个态，margin > guard）通过；已有窗口（ope3 half window、run 的 (1,1) window）只在过门槛且是完整前缀时作为 prior。分开并 polish
   C₂ 簇；简并对旋转为 P_seg 本征态（`ph_resolve!`）；门槛同 §3.4。内存守护拒绝会超出 `memory_limit_bytes` 的步骤。
   `step = "windows"` 加 `windows = [L…]` 可让多个作业把不同 L 解到同一目录。
2. **控制量**：vacuum 的 P_seg 符号用最低的 C₂ = 2（half）/ C₂ = 5（(1,1)）态，对照同能量的 integer (0,0) L0 态锚定（≤ 1e-6）；
   half sector 中 current J 必须有 Δ = 2（≤ 1e-6）且 P = +1，raw φ–J 矩阵元须重现 ope3 的值（相对 ≤ 1e-7）。
3. **KK**：E ≤ E₀ + raw_delta_max/α 且 C₂ ∈ {0, 2, 3, 5, 6, 8, 9, 10, 14} 的 source，lowering edge 到 L − 1、L、L + 1，按 P_seg sector
   分类，用 **integer run 的 generator 系数**（§7），threshold 0.25。
4. **full-Q OPE**：projected scalar source（Δ ≤ `source_delta_max` 5.0）× projected spin-l target（Δ ≤ `target_delta_max` 6.6），
   l = 0…lmax，insertion Γ²、Q_G（各两部分，后续再组合）与 O_S；输出 `xy_blocks_raw.csv`、`projected_catalog.csv`、`kk_coverage.csv`、
   `raw_spectrum_Lle{lmax+1}.csv`。
5. **raw target**（`rawtarget = true` 或 step `rawtarget`）：同一组 source 投到完整 raw target window；把它投影到 target mode 上必须
   重现第 4 步的 edge 与 image Gram（相对 ≤ 1e-10）。
身份：所有 checkpoint 绑定 `input_binding.toml`（N、g、sector、raw Δ 截断、calibration/normalisation/segment SHA）；续算复用身份一致的
每个 window、edge、projection（审计 A2）。

**v0.3.1 新增**（种子 88000 + 100L + k、各门槛和 Hamiltonian 都不变）：
6. **window certificate**（`extra_states` = 4）。每个扩窗步多求 `extra_states` 个态，即共求 k + `extra_states` 个态。
   存盘的 window 截到第一个满足以下两条的下标 j ≥ k：E_j − E_{j−1} > guard，且 E_{j+1} − E_j > guard。能隙
   E_{j+1} − E_j 就是 complement margin，与第 1 步 deflated complement solve 算的是同一个量（后者的最低态就是第 j + 1 个态）。
   多求的态在存盘前去掉，所以 window 的态数与 v0.3.0 相同。这时 prior window 不再单独做 complement check：第一次扩窗以它
   为 deflation basis 求新态，如有新态低于它的顶端就报错停下。仍需做的 complement check（`extra_states = 0`，或某一步没有新
   求解）用收敛标准 `complement_tol`（1e-6，上限 guard/10；Ritz value 的误差不超过残差范数）。PH 分解前，对 certified
   window 再用触发线 `final_polish_trigger`（1e-9）做一次 C₂ polish。没有这一步时，N8 L2 一对能隙 1.7e-6、C₂ 分别为 10 和 2
   的态，留下了随 Lanczos 路径而变的 C₂ 残差 2.7e-8。window checkpoint 记录 `certificate`（`extra_states` /
   `deflated_complement` / `whole_sector`）和 `complement_tol`，manifest 按 L 列出。
   原因：N12 L2（D = 1.4×10⁸）上，对 81 态 window 做收敛标准 1e-10 的 complement solve，29 h 以上仍未结束；而观测到的
   margin 都 ≥ 3.3e-4（guard 的 33 倍）。`extra_states = 0`、`complement_tol = 1e-10`、`final_polish_trigger = 0` 时与
   v0.3.0 完全相同（window 相同；制作 v0.3.1 时在 N6 上核对过）。
7. **partial checkpoint**（`partial_checkpoints = true`）。每个还没盖过截断的扩窗步，都把带入下一步的 window 存到
   `windows/partial/L<L>_k<k>.ckpt`；新的写完后才删除旧的。重启时从 k 最大的 partial 续算，下一个 k 和种子都与不中断时
   相同，走的是同一个步骤序列。`stop_after_k = k`（只用于 `step = "windows"`）在存下态数 ≥ k 的 partial 后干净停止。
   window 存盘后删除 partial。代价：每个 L 磁盘上最多同时有两份 partial（N12 L2 81 态约 91 GB）。
8. **preliminary top window**（`prelim_top_window = <partial checkpoint>`，`step = "all"`，lmax ≥ 1）。用一份 partial
   window（身份检查与续算相同）充当 L = lmax + 1 的 raw window，它只作 KK target。各 L 的 KK source 限于 E < min over
   targets (frontier − atol)，也就是 KK 步骤原本断言的 guard；去掉的 source 记入 `prelim_truncated_sources.csv`。partial
   window 没有 PH 标签（`raw_spectrum` 中 pseg = P = 0）。manifest 的 status 为
   `preliminary_partial_top_window_truncated_KK_sources`，并有一节 `[prelim]`。`prelim_complement = true` 在导出后对 partial
   window 做 complement check（`prelim_complement_L<L>.toml`）；不通过时报错退出，但已写的输出保留。例：L2 window 未完成时，
   可先得到 L0–1 的 half sector KK/OPE；φφJ 与 L2 截断无关，因为 K₂(J) = 0（J 是 L0–L2 中最低的 C₂ = 3 态）。这类初步输出
   应与完整 window 的结果分开。

## 12. Integer sector 的 full-Q OPE（`src/fullq.jl`，step `fullq`）
v0.3.0 由两个早期脚本合并而来（全部 sector；最大 N 的 sector 列表）。对每个选定的 (s_A, s_B) pair：KK scalar
source（Δ ≤ 5.0）用其**完整**保存的系数向量（无 support 阈值；独立 axpy 探针检查收缩 ≤ 1e-10），对 rank l = 0…`rank_max` 的 KK target
（Δ ≤ 6.6）；每个 pair 做 singlet insertion，对角 pair 另做 Γ² 与 Q_G；parity 从 `parity_kk.csv` 读入（每个 mode 都必须有，|ΔΔ| ≤ 1e-7）。
`sectors` 默认：顺序 (0,1)、(0,2)、(1,2)、(0,0)、(1,1)、(2,2) 中 run 实际有的 pair。

## 13. 全部可调参数

类别：**物理**（定义模型/sector）、**精度**（门槛内的误差大小）、**窗口**（算到多高的能量，因而决定覆盖与 channel 状态）、
**性能**（只影响速度/内存；结果在舍入内相同）。

| TOML 键 / 位置 | 默认 | 原始研究（`configs/examples/`） | 类别 |
|---|---|---|---|
| run.N、run.g | — | §4、configs/examples | 物理 |
| pseudopotential、segment spin {0,1,2}、C₂ cross-density 符号 | 代码常数 | 同左 | 物理 |
| plan.sectors / plan.lmax → spectrum.blocks | 六个 pair × L0–2 | configs/examples | 窗口 |
| spectrum.lmax | 2 | 3（run 身份中的旧键） | 物理键 |
| spectrum.exchange、parity | true、both | 同左 | 物理 / 窗口 |
| spectrum.nev、block_nev、parity_nev、extend | 12、—、—、false | configs/examples | 窗口 |
| spectrum.tol；Lanczos maxiter、Krylov 规则、稠密阈值 400、种子 | 1e-10；代码 | 同左 | 精度 |
| coverage.delta_cut、kmax；增长 ×1.6 / +8 | —、3000 | 6.0、400 | 窗口 |
| deflation σ、polish window/trigger、resolve_c2 容差、GATE | 代码常数 | 同左 | 精度 / 验收 |
| tune.start、bounds、maxpoints、ftol、xtol、exchange | §4 | §4 | 物理（决定 g） |
| basis.segment_checkpoint、basis.ph_half | ""、auto | ""、auto | 只影响 gauge |
| targets.initial_nev、max_nev、guard、source_delta_max（= plan.delta_max） | 12、100、1e-5、无 | nev、4000/400、1e-5、6.2 | 窗口 |
| targets.target_growth、first_k、independent_check、target_parity | fresh、estimate、complement、both | 同左 | 性能 / 验收 |
| kk.thresholds、report_threshold、pairs、lmax、legacy_calibration | [0.05,0.1,0.25,0.5]、0.25、全部、source_lmax、无 | 同左 | 判定 / 窗口 |
| ope.edges、half（= plan.half）、threshold、reference | [[0,0,0],[2,0,2]]、false、0.25、无 | 默认、true | 窗口 |
| ope3.* | §9 | §9 | 窗口 / 判定 |
| parity.targets、vacuum_raw | true（step）、取自 (0,0) L0 | 同左 | 约定 |
| fullq.sectors、rank_max、source/target_delta_max、threshold | §12、4、5.0/6.6、0.25 | N12：(0,1),(0,2),(0,0)、3 | 窗口 |
| stageb.sector(s)、lmax、raw_delta_max、source/target_delta_max、max_nev、threshold、insertions | half、4、6.8、5.0/6.6、384、0.25、三种 | N11：lmax 3；N12：lmax 1 | 窗口 |
| stageb.step、windows、memory_limit_bytes、post_chunk、image_bytes | all、all、Slurm 85 %、8、4e9 | 按作业 | 性能 |
| stageb.extra_states、complement_tol、final_polish_trigger（v0.3.1） | 4、1e-6、1e-9 | 0、1e-10、关（v0.3.0 的证书） | 精度 / 验收 |
| stageb.partial_checkpoints、stop_after_k（v0.3.1） | true、0 | N12 L2：长 window 作业的 partial checkpoint | 性能 |
| stageb.prelim_top_window、prelim_complement（v0.3.1） | ""、false | N12：由部分 L2 window 得到初步 KK/OPE | 窗口（初步） |
| threads.*、spectrum.residency、concurrency、post_bytes、async_io、kk.cache、ope3.image_bytes | §2.5 | auto | 性能 |

## 14. 读者必须知道的限制
1. **finite-size 匹配**：g(N) 由 Δ_T = 3 匹配，Δ 以 Δ_J = 2 定标，没有 1/N 修正；所有 OPE 数值都是有限 N 的原始估计。
2. **N12 参考 plan**：只有 (0,x) block 的 L ≤ 3（KK source 到 L3，target 到 L4）；没有 (1,1)、(1,2)、(2,2) 的 KK；(0,0) L3 的 C₂ = 14
   channel 没有 guarded source；Stage B half 只到 raw L2 / KK L1。完整的 N12 L2 half window（D = 1.4×10⁸，约 180 态）单节点要算数天；
   §11 第 8 项给出初步路线。
3. **复用 segment basis**：其他代码版本建的 `segment_spaces.ckpt` 可以复用（`basis.segment_checkpoint`），前提是与当前代码新建的 basis
   张成相同的 multiplet 子空间；原始研究的 N11、N12 basis 已直接核对（‖UᵀU − I‖ = 3.1e-14，‖B − AU‖ = 1.4e-12）。要逐位重现
   eigenvector 需用同一份 basis 文件；物理数值不需要。
4. **g 的来源**：`configs/examples/` 中的 g 由同一算法的早期版本选定，以数值写入 plan；每个 `physics_report.toml` 在最终窗口上重算 F，
   请检查你自己的 run。
5. **回归**：v0.3.0 只在 N5、N6 上与早期按 N 的代码做过对照（差异与两次运行之间的噪声同量级，绝对值 ≤ 2e-11）；更大的 N 没有用
   v0.3.x 重算。
