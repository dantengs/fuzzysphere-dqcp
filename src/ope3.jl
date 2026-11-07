# ope3.jl: three-point local insertions and extended phi x phi channels
# on finished Tcc run directories.  Read-only on the run: every result goes to a separate directory.
#
# The F (half-sector) tensor component is not used: at N10/N11 F = G to 1e-14.
#
# Integer sector (sA,sB) = (0,0), windows of the run (raw eigenstates and KK-projected modes):
#   Gamma^2 density  n_l = n_A - n_B (phi component), any rank l:     <Y|n_l|X>
#   singlet insertion O_S = b_rho B_rho + b_Delta B_Delta (rank 0):   B = rho^dag rho, Delta^dag Delta
#       calibration: <stress|O_S^(rank 2)|0> = 0 and <S|O_S|0> = Z_S, Z_S = +-1 (S anchored by <phi|n_0|S> > 0)
#       lambda_XYS = (<Y|O_S|X> - delta_XY <0|O_S|0>) / Z_S
#   tensor insertion Q_G (G = (-Q11+4Q22-Q33-Q44-Q55)/sqrt20, exact Fierz form), Z_T = |<T|Q_G|0>|
#       lambda_TTT = <T|Q_G|T>/(Z_T 3/sqrt20) (trace CG; isometric x sqrt(27/20)),
#       lambda_STT(tensor insertion) = <S|Q_G|T>/Z_T
#   phi x phi -> spin l:  ratio = |<O_l|n_l|phi>| / |<phi|n_0|0>|,
#       lambda = c_l * CG_SO5 * ratio,  c_l = sqrt((2l+1)!!/l!)  (unit-normalised symmetric traceless
#       polarisation: c_1 = sqrt3 reproduces f_phiphiJ, c_2 = sqrt(15/2) = 2 x production stress factor)
#       CG_SO5: singlet 1, T14 (integer G component) sqrt5/2, 10 (half sector) 1, T14 (half) sqrt2.
# Half-integer sector (1/2,1/2), weight (1,1): windows L = 0..half_lmax solved here, n_l towers.
# All numbers are finite-N proxies, not certified CFT OPE coefficients.

const OPE3_VERSION = "ope3-v1"
spin_factor(l::Int) = sqrt(prod(1:2:(2l+1)) / factorial(l))

"""Q_G via the exact SO(5) Fierz identity."""
function tensor_g_fierz_couplings(nm; radius2=1.0)
    n = GetDensityObs(nm,2,Matrix{Float64}(I,2,2);norm_r2=1.0)
    Ω = Float64[0 1;0 0]
    pair = StoreComps(GetPairingObs(nm,2,Ω;norm_r2=1.0,mom_incr=false))
    Δ = StoreComps(SphereObs(pair.s2,pair.l2m,
        (l2,m2) -> (isodd(m2÷2) ? -1 : 1) * pair.get_comp(l2,-m2)))
    zero_shift = [0,0,0]
    aa = SingleSegCouple(2,1,SimplifyTerms(GetComponent(n*n,0,0)),zero_shift)
    bb = SingleSegCouple(2,2,SimplifyTerms(GetComponent(n*n,0,0)),zero_shift)
    ab = ContactCouple([n,n],zeros(Int,3,2),0)/sqrt(4π)
    na = SingleSegCouple(2,1,SimplifyTerms(GetComponent(n,0,0)),zero_shift)
    nb = SingleSegCouple(2,2,SimplifyTerms(GetComponent(n,0,0)),zero_shift)
    pair_ab = ContactCouple([Δ',Δ],[2 -2;0 0;0 0],0)/sqrt(4π)
    pair_ba = ContactCouple([Δ,Δ'],[-2 2;0 0;0 0],0)/sqrt(4π)
    k = nm/(4π)
    (4/(sqrt(20)*radius2^2)) *
        (aa+bb-ab-k*(na+nb)-2*(pair_ab+pair_ba))
end

# ---- four-flavour Terms gate for Q_G (own CG for j j -> 0) ----
const _τx = ComplexF64[0 1; 1 0]; const _τy = ComplexF64[0 -im; im 0]; const _τz = ComplexF64[1 0; 0 -1]
const _id2 = Matrix{ComplexF64}(I, 2, 2)
const _Γ5 = (kron(_id2, _τx), kron(_id2, _τz), kron(_τx, _τy), kron(_τy, _τy), kron(_τz, _τy))
_cg_jj0(j, m) = (iseven(round(Int, j - m)) ? 1.0 : -1.0) / sqrt(2j + 1)      # <j m j -m | 0 0>

function _qg_direct_terms(nm, radius2)
    rho = [GetDensityObs(nm, 4, _Γ5[i]; norm_r2=radius2) for i in 1:5]
    obs = zero(SphereObs)
    for (z, i) in zip((-1, 4, -1, -1, -1) ./ sqrt(20), 1:5)
        obs += z * rho[i] * rho[i]
    end
    SimplifyTerms(GetComponent(obs, 0, 0))
end

function _expanded_coupling_terms(nm, cpds)
    maps = (Dict(2o+f => 4o+(f == 1 ? 1 : 3) for o in 0:nm-1 for f in 1:2),
            Dict(2o+f => 4o+(f == 1 ? 2 : 4) for o in 0:nm-1 for f in 1:2))
    out = Term[]
    for cpd in cpds
        j1, j2 = cpd.ch[1,1], cpd.ch[1,2]
        j1 == j2 || error("non-scalar channel")
        for m in -j1:2:j1
            t1 = cpd.amd[1] === :Identity ? one(Terms) : RelabelOrbs(GetComponent(cpd.amd[1], j1/2, m/2), maps[1])
            t2 = cpd.amd[2] === :Identity ? one(Terms) : RelabelOrbs(GetComponent(cpd.amd[2], j2/2, -m/2), maps[2])
            append!(out, cpd.coeff * _cg_jj0(j1/2, m/2) * (t1 * t2))
        end
    end
    SimplifyTerms(out)
end

"""Run `f` with FuzzifiED.ObsNormRadSq = r2 (ContactCouple coefficients scale with it; Q_G is built
with the package default 1.0, whereas KKGenerator sets it to N)."""
function with_obs_r2(f, r2)
    old = FuzzifiED.ObsNormRadSq; FuzzifiED.ObsNormRadSq = Float64(r2)
    try; return f(); finally; FuzzifiED.ObsNormRadSq = old; end
end

"""Max |coefficient difference| between direct four-flavour Q_G Terms and the two-segment expansion."""
function qg_terms_gate(nm; radius2=Float64(nm))
    a = _qg_direct_terms(nm, radius2); b = with_obs_r2(() -> _expanded_coupling_terms(nm, tensor_g_fierz_couplings(nm; radius2)), 1.0)
    da = Dict(Tuple(t.cstr) => t.coeff for t in SimplifyTerms(a)); db = Dict(Tuple(t.cstr) => t.coeff for t in SimplifyTerms(b))
    (error=maximum((abs(get(da, k, 0im) - get(db, k, 0im)) for k in union(keys(da), keys(db))); init=0.0),
     n_direct=length(da), n_expanded=length(db))
end

"""Operator term sets, cached per (N, kind, rank) next to the KK term cache."""
function ope3_terms(N::Int, kind::Symbol, rank::Int; cache::AbstractString="")
    build() = kind === :gamma2 ? gamma2_terms(N, rank) :
              kind === :singlet ? (c = KKGenerator.generator_candidates(N; rank); [c.density_squared, c.symplectic_pair_squared]) :
              kind === :tensor_g ? (rank == 0 || error("Q_G is rank 0"); with_obs_r2(() -> [tensor_g_fierz_couplings(N; radius2=Float64(N))], 1.0)) :
              error("unknown term kind $kind")
    isempty(cache) && return build()
    key = bytes2hex(sha256(string("ope3_terms-v2|", kind, "|r=", rank, "|N=", N, "|src=", file_digest(@__FILE__),
                                  "|gen=", file_digest(joinpath(@__DIR__, "kk_generator.jl")),
                                  "|fz=", tree_digest(dirname(pathof(FuzzifiED))), "|jl=", VERSION)))
    dir = joinpath(cache, "ope3_$(kind)_r$(rank)_N$(N)_$(key[1:16])")
    isdir(dir) && return _checked_load(dir).terms
    t = build()
    try; _atomic_save(dir, (key=key, N=N, terms=t)); catch; isdir(dir) || rethrow(); end
    t
end

"""Segment spaces of a run directory exactly as the pipeline loads them."""
function ope3_spaces(c)
    out = c["out"]; sp = joinpath(out, "segment_split")
    if get(c, "residency", "all") != "all" && isdir(joinpath(sp, "space_2"))
        return SpaceStore(sp), _checked_load(joinpath(sp, "bases")).source_sha
    end
    seg = c["segment_checkpoint"]
    if isempty(seg)
        own = joinpath(out, "segment_spaces.ckpt"); return _checked_load(own).spaces, String(_ckpt_sha(own))
    end
    s = load_segment_spaces(seg); s.spaces, String(s.sha)
end

"""Integer (0,0) window of angular momentum l: run block if present, else the KK target window."""
function _ope3_window(out, l)
    w = source_window(out, 0, 0, l)
    w !== nothing && return merge(w, (kind="block",))
    t = target_window(out, 0, 0, l)
    t === nothing ? nothing : merge(t, (kind="target", parity=t.parity))
end

"""Cached matrix-element edge: overlaps[p] = T' O_p S[:, scols].
`legacy = (identity=..., ok=path -> Bool)` (v0.2.8): an edge stored under an older identity format is
still returned when `ok(path)` proves it was built from the current windows."""
function _ope3_edge(od, name, lib, ss, csS, csT, terms, ltot2, S, scols, T, ident; workers, image_bytes, legacy=nothing)
    path = joinpath(od, "edges", name * ".ckpt")
    if isdir(path)
        e = _checked_load(path); e.identity == ident && return e
        if legacy !== nothing && e.identity == legacy.identity && legacy.ok(path)
            println("OPE3_EDGE_LEGACY $name (pre-v0.2.8 identity, written after the current windows)"); flush(stdout)
            return e
        end
        mv(path, path * ".stale.$(round(Int, time()))")
    end
    t0 = time(); np = length(terms); Dt = size(T.V, 1)
    chunk = clamp(Int(fld(image_bytes, 8 * max(Dt, 1) * np)), 1, max(length(scols), 1))
    A = edge_arrays(lib, ss, csS, csT, terms, ltot2, Matrix(S.V[:, scols]), T.V; workers, chunk)
    e = (identity=ident, name=name, scols=collect(scols), overlaps=A.overlaps, image_gram=A.image_gram, seconds=time() - t0)
    _atomic_save(path, e)
    println("OPE3_EDGE $name sources=$(length(scols)) targets=$(size(T.V, 2)) seconds=$(round(time() - t0; digits=1))"); flush(stdout)
    e
end

"""KK-projected modes (threshold thr) of a run window, as coefficient vectors in window coordinates."""
function _ope3_modes(out, l, W, thr)
    p = joinpath(out, "kk", "projections", "s0_s0_L$(l).ckpt")
    isdir(p) || return Any[]
    x = _checked_load(p)
    (x.source_sha == W.sha && length(x.source_E) == length(W.E)) || (println("OPE3_WARN projection L$l built from another window; skipped"); return Any[])
    hit = filter(t -> t[1] == thr, x.threshold_channels); isempty(hit) && return Any[]
    modes = Any[]
    for ch in last(only(hit))
        ch.Q === nothing && continue
        k2 = diag(ch.Q' * ch.K2 * ch.Q)
        for j in axes(ch.Q, 2)
            v = zeros(length(W.E)); v[ch.source_ids] = ch.Q[:, j]
            push!(modes, (c2=ch.C2, mode=j, energy=ch.energies[j], sigmaE=ch.energy_mixing_residual[j], K2=k2[j], coeff=v,
                          label="L$(l)_C2_$(ch.C2)_m$(j)"))
        end
    end
    modes
end

_c2i(c) = abs(c - round(c)) < 1e-4 ? round(Int, c) : -1
_pickc(W, c) = sort([i for i in eachindex(W.E) if _c2i(W.C2[i]) == c]; by=i -> W.E[i])

"""
    run_ope3(cfg, spaces, basis_sha, od; opts...)

cfg: pipeline_config of the run (with resolved g).  od: new output directory.
opts (Dict): src_dmax (5.0), pair_dmax (6.5), int_ranks ([1,2,3,4]), half_lmax (3), half_nev (24), half_only (false),
threshold (0.25), image_bytes (16e9), qg_gate (N <= 7), cache (cfg kk_cache).
"""
function run_ope3(cfg, spaces, basis_sha, od; opts=Dict{String,Any}())
    out = cfg["out"]; N = cfg["N"]; g = cfg["g"]; act = get(cfg, "action", Threads.nthreads())
    src_dmax = Float64(get(opts, "src_dmax", 5.0)); pair_dmax = Float64(get(opts, "pair_dmax", 6.5))
    ranks = Int.(get(opts, "int_ranks", [1, 2, 3, 4])); thr = Float64(get(opts, "threshold", 0.25))
    hl = Int(get(opts, "half_lmax", 3)); hnev = Int(get(opts, "half_nev", 24)); imb = Float64(get(opts, "image_bytes", 16e9))
    cache = String(get(opts, "cache", get(cfg, "kk_cache", "")))
    mkpath(od); rep = Dict{String,Any}("N" => N, "g" => g, "run" => out, "version" => OPE3_VERSION, "tcc_version" => string(TCC_VERSION),
                                      "code_tree" => tree_digest(dirname(@__DIR__)), "git_commit" => get(ENV, "TCC_GIT_COMMIT", git_commit(@__DIR__)))
    gates = Dict{String,Any}()
    lib = SegLibrary(builder=:tiled, tile_width=get(cfg, "tile", 64), workers=get(cfg, "workers", Threads.nthreads()),
                     opmat_threads=get(cfg, "native", 8))
    if get(opts, "half_only", false)
        # v0.2.6: solve and cache only the half-integer (1/2,1/2) windows and edges (independent of the integer
        # run windows); a later full run_ope3 in the same OUT_DIR reuses them (same segment SHA, g, window size).
        half = _ope3_half!(od, N, g, lib, hl, hnev; act, image_bytes=imb, cache)
        rep["half_dims"] = half.dims; rep["half_only"] = true; rep["half_reverse_over_forward"] = half.revfwd
        open(io -> TOML.print(io, rep), joinpath(od, "ope3_half_only.toml"), "w")
        println("OPE3_HALF_ONLY_DONE dims=$(half.dims)"); flush(stdout)
        return rep
    end
    sp0 = spaces isa SpaceStore ? getspace!(spaces, 0) : spaces[1]; ss = [sp0, sp0]
    F = tuning_F(run_rows(out)); F === nothing && error("run has no ground/current/stress rows")
    alpha = 2 / (F.EJ - F.E0); E0 = F.E0; Δ(E) = alpha * (E - E0)
    rep["alpha"] = alpha; rep["E0"] = E0
    if get(opts, "qg_gate", N <= 7)
        q = qg_terms_gate(N); gates["QG_terms_error"] = q.error; gates["QG_terms_count"] = [q.n_direct, q.n_expanded]
        (q.error < 1e-11 && q.n_direct == q.n_expanded) || error("Q_G four-flavour Terms gate failed: $q")
        println("OPE3_GATE QG_terms error=$(q.error) terms=$(q.n_direct)")
    end
    # ---------------- windows ----------------
    W = Dict{Int,Any}()
    for l in 0:maximum([2; ranks])
        w = _ope3_window(out, l); w === nothing && continue
        w.basis_sha == basis_sha || error("window L$l computed in another segment basis")
        W[l] = w
    end
    haskey(W, 0) && haskey(W, 2) || error("ope3 needs the (0,0) L0 and L2 windows")
    cs = Dict(l => BuildCompSpace(ss, [2N, 0, 0], 2l) for l in keys(W))
    for l in keys(W); cs[l].dim == W[l].dim || error("CompSpace dimension mismatch L$l"); end
    W0 = W[0]; W2 = W[2]
    i0 = _pickc(W0, 0); i2 = _pickc(W0, 2); i5 = _pickc(W0, 5)
    (length(i0) >= 2 && !isempty(i2) && !isempty(i5)) || error("L0 window lacks vacuum/S/phi/T")
    ivac, iS, iphi, iT = i0[1], i0[2], i2[1], i5[1]
    istress = _pickc(W2, 0); isempty(istress) && error("L2 window lacks the stress tensor"); istress = istress[1]
    low = [i for i in eachindex(W0.E) if Δ(W0.E[i]) <= src_dmax]
    all0 = collect(eachindex(W0.E))
    idt(T, S, cols, tag) = (tag=tag, source_sha=S.sha, target_sha=T.sha, basis=basis_sha, cols=collect(cols), version=OPE3_VERSION)
    # ---------------- integer edges ----------------
    g0 = _ope3_edge(od, "gamma2_r0_L0_L0", lib, ss, cs[0], cs[0], ope3_terms(N, :gamma2, 0; cache), 0, W0, all0, W0,
                    idt(W0, W0, all0, "gamma2_0"); workers=act, image_bytes=imb).overlaps[1]
    b0 = _ope3_edge(od, "singlet_r0_L0_L0", lib, ss, cs[0], cs[0], ope3_terms(N, :singlet, 0; cache), 0, W0, all0, W0,
                    idt(W0, W0, all0, "singlet_0"); workers=act, image_bytes=imb).overlaps
    b2 = _ope3_edge(od, "singlet_r2_L0vac_L2", lib, ss, cs[0], cs[2], ope3_terms(N, :singlet, 2; cache), 4, W0, [ivac], W2,
                    idt(W2, W0, [ivac], "singlet_2"); workers=act, image_bytes=imb).overlaps
    q0 = _ope3_edge(od, "tensorG_r0_L0_L0", lib, ss, cs[0], cs[0], ope3_terms(N, :tensor_g, 0; cache), 0, W0, all0, W0,
                    idt(W0, W0, all0, "tensorG_0"); workers=act, image_bytes=imb).overlaps[1]
    all2 = collect(eachindex(W2.E))
    b22 = _ope3_edge(od, "singlet_r0_L2_L2", lib, ss, cs[2], cs[2], ope3_terms(N, :singlet, 0; cache), 0, W2, all2, W2,
                     idt(W2, W2, all2, "singlet_0"); workers=act, image_bytes=imb).overlaps
    gl = Dict{Int,Any}(0 => g0)
    for l in ranks
        haskey(W, l) || (println("OPE3_SKIP rank $l: no (0,0) L$l window"); continue)
        gl[l] = _ope3_edge(od, "gamma2_r$(l)_L0low_L$(l)", lib, ss, cs[0], cs[l], ope3_terms(N, :gamma2, l; cache), 2l, W0, low, W[l],
                           idt(W[l], W0, low, "gamma2_$l"); workers=act, image_bytes=imb).overlaps[1]
    end
    # ---------------- gates on the small matrices ----------------
    herm(M) = norm(M - M') / max(norm(M), eps())
    gates["hermiticity_singlet_L0"] = maximum(herm.(b0)); gates["hermiticity_tensorG_L0"] = herm(q0)
    gates["hermiticity_gamma2_L0"] = herm(g0); gates["hermiticity_singlet_L2"] = maximum(herm.(b22))
    maximum(v for (k, v) in gates if startswith(k, "hermiticity")) < 1e-7 || error("ope3 hermiticity gate failed: $gates")
    # ---------------- calibration ----------------
    nrm0 = g0[iphi, ivac]; aphi = sign(nrm0); nrm = abs(nrm0); nrm > 1e-12 || error("vanishing phi normalisation")
    aS = sign(aphi * g0[iphi, iS]); aS != 0 || error("phi-to-S anchor vanished")
    stress_raw = [b2[c][istress, 1] for c in 1:2]; S_raw = [aS * b0[c][iS, ivac] for c in 1:2]
    pst = sign(stress_raw[1]); pS = sign(S_raw[1]); rs = pst .* stress_raw; rS = pS .* S_raw
    R = [rs[1] rs[2]; rS[1] rS[2]]; beta = R \ [0.0, 1.0]
    nullres = abs(dot(rs, beta)) / max(norm(rs) * norm(beta), eps()); z_raw = dot(S_raw, beta)
    OS0 = beta[1] * b0[1] + beta[2] * b0[2]; OS2 = beta[1] * b22[1] + beta[2] * b22[2]; v0 = OS0[ivac, ivac]
    ZT_raw = q0[iT, ivac]; aT = sign(ZT_raw); ZT = abs(ZT_raw); ZT > 1e-12 || error("vanishing Z_T")
    cgT = 3 / sqrt(20)
    gates["stress_null_residual"] = nullres; gates["Z_S_raw"] = z_raw; gates["cond_R"] = cond(R)
    gates["QG_vacuum_forbidden"] = q0[ivac, ivac]; gates["QG_S_forbidden"] = q0[iS, iS]
    (nullres < 1e-8 && abs(abs(z_raw) - 1) < 1e-8) || error("singlet calibration gate failed")
    (abs(q0[ivac, ivac]) < 1e-8 * ZT && abs(q0[iS, iS]) < 1e-8 * ZT) ||
        error("Q_G SO(5) selection gate failed: <0|Q_G|0>=$(q0[ivac, ivac]) <S|Q_G|S>=$(q0[iS, iS]) Z_T=$ZT")
    rep["beta"] = beta; rep["vacuum_expectation_Os"] = v0; rep["Z_T"] = ZT; rep["phi_normalization"] = nrm
    # ---------------- named core channels (raw states, frozen conventions) ----------------
    named = Any[]
    addn(name, val, conv, note="") = push!(named, (name, "raw", val, conv, note))
    addn("phi_phi_S_Gamma2", aphi * aS * g0[iphi, iS] / nrm, "signed, S anchored <phi|n0|S> > 0")
    addn("phi_phi_S_Os", (OS0[iphi, iphi] - v0) / z_raw, "signed, same S anchor (independent insertion route)")
    addn("phi_phi_T14_Gamma2", sqrt(5) / 2 * abs(g0[iphi, iT]) / nrm, "|.| x sqrt5/2")
    addn("SSS", (OS0[iS, iS] - v0) / z_raw, "signed, S anchored by Gamma2")
    addn("STT_singlet_insertion", (OS0[iT, iT] - v0) / z_raw, "signed, S anchored by Gamma2")
    addn("STT_tensor_insertion", aS * aT * q0[iS, iT] / ZT, "signed, S anchored by Gamma2, T by Z_T > 0")
    addn("TTT_trace", q0[iT, iT] / (ZT * cgT), "trace Clebsch 3/sqrt20")
    addn("TTT_isometric", sqrt(27 / 20) * q0[iT, iT] / (ZT * cgT), "isometric Clebsch")
    addn("stress_S_stress_rank0", (OS2[istress, istress] - v0) / z_raw, "rank-0 component of <T_mn|O_S|T_mn>, signed")
    haskey(gl, 2) && addn("phi_phi_stress_canonical", spin_factor(2) * abs(gl[2][istress, findfirst(==(iphi), low)]) / nrm,
         "c_2 = sqrt(15/2) (production f_phiphistress = 1/2 of this)")
    # ---------------- generic three-point tables (raw states and KK modes) ----------------
    rows3 = Any[]
    function table!(basis, st0, M0, MS, MQ, nsrc)
        for a in eachindex(st0), b in a:length(st0)
            x = st0[a]; y = st0[b]; cx = x.c2; cy = y.c2
            gm = dot(y.coeff, M0 * x.coeff); sm = dot(y.coeff, MS * x.coeff); qm = dot(y.coeff, MQ * x.coeff)
            diagp = a == b
            pr = Set([cx, cy])
            cgG = pr == Set([2, 0]) ? 1.0 : pr == Set([2, 5]) ? sqrt(5) / 2 : NaN
            cgQ = pr == Set([5, 0]) ? 1.0 : pr == Set([5]) ? cgT : NaN
            push!(rows3, (basis, "Gamma2_phi", x.label, y.label, cx, cy, Δ(x.energy), Δ(y.energy), x.sigmaE, y.sigmaE, gm, nrm, cgG,
                          isnan(cgG) ? abs(gm) / nrm : cgG * abs(gm) / nrm, isnan(cgG) ? "CG_not_derived_ratio_only" : "abs"))
            cx == cy && push!(rows3, (basis, "O_S", x.label, y.label, cx, cy, Δ(x.energy), Δ(y.energy), x.sigmaE, y.sigmaE, sm, z_raw, 1.0,
                          (sm - (diagp ? v0 : 0.0)) / z_raw, diagp ? "signed_vacuum_subtracted" : "signed_state_phase_arbitrary"))
            push!(rows3, (basis, "Q_G", x.label, y.label, cx, cy, Δ(x.energy), Δ(y.energy), x.sigmaE, y.sigmaE, qm, ZT, cgQ,
                          isnan(cgQ) ? qm / ZT : qm / (ZT * cgQ), isnan(cgQ) ? "CG_not_derived_ratio_only" : diagp ? "signed" : "signed_state_phase_arbitrary"))
        end
    end
    sel = [i for i in eachindex(W0.E) if Δ(W0.E[i]) <= pair_dmax && _c2i(W0.C2[i]) in (0, 2, 5)]
    phase(i) = i == iphi ? aphi : i == iS ? aS : i == iT ? aT : 1.0
    raw = [(c2=_c2i(W0.C2[i]), energy=W0.E[i], sigmaE=0.0, K2=NaN, label="raw_$(i)",
            coeff=(v = zeros(length(W0.E)); v[i] = phase(i); v)) for i in sel]
    table!("raw", raw, g0, OS0, q0, 0)
    modes = [m for m in _ope3_modes(out, 0, W0, thr) if m.c2 in (0, 2, 5) && Δ(m.energy) <= pair_dmax]
    if !isempty(modes)
        # anchor every mode: <phi|n0|mode> > 0 (C2 0,5), <mode|n0|0> > 0 (C2 2); else <mode|O_S|0>, <mode|Q_G|0>
        anch(m) = (s = m.c2 == 2 ? dot(m.coeff, g0[:, ivac]) : m.c2 == 0 ? dot(g0[iphi, :], m.coeff) * aphi :
                       dot(g0[iphi, :], m.coeff) * aphi;
                   abs(s) > 1e-10 ? sign(s) : (t = m.c2 == 5 ? dot(m.coeff, q0[:, ivac]) : dot(m.coeff, OS0[:, ivac]); t == 0 ? 1.0 : sign(t)))
        modes = [merge(m, (coeff=anch(m) .* m.coeff,)) for m in modes]
        table!("kk_t$(thr)", modes, g0, OS0, q0, 0)
        bym(c, j) = (k = findfirst(m -> m.c2 == c && m.mode == j, modes); k === nothing ? nothing : modes[k])
        vp = bym(2, 2); tp = bym(5, 2); Sm = bym(0, 2); Tm = bym(5, 1); ph = bym(2, 1)
        nm(s) = s === nothing ? "missing" : "Delta=$(round(Δ(s.energy); digits=4)) sigmaE=$(round(s.sigmaE; sigdigits=3)) K2=$(round(s.K2; sigdigits=3))"
        if vp !== nothing && ph !== nothing
            Sx = Sm === nothing ? raw[findfirst(r -> r.label == "raw_$(iS)", raw)] : Sm
            Tx = Tm === nothing ? raw[findfirst(r -> r.label == "raw_$(iT)", raw)] : Tm
            push!(named, ("phi_S_vprime_Gamma2", "kk", abs(dot(vp.coeff, g0 * Sx.coeff)) / nrm, "|.|", "v'=" * nm(vp)))
            push!(named, ("phi_S_vprime_Os", "kk", abs(dot(vp.coeff, OS0 * ph.coeff)) / abs(z_raw), "|.| (O_S route)", "v'=" * nm(vp)))
            push!(named, ("phi_T_vprime_Gamma2", "kk", sqrt(5) / 2 * abs(dot(vp.coeff, g0 * Tx.coeff)) / nrm, "|.| x sqrt5/2", "v'=" * nm(vp)))
            push!(named, ("vprime_vprime_S", "kk", (dot(vp.coeff, OS0 * vp.coeff) - v0) / z_raw, "signed (S anchored)", "v'=" * nm(vp)))
            push!(named, ("density_vprime_admixture", "kk", abs(dot(vp.coeff, g0[:, ivac])) / nrm,
                          "|<v'|n0|0>|/|<phi|n0|0>|: v' content of the inserted density (finite-N systematic, not an OPE)", "v'=" * nm(vp)))
        end
        if tp !== nothing
            Tx = Tm === nothing ? raw[findfirst(r -> r.label == "raw_$(iT)", raw)] : Tm
            push!(named, ("T_T_tprime", "kk", dot(tp.coeff, q0 * Tx.coeff) / (ZT * cgT), "trace Clebsch, sign arbitrary", "t'=" * nm(tp)))
            push!(named, ("tprime_tprime_S", "kk", (dot(tp.coeff, OS0 * tp.coeff) - v0) / z_raw, "signed (S anchored)", "t'=" * nm(tp)))
            push!(named, ("phi_phi_tprime_Gamma2", "kk", sqrt(5) / 2 * abs(dot(g0[iphi, :], tp.coeff)) / nrm, "|.| x sqrt5/2", "t'=" * nm(tp)))
        end
    else
        println("OPE3_NOTE no KK projection for (0,0) L0: generic table in raw basis only")
    end
    # stress S stress / spin-2 diagonal table in L2 (raw and KK modes)
    rows2 = Any[]
    sel2 = [i for i in eachindex(W2.E) if Δ(W2.E[i]) <= pair_dmax && _c2i(W2.C2[i]) in (0, 5)]
    for i in sel2
        push!(rows2, ("raw", "raw_$(i)", _c2i(W2.C2[i]), Δ(W2.E[i]), 0.0, OS2[i, i], (OS2[i, i] - v0) / z_raw))
    end
    for m in _ope3_modes(out, 2, W2, thr)
        m.c2 in (0, 5) && Δ(m.energy) <= pair_dmax || continue
        s = dot(m.coeff, OS2 * m.coeff); push!(rows2, ("kk_t$(thr)", m.label, m.c2, Δ(m.energy), m.sigmaE, s, (s - v0) / z_raw))
    end
    # ---------------- phi x phi spin-l towers, integer sector ----------------
    rowsT = Any[]
    jphi = findfirst(==(iphi), low)
    for (l, M) in sort(collect(gl); by=first)
        Tw = W[l]; col = l == 0 ? M[:, iphi] : M[:, jphi]
        for t in eachindex(Tw.E)
            c = _c2i(Tw.C2[t]); c in (0, 5) || continue
            r = abs(col[t]) / nrm; cg = c == 0 ? 1.0 : sqrt(5) / 2
            push!(rowsT, ("integer", "raw", l, "raw_$(t)", c, Tw.E[t], Δ(Tw.E[t]), NaN, NaN, r, spin_factor(l), cg, spin_factor(l) * cg * r,
                          isodd(l) ? "odd_spin_symmetric_rep: primaries must vanish (Bose); descendants allowed" : ""))
        end
        for m in _ope3_modes(out, l, Tw, thr)
            m.c2 in (0, 5) || continue
            r = abs(dot(m.coeff, col)) / nrm; cg = m.c2 == 0 ? 1.0 : sqrt(5) / 2
            push!(rowsT, ("integer", "kk_t$(thr)", l, m.label, m.c2, m.energy, Δ(m.energy), m.sigmaE, m.K2, r, spin_factor(l), cg,
                          spin_factor(l) * cg * r, isodd(l) ? "odd_spin_symmetric_rep: primaries must vanish (Bose)" : ""))
        end
    end
    # ---------------- half-integer sector towers ----------------
    if hl >= 0
        half = _ope3_half!(od, N, g, lib, hl, hnev; act, image_bytes=imb, cache)
        rep["half_dims"] = half.dims; gates["half_reverse_over_forward"] = half.revfwd
        for (l, w) in sort(collect(half.win); by=first)
            col = half.amp[l]
            for t in eachindex(w.energy)
                c = _c2i(w.c2[t]); r = abs(col[t]) / nrm
                cg = c == 3 ? 1.0 : c == 5 ? sqrt(2) : NaN
                note = c == 3 ? (iseven(l) ? "even_spin_antisymmetric_rep: primaries must vanish (Bose)" : "") :
                       c == 5 ? (isodd(l) ? "odd_spin_symmetric_rep: primaries must vanish (Bose)" : "") :
                       "not in 5x5 = 1+10+14: must vanish (selection)"
                push!(rowsT, ("half", "raw", l, "half_raw_$(t)", c, w.energy[t], Δ(w.energy[t]), NaN, NaN, r, spin_factor(l), cg,
                              isnan(cg) ? NaN : spin_factor(l) * cg * r, note))
            end
        end
        sel_bad = [r[10] for r in rowsT if r[1] == "half" && !(r[5] in (3, 5))]
        gates["half_selection_max_ratio"] = isempty(sel_bad) ? 0.0 : maximum(sel_bad)
        hp = half.win[0]; ip = _pickc((E=hp.energy, C2=hp.c2), 2)
        rep["half_vs_integer_phi_energy_difference"] = hp.energy[ip[1]] - W0.E[iphi]
        abs(rep["half_vs_integer_phi_energy_difference"]) < 1e-7 || error("half-sector phi energy differs from the integer phi")
        # cross-sector consistency: T14 spin 0 and spin 2 (lowest states) integer vs half
        for l in (0, 2)
            haskey(gl, l) && haskey(half.win, l) || continue
            ti = _pickc(W[l], 5); th = _pickc((E=half.win[l].energy, C2=half.win[l].c2), 5)
            (isempty(ti) || isempty(th)) && continue
            colI = l == 0 ? gl[0][:, iphi] : gl[l][:, jphi]
            li = spin_factor(l) * sqrt(5) / 2 * abs(colI[ti[1]]) / nrm; lh = spin_factor(l) * sqrt(2) * abs(half.amp[l][th[1]]) / nrm
            gates["T14_spin$(l)_integer_vs_half"] = [li, lh, W[l].E[ti[1]] - half.win[l].energy[th[1]]]
        end
        gates["half_selection_max_ratio"] < 1e-7 || println("OPE3_WARN half-sector selection rule violated: $(gates["half_selection_max_ratio"])")
    end
    # ---------------- outputs ----------------
    _csv(joinpath(od, "ope3_named.csv"), ["channel", "basis", "value", "convention", "states"], named)
    _csv(joinpath(od, "ope3_three_point.csv"), ["basis", "insertion", "ket", "bra", "C2_ket", "C2_bra", "Delta_ket", "Delta_bra",
         "sigmaE_ket", "sigmaE_bra", "matrix_element", "normalization", "CG", "value", "convention"], rows3)
    _csv(joinpath(od, "ope3_spin2_singlet_insertion.csv"), ["basis", "state", "C2", "Delta", "sigmaE", "matrix_element", "lambda_XXS"], rows2)
    _csv(joinpath(od, "ope3_phiphi_spin.csv"), ["sector", "basis", "spin", "state", "C2", "energy", "Delta", "sigmaE", "K2", "ratio",
         "c_l", "CG_SO5", "lambda_canonical", "note"], rowsT)
    for r in named; rep["named:" * r[1]] = r[3]; end
    rep["gates"] = gates
    rep["interpretation"] = "finite-N local-operator proxies; no continuum extrapolation; not certified CFT OPE data"
    open(io -> TOML.print(io, rep), joinpath(od, "ope3.toml"), "w")
    println("OPE3_DONE ", join(("$(r[1])=$(round(r[3]; sigdigits=8))" for r in named), " ")); flush(stdout)
    rep
end

"""v0.2.8: a pre-v0.2.8 half edge provably belongs to the current windows when none of them was ever replaced
in this directory (no eig.ckpt.stale.*) and the edge was written after each of them."""
function _half_legacy_ok(edge, eigs)
    te = mtime(joinpath(edge, "payload.jls"))
    all(p -> !any(startswith("eig.ckpt.stale."), readdir(dirname(p))) && te >= mtime(joinpath(p, "payload.jls")), eigs)
end

"""Half-integer (1/2,1/2) weight-(1,1) windows L = 0..lmax and Gamma^2 rank-l edges from phi (L0)."""
function _ope3_half!(od, N, g, lib, lmax, nev; act, image_bytes, cache, tol=1e-10)
    hd = joinpath(od, "half"); segp = joinpath(hd, "segment.ckpt")
    isdir(segp) || _atomic_save(segp, (N=N, space=build_half_segment(N)))
    seg = _checked_load(segp).space; hs = [seg, seg]; tpl = hamiltonian_template(N); ssha = String(_ckpt_sha(segp))
    cs = [BuildCompSpace(hs, [2N, 0, 2], 2l) for l in 0:lmax]
    haskey(HALF_DIMS, N) && lmax >= 1 && (cs[1].dim, cs[2].dim) != HALF_DIMS[N] && error("half-sector dimension mismatch")
    sH, fH = segops_shared!(lib, hs, tpl.template); sC, fC = segops_shared!(lib, hs, tpl.c2)
    win = Dict{Int,Any}()
    for l in 0:lmax
        p = joinpath(hd, "blocks", "L$l", "eig.ckpt"); D = cs[l+1].dim; k = min(nev, D)
        if isdir(p)
            w = _checked_load(p)
            if w.segment_sha == ssha && w.g == g && length(w.energy) >= k
                win[l] = w; continue
            end
            mv(p, p * ".stale.$(round(Int, time()))")
        end
        t0 = time()
        H = absorb!(set_g!(BuildCompOperator(cs[l+1], tpl.template, sH; num_th=act), tpl, g), fH)
        C = absorb!(BuildCompOperator(cs[l+1], tpl.c2, sC; num_th=act), fC)
        E, Vs, info = lowest(v -> *(H, v; num_th=act), D, k, 616161 + l; tol)
        V = reduce(hcat, Vs); par = zeros(Int, length(E))
        resolve_c2!(E, V, par, C; tol=1e-6, workers=act)
        polish_c2!(E, V, par, H, C; workers=act)
        o = sortperm(E); E = E[o]; V = V[:, o]
        rh = post_h(H, V, E; workers=act); c2, rc = post_c(C, V; workers=act)
        maximum(rh; init=0.0) <= 1e-6 && norm(V' * V - I) <= 1e-6 || error("half L$l H/orthogonality gate failed")
        nbad = count(>(1e-6), rc)
        w = (energy=E, vectors=V, c2=c2, rh=rh, rc=rc, dim=D, g=g, segment_sha=ssha, c2_unresolved=nbad, converged=info.converged)
        _atomic_save(p, w); win[l] = w
        H = nothing; C = nothing; GC.gc()
        println("OPE3_HALF L$l dim=$D k=$k seconds=$(round(time() - t0; digits=1)) maxrh=$(maximum(rh)) c2_unresolved=$nbad"); flush(stdout)
    end
    ip = _pickc((E=win[0].energy, C2=win[0].c2), 2); isempty(ip) && error("half sector: no phi"); ip = ip[1]
    # v0.2.8 (audit A3): half edges are bound to the payload SHA of the exact eig checkpoints (it changes with g,
    # the window size and any recomputation) and to g.  v0.2.7 used "segment|L|count", so a window recomputed at
    # another g with the same number of states silently reused the old matrix elements.
    eigp(l) = joinpath(hd, "blocks", "L$l", "eig.ckpt")
    wsha = Dict(l => String(_ckpt_sha(eigp(l))) for l in 0:lmax)
    oldsha(l) = ssha * "|L$l|" * string(length(win[l].energy))            # the v0.2.7 identity strings
    W0h = (V=win[0].vectors, sha=wsha[0])
    amp = Dict{Int,Vector{Float64}}(); revfwd = Dict{String,Any}()
    for l in 0:lmax
        Tw = (V=win[l].vectors, sha=wsha[l])
        ok = path -> _half_legacy_ok(path, eigp.(unique([0, l])))
        e = _ope3_edge(hd, "half_gamma2_r$(l)_phi_L$(l)", lib, hs, cs[1], cs[l+1], ope3_terms(N, :gamma2, l; cache), 2l, W0h, [ip], Tw,
                       (tag="half_gamma2_$l", src=W0h.sha, tgt=Tw.sha, g=g, segment=ssha, version=OPE3_VERSION); workers=act, image_bytes=image_bytes,
                       legacy=(identity=(tag="half_gamma2_$l", src=oldsha(0), tgt=oldsha(l), version=OPE3_VERSION), ok=ok))
        amp[l] = e.overlaps[1][:, 1]
        # reverse element <phi|n_l|X> for the largest forward target (conjugacy pattern, diagnostic)
        t = argmax(abs.(amp[l]))
        if abs(amp[l][t]) > 1e-10
            Xw = (V=win[l].vectors[:, t:t], sha=Tw.sha * "|col$t")
            r = _ope3_edge(hd, "half_gamma2_r$(l)_rev_L$(l)", lib, hs, cs[l+1], cs[1], ope3_terms(N, :gamma2, l; cache), 2l, Xw, [1], W0h,
                           (tag="half_gamma2_rev_$l", src=Xw.sha, tgt=W0h.sha, g=g, segment=ssha, version=OPE3_VERSION); workers=act, image_bytes=image_bytes,
                           legacy=(identity=(tag="half_gamma2_rev_$l", src=oldsha(l) * "|col$t", tgt=oldsha(0), version=OPE3_VERSION), ok=ok))
            revfwd["L$l"] = [r.overlaps[1][ip, 1] / amp[l][t], (-1)^l * sqrt(2l + 1)]
        end
    end
    (win=win, amp=amp, dims=[c.dim for c in cs], revfwd=revfwd)
end
