# Stage B (v0.3.0): one more (sA,sB) sector of a finished run, solved directly as raw L windows, then
# KK projection with the run's own generator fit, then full-Q three-point OPE components.
#
#   sector "half": the half-integer sector (1/2,1/2) (segments with odd ne, segment spin 1/2, Q = [2N,0,2]);
#   sector "11"  : the integer (1,1) sector with windows grown to the raw Delta cut.
#
# The window solver uses memory-lean buffers; its functions repeat the floating-point operations of the
# Tcc functions named in their docstrings (bitwise identical results).
# Options are keys of the [stageb] table:
#
#   out, ope3 (the run's ope3 directory: half segment basis, half windows as deflation priors, phi-J edge),
#   sector ("half"), lmax (4; KK sources and OPE spins 0..lmax, raw windows 0..lmax+1),
#   step ("all" | "windows" | "probe"), windows ("all" or a list of L for a window-only job),
#   raw_delta_max (6.8), source_delta_max (5.0), target_delta_max (6.6), max_nev (384), threshold (0.25),
#   insertions (["gamma2","tensorG","singlet"]), memory_limit_bytes (85 % of SLURM_MEM_PER_NODE, else 280e9; a solve
#   step that would exceed it is refused before allocating), image_bytes (4e9), post_chunk (8),
#   rawtarget (false: also the OPE edges onto the full raw target windows, see run_rawtarget).
#
# v0.3.1 merges the later lean scripts (same Hamiltonian, seeds and thresholds; docs/en/METHODS.md section 10):
#   stageB-v2.1-lean (2026-11-02)  partial_checkpoints (true): every grown window that does not yet guard
#                                  the raw cut is saved to windows/partial/L<L>_k<k>.ckpt; a restart resumes
#                                  from the largest k on the same step sequence.  stop_after_k (0): with
#                                  step = "windows", stop cleanly once a partial window with >= k states is saved.
#   stageB-v2.2-lean (2026-11-06)  extra_states (4): every grown step solves this many states beyond k; the gap
#                                  above the certified top is the complement margin (the number the separate
#                                  deflated complement solve computes).  complement_tol (1e-6, at most guard/10):
#                                  tolerance of a remaining deflated complement check.  final_polish_trigger
#                                  (1e-9; 0 = off): C2 polish of the certified window before the PH resolution.
#   stageB-v2.1-prelim81 (11-04)   prelim_top_window (""): a partial checkpoint used as the raw window of the
#                                  top raw L (= lmax + 1, a KK target only); KK sources above its guard are
#                                  dropped (prelim_truncated_sources.csv) and the export is marked preliminary.
#                                  prelim_complement (false): complement check of that window after the export.
#   extra_states = 0, complement_tol = 1e-10, final_polish_trigger = 0 reproduce v0.3.0 (stageB-v2-lean).
#
# Identity: every checkpoint is bound to `input_binding.toml` (N, g, sector, raw Delta cut, calibration,
# normalization and segment basis SHAs, source directories).  Code versions are recorded in
# stageB_manifest.toml but are not part of the identity (as in the pipeline), so a restart with another
# Tcc version reuses windows, edges and projections whose physical identity matches.

const STAGEB_VERSION = "stageB-v3"                              # identity format of input_binding.toml (unchanged in v3.1)
const STAGEB_REVISION = "stageB-v3.1"                           # code revision, recorded in jobs/ and the manifest
const STAGEB_REPS = Set([0, 2, 3, 5, 6, 8, 9, 10, 14])         # C2 of the SO(5) irreps given to the KK step

"""Thrown by the window loop once a partial window with >= stop_after_k states is saved."""
struct StageBPartialStop <: Exception
    L::Int
    states::Int
    next_k::Int
end

"""
Window top certified by the solved states above it (sorted E): an index j with E[j]-E[j-1] > guard (the
frozen sector_frontier) and E[j+1]-E[j] > guard (complement margin).  The first such j >= k, else the last
one below k; 0 if there is none.
"""
function certified_top(E, k, guard)
    ok(j) = 2 <= j < length(E) && E[j] - E[j-1] > guard && E[j+1] - E[j] > guard
    for j in k:length(E)-1; ok(j) && return j; end
    for j in min(k, length(E))-1:-1:2; ok(j) && return j; end
    0
end

"""Vnew[:, j] = V[:, o[j]] in place (one column buffer); same bytes as V[:, o]."""
function permute_cols!(V::AbstractMatrix{Float64}, o::AbstractVector{Int})
    n = length(o); n == size(V, 2) || error("permutation length")
    all(o[j] == j for j in 1:n) && return V
    done = falses(n); tmp = Vector{Float64}(undef, size(V, 1))
    for i in 1:n
        (done[i] || o[i] == i) && (done[i] = true; continue)
        copyto!(tmp, view(V, :, i)); j = i
        while true
            done[j] = true; k = o[j]
            k == i && (copyto!(view(V, :, j), tmp); break)
            copyto!(view(V, :, j), view(V, :, k)); j = k
        end
    end
    V
end

function _procmem()
    isfile("/proc/self/status") || return (rss=Float64(Base.gc_live_bytes()), hwm=Float64(Sys.maxrss()))
    s = read("/proc/self/status", String)
    f(key) = (m = match(Regex("$key:\\s+(\\d+) kB"), s); m === nothing ? -1.0 : parse(Float64, m[1]) * 1024)
    (rss=f("VmRSS"), hwm=f("VmHWM"))
end
function _memlog(tag)
    GC.gc(); p = _procmem()
    println("STAGEB_MEM $tag rss_gb=$(round(p.rss / 1e9; digits=2)) hwm_gb=$(round(p.hwm / 1e9; digits=2))"); flush(stdout)
    p
end

# ---- lean window solver -------------------------------------------------------------------------

"""Tcc.lowest without the final per-vector copies (Krylov path); dense path delegated."""
function lean_lowest(f, n::Int, k::Int, seed::Int; tol=1e-10, maxiter=500)
    k = min(k, n)
    n <= 400 && return lowest(f, n, k, seed; tol, maxiter)[1:2]
    alg = KrylovKit.Lanczos(krylovdim=min(n, krylovdim(k)), tol=tol, maxiter=maxiter, verbosity=0)
    E, V, info = KrylovKit.eigsolve(f, normalize(randn(MersenneTwister(seed), n)), k, :SR, alg)
    o = sortperm(E)[1:min(k, length(E))]
    Float64.(E[o]), [V[i] isa Vector{Float64} ? V[i] : Vector{Float64}(V[i]) for i in o]
end

"""Tcc.lowest_deflated; the new eigenvectors are written into `dest` (n x k view), energies returned."""
function lean_deflated_into!(f, n::Int, k::Int, V0::AbstractMatrix{Float64}, dest::AbstractMatrix{Float64}, seed::Int;
                             tol=1e-10, maxiter=500, Eref=nothing)
    m = size(V0, 2); k = min(k, n - m)
    k <= 0 && return Float64[]
    m == 0 && error("lean_deflated_into! needs a non-empty deflation basis")
    scale = Eref === nothing ? 1.0 : maximum(abs, Eref; init=1.0)
    sigma = 10 * scale + 100
    buf = zeros(m); pv = zeros(n)
    function g(v)
        mul!(buf, transpose(V0), v); mul!(pv, V0, buf)
        y = f(v - pv)
        mul!(buf, transpose(V0), y); mul!(y, V0, buf, -1.0, 1.0)
        y .+= sigma .* pv
        y
    end
    E, V = lean_lowest(g, n, k, seed; tol, maxiter)
    GC.gc()
    r = length(V); W = Matrix{Float64}(undef, n, r)
    for j in 1:r
        v = V[j]; W[:, j] = v - V0 * (V0' * v); V[j] = Float64[]
    end
    V = nothing; GC.gc()
    Qw = Matrix(qr!(W).Q)                        # qr(W) of the reference code = qr! on its copy
    size(Qw, 2) == r || (Qw = Qw[:, 1:r])
    W = nothing; GC.gc()
    FQ = Matrix{Float64}(undef, n, r)
    for j in 1:r; FQ[:, j] = f(Qw[:, j]); end
    Hs = Qw' * FQ
    FQ = nothing; GC.gc()
    F = eigen(Symmetric((Hs + Hs') / 2))
    mul!(view(dest, :, 1:r), Qw, F.vectors)
    maximum(abs.(F.values .- E)) < 1e-6 * max(1, maximum(abs, E)) ||
        @warn "deflated Ritz values moved: $(maximum(abs.(F.values .- E)))"
    Float64.(F.values)
end

"""
Tcc._target_window(H, cp, 0, 0, k, seed, false, act, tol; prior) for one full block (no exchange
split).  `box[]` holds the prior window (energy, vectors) or nothing; it is consumed: the prior
columns are copied once into the grown window and the old matrix is released before Lanczos.
"""
function lean_target_window!(box::Ref{Any}, H, D::Int, k::Int, seed::Int, act::Int, tol)
    f = v -> *(H, v; num_th=act)
    pr = box[]; box[] = nothing
    if pr === nothing || isempty(pr.energy)                  # empty prior -> plain lowest
        e, V = lean_lowest(f, D, k, seed + 7; tol)
        GC.gc()
        o = sortperm(e); W = Matrix{Float64}(undef, D, length(e))
        for (j, i) in enumerate(o); copyto!(view(W, :, j), V[i]); V[i] = Float64[]; end
        return (energy=e[o], vectors=W)
    end
    m = length(pr.energy); size(pr.vectors) == (D, m) || error("prior window shape")
    a = max(0, min(k - m, D - m))
    Eref = copy(pr.energy)
    Vout = Matrix{Float64}(undef, D, m + a); copyto!(view(Vout, :, 1:m), pr.vectors)
    pr = nothing; GC.gc()
    e = Float64[]
    if a > 0
        e = lean_deflated_into!(f, D, a, view(Vout, :, 1:m), view(Vout, :, m+1:m+a), seed + 7 + 104729; tol, Eref)
        !isempty(e) && minimum(e) < maximum(Eref) - 1e-8 &&
            error("target window sector 0: a deflated state lies below the prior window")
        length(e) < a && (Vout = Vout[:, 1:m+length(e)]; GC.gc())
    end
    E = vcat(Eref, e); o = sortperm(E)
    permute_cols!(Vout, o)
    (energy=E[o], vectors=Vout)
end

"""Tcc.polish_c2! with the C2 residual pass in column chunks (identical per-column arithmetic)."""
function lean_polish_c2!(E, V, parity, H, C; window=1e-3, trigger=1e-7, workers=Threads.nthreads(), chunk=8)
    isempty(E) && return 0
    _, rc = post_c(C, V; workers, chunk)
    done = falses(length(E)); ngroups = 0
    for i in sortperm(rc; rev=true)
        (rc[i] <= trigger || done[i]) && continue
        idx = [j for j in eachindex(E) if parity[j] == parity[i] && abs(E[j] - E[i]) < window]
        length(idx) < 2 && continue
        W = V[:, idx]; CW = zeros(size(W)); act_block!(CW, C, W; workers)
        F = eigen(Hermitian((W' * CW + CW' * W) / 2)); U = W * F.vectors
        groups = Vector{Int}[]
        for (k, c) in enumerate(F.values)
            isempty(groups) || abs(c - F.values[last(groups)[1]]) > 1e-6 ? push!(groups, [k]) : push!(last(groups), k)
        end
        newV = Vector{Float64}[]; newE = Float64[]
        for gk in groups
            Ug = U[:, gk]; HU = zeros(size(Ug)); act_block!(HU, H, Ug; workers)
            G = eigen(Hermitian((Ug' * HU + HU' * Ug) / 2)); R = Ug * G.vectors
            append!(newE, G.values); append!(newV, [R[:, q] for q in axes(R, 2)])
        end
        o = sortperm(newE); ix = sort(idx; by=j -> E[j])
        for (q, j) in enumerate(ix); E[j] = newE[o[q]]; V[:, j] = newV[o[q]]; end
        done[idx] .= true; ngroups += 1
    end
    ngroups
end

"""Tcc._complement_check(H, cp, 0, 0, w, Dict(0=>D), false, act, tol, seed).min_margin with the window
matrix itself as deflation basis (the reference function copies it into V0 first)."""
function lean_complement_check(H, D::Int, E::AbstractVector, V::Matrix{Float64}, act::Int, tol, seed::Int; nextra::Int=2)
    m = length(E); m >= D && return (min_margin=Inf,)
    f = v -> *(H, v; num_th=act)
    top = m == 0 ? -Inf : maximum(E)
    Ec, _, _ = lowest_deflated(f, D, nextra, V, seed; tol, Eref=E[1:m])
    (min_margin=isempty(Ec) ? Inf : minimum(Ec) - top,)
end

"""Rotate PH-degenerate pairs (same E and C2) into P_seg eigenstates; returns P_seg and the worst residual."""
function ph_resolve!(E, V, C2, plan)
    k = length(E); pm = zeros(k, k)
    for j in 1:k
        pv = apply_parity(plan, Matrix(V[:, j:j])); pm[:, j] = V' * view(pv, :, 1)
    end
    done = falses(k)
    for i in 1:k
        done[i] && continue
        ids = [j for j in i:k if !done[j] && abs(E[j] - E[i]) <= 1e-7 && abs(C2[j] - C2[i]) < 1e-5]
        if length(ids) > 1
            u = eigen(Hermitian((pm[ids, ids] + pm[ids, ids]') / 2)).vectors; V[:, ids] = V[:, ids] * u
        end
        done[ids] .= true
    end
    p = zeros(Int, k); res = 0.0
    for j in 1:k
        vals, rs = parity_values(plan, Matrix(V[:, j:j])); res = max(res, only(rs)); p[j] = round(Int, only(vals))
        abs(abs(only(vals)) - 1) <= 1e-6 && only(rs) <= 1e-6 || error("Raw state not a pure PH eigenstate")
    end
    p, res
end

"""
Tensor insertion Q_G of rank l as two parts (density, pairing); the (-1)^rank combination is applied by the
analysis, never here.  ObsNormRadSq = 1 and the 1/N^2 radius factor in the prefactor.
"""
function tensor_g_parts(N, l)
    n = GetDensityObs(N, 2, Matrix{Float64}(I, 2, 2); norm_r2=1.0)
    pair = StoreComps(GetPairingObs(N, 2, [0.0 1; 0 0]; norm_r2=1.0, mom_incr=false))
    d = StoreComps(SphereObs(pair.s2, pair.l2m, (l2, m2) -> (-1)^(m2 ÷ 2) * pair.get_comp(l2, -m2)))
    nn = FuzzifiED.SO3lver.AngModes(StoreComps(n * n)); one = FuzzifiED.SO3lver.AngModes(StoreComps(n)); q = [0, 0, 0]
    aa = SingleSegCouple(2, 1, nn, 2l, q); bb = SingleSegCouple(2, 2, nn, 2l, q)
    ab = ContactCouple([n, n], zeros(Int, 3, 2), 2l) / sqrt(4pi)
    na = SingleSegCouple(2, 1, one, 2l, q); nb = SingleSegCouple(2, 2, one, 2l, q)
    pp = (ContactCouple([d', d], [2 -2; 0 0; 0 0], 2l) + ContactCouple([d, d'], [-2 2; 0 0; 0 0], 2l)) / sqrt(4pi)
    pref = 4 / (sqrt(20) * N^2)
    [pref * (aa + bb - ab - N / (4pi) * (na + nb)), pref * (-2pp)]
end
insertion_terms(N, ins::Symbol, l; cache="") =
    ins == :tensorG ? with_obs_r2(() -> tensor_g_parts(N, l), 1.0) : ope3_terms(N, ins, l; cache)

function _canonical_future(path)
    p = abspath(path); tail = String[]
    while !ispath(p)
        pushfirst!(tail, basename(p)); q = dirname(p); q == p && error("Cannot resolve path parent"); p = q
    end
    normpath(joinpath(realpath(p), tail...))
end
function _contains_path(parent, child)
    p = Sys.iswindows() ? lowercase(parent) : parent; c = Sys.iswindows() ? lowercase(child) : child
    c == p || startswith(c, p * (Sys.iswindows() ? "\\" : "/"))
end
_parse_ls(x) = x == "all" ? nothing : x isa AbstractString ? sort(unique(parse.(Int, split(x, ',')))) : sort(unique(Int.(x)))

"""
    run_stageb(c, o) -> stageB_manifest path (or nothing for a window-only / probe job)

`c`: pipeline config of the finished run (pipeline_config, g resolved); `o`: the [stageb] table.
"""
function run_stageb(c, o)
    N = Int(c["N"]); g = Float64(c["g"]); run = _canonical_future(c["out"])
    out = _canonical_future(o["out"]); o3dir = _canonical_future(o["ope3"]); sector = String(get(o, "sector", "half"))
    sector in ("half", "11") || error("stageb.sector must be half or 11")
    any(p -> _contains_path(p, out) || _contains_path(out, p), (run, o3dir)) && error("Stage B output must not overlap production/OPE directory trees")
    cap = Int(get(o, "max_nev", 384))
    raw_dmax = Float64(get(o, "raw_delta_max", 6.8)); raw_dmax >= 6.6 || error("KK raw basis must reach at least Delta6.6")
    src_dmax = Float64(get(o, "source_delta_max", 5.0)); tgt_dmax = Float64(get(o, "target_delta_max", 6.6))
    kk_lmax = Int(get(o, "lmax", 4)); 0 <= kk_lmax <= 4 || error("stageb.lmax must be 0..4"); raw_lmax = kk_lmax + 1
    step = String(get(o, "step", "all")); step in ("probe", "windows", "all") || error("stageb.step must be probe, windows or all")
    wl = _parse_ls(get(o, "windows", "all")); window_ls = wl === nothing ? collect(0:raw_lmax) : wl
    all(L -> 0 <= L <= raw_lmax, window_ls) || error("stageb.windows outside 0:$raw_lmax")
    postchunk = Int(get(o, "post_chunk", 8)); thr = Float64(get(o, "threshold", 0.25))
    inss = Symbol.(get(o, "insertions", ["gamma2", "tensorG", "singlet"]))
    tol = 1e-10; guard = 1e-5; act = Threads.nthreads(); image_bytes = Float64(get(o, "image_bytes", 4e9))
    partial_ckpt = Bool(get(o, "partial_checkpoints", true))
    stop_k = Int(get(o, "stop_after_k", 0)); stop_k >= 0 || error("stageb.stop_after_k must be >= 0")
    stop_k > 0 && !partial_ckpt && error("stageb.stop_after_k needs partial_checkpoints = true")
    stop_k > 0 && step != "windows" && error("stageb.stop_after_k is only for step = \"windows\"")
    nextra = Int(get(o, "extra_states", 4)); 0 <= nextra <= 16 || error("stageb.extra_states must be 0..16")
    # Ritz value error <= residual norm, so a residual tolerance of guard/10 resolves a margin > guard
    ctol = Float64(get(o, "complement_tol", 1e-6)); tol <= ctol <= guard / 10 || error("stageb.complement_tol must lie in [$tol, $(guard / 10)]")
    final_polish = Float64(get(o, "final_polish_trigger", 1e-9)); 0 <= final_polish <= 1e-7 || error("stageb.final_polish_trigger must lie in [0, 1e-7]")
    prelim_path = String(get(o, "prelim_top_window", "")); prelim = !isempty(prelim_path); prelim_L = raw_lmax
    prelim_complement = Bool(get(o, "prelim_complement", false))
    if prelim
        step == "all" || error("stageb.prelim_top_window needs step = \"all\"")
        kk_lmax >= 1 || error("stageb.prelim_top_window needs lmax >= 1 (the current control uses the L1 window)")
        isdir(prelim_path) || error("stageb.prelim_top_window is not a checkpoint directory: $prelim_path")
        prelim_path = realpath(prelim_path)
    end
    prelim_complement && !prelim && error("stageb.prelim_complement needs prelim_top_window")
    cache = joinpath(out, "operator_terms"); mkpath(out)
    # default: 85 % of the Slurm allocation (as every production job set it), else 280e9
    memory_limit = Float64(get(o, "memory_limit_bytes", haskey(ENV, "SLURM_MEM_PER_NODE") ? parse(Float64, ENV["SLURM_MEM_PER_NODE"]) * 2^20 * 0.85 : 280e9))
    _memlog("start")
    fit = _checked_load(joinpath(run, "kk", "calibration.ckpt"))
    report = TOML.parsefile(joinpath(o3dir, "ope3.toml"))
    Int(report["N"]) == N && abs(report["g"] - g) < 1e-12 || error("OPE normalization is from another N/g")
    abs(fit.alpha - report["alpha"]) < 1e-8 && abs(fit.E0 - report["E0"]) < 1e-8 || error("Integer KK and OPE calibration differ")
    cut = fit.E0 + raw_dmax / fit.alpha; need = cut + max(1e-4, 100guard)
    Delta(E) = fit.alpha * (E - fit.E0)
    lib = SegLibrary(builder=:tiled, tile_width=get(c, "tile", 64), workers=get(c, "workers", act), opmat_threads=get(c, "native", 8))

    if sector == "half"
        oldseg = joinpath(o3dir, "half", "segment.ckpt")
        basis_source = isdir(oldseg) ? String(_ckpt_sha(oldseg)) : "fresh_half_segment_N$N"
        make_seg = isdir(oldseg) ? (() -> _checked_load(oldseg).space) : (() -> build_half_segment(N))
        qn = [2N, 0, 2]
    else
        spaces, bsha0 = ope3_spaces(c)
        seg11 = spaces isa SpaceStore ? getspace!(spaces, 1) : spaces[2]
        spaces = nothing; basis_source = String(bsha0); qn = [2N, 0, 0]
        make_seg = () -> seg11
    end
    binding = Dict("version" => STAGEB_VERSION,
        "calibration_sha" => String(_ckpt_sha(joinpath(run, "kk", "calibration.ckpt"))),
        "normalization_sha" => file_digest(joinpath(o3dir, "ope3.toml")),
        "basis_source" => basis_source, "N" => N, "g" => g, "sector" => sector, "raw_dmax" => raw_dmax,
        "source_run" => run, "ope3_source" => o3dir)
    binding_path = joinpath(out, "input_binding.toml")
    if isfile(binding_path)
        TOML.parsefile(binding_path) == binding || error("StageB input binding changed; choose another output directory")
    else
        tmpb = binding_path * ".tmp.$(getpid())"; open(io -> TOML.print(io, binding), tmpb, "w")
        try
            mv(tmpb, binding_path)
        catch
            rm(tmpb; force=true)
            TOML.parsefile(binding_path) == binding || error("StageB input binding changed (concurrent job)")
        end
    end
    mkpath(joinpath(out, "jobs"))
    open(joinpath(out, "jobs", "$(step)_L$(join(window_ls, '-'))_$(get(ENV, "SLURM_JOB_ID", "nojob"))_$(getpid()).toml"), "w") do io
        TOML.print(io, Dict("step" => step, "windows" => window_ls, "lmax" => kk_lmax, "memory_limit_bytes" => memory_limit,
            "post_chunk" => postchunk, "threads" => act, "host" => gethostname(), "slurm_job" => get(ENV, "SLURM_JOB_ID", ""),
            "revision" => STAGEB_REVISION, "partial_checkpoints" => partial_ckpt, "stop_after_k" => stop_k,
            "extra_states" => nextra, "complement_tol" => ctol, "final_polish_trigger" => final_polish,
            "prelim_top_window" => prelim_path, "prelim_complement" => prelim_complement,
            "code" => Dict(string(k) => v for (k, v) in pairs(producer_info()))))
    end
    segpath = joinpath(out, "segment.ckpt")
    if !isdir(segpath)
        seg0 = make_seg()
        try
            _atomic_save(segpath, (N=N, sector=sector, space=seg0, basis_source=basis_source))
        catch
            isdir(segpath) || rethrow()           # a parallel job saved the shared segment first
        end
        seg0 = nothing; GC.gc()
    end
    make_seg = nothing
    savedseg = _checked_load(segpath)
    savedseg.N == N && savedseg.sector == sector && savedseg.basis_source == basis_source || error("Output segment identity mismatch")
    seg = savedseg.space; savedseg = nothing; GC.gc()           # every job computes with the saved copy
    bsha = String(_ckpt_sha(segpath)); ss = [seg, seg]
    maps, _ = segment_ph_maps(seg, N)
    cs = Dict(L => BuildCompSpace(ss, qn, 2L) for L in 0:raw_lmax)
    sector == "half" && haskey(HALF_DIMS, N) && ((cs[0].dim, cs[1].dim) == HALF_DIMS[N] || error("StageB half-sector dimension regression failed"))
    tpl = hamiltonian_template(N); sH, fH = segops_shared!(lib, ss, tpl.template); sC, fC = segops_shared!(lib, ss, tpl.c2)
    println("STAGEB_SETUP N=$N sector=$sector seglib_gb=$(resident_bytes(lib) / 1e9) dims=$([cs[L].dim for L in 0:raw_lmax]) lmax=$kk_lmax step=$step windows=$window_ls threads=$act"); flush(stdout)
    step == "probe" && (println("STAGEB_PROBE_DONE N=$N sector=$sector"); return nothing)
    windowpath(L) = joinpath(out, "windows", "L$L.ckpt")
    projectionpath(L) = joinpath(out, "kk", "L$L.ckpt")
    partialdir = joinpath(out, "windows", "partial")
    partialpath(L, k) = joinpath(partialdir, "L$(L)_k$(lpad(k, 3, '0')).ckpt")
    function partial_ks(L)
        isdir(partialdir) || return Int[]
        ks = Int[]
        for f in readdir(partialdir)
            m = match(Regex("^L$(L)_k(\\d+)\\.ckpt\$"), f); m === nothing || push!(ks, parse(Int, m[1]))
        end
        sort(ks)
    end
    drop_partials(L; keep=0) = foreach(k -> k == keep || rm(partialpath(L, k); recursive=true, force=true), partial_ks(L))
    # The window of a step that does not yet guard `need`: exactly the (energy, vectors) the loop carries
    # into the next step, with the k and next k of the loop.  Older partials of L go only after it is complete.
    function save_partial(L, E, V, k, next_k, frontier, c2, rh, rc)
        path = partialpath(L, k)
        ispath(path) && rm(path; recursive=true, force=true)      # stale copy of the same step from a crashed run
        _atomic_save(path, (energy=E, vectors=V, L=L, k=k, next_k=next_k, dim=size(V, 1), g=g, basis_sha=bsha,
            raw_dmax=raw_dmax, frontier=frontier, required_energy=need, c2=c2, rh=rh, rc=rc,
            input_binding=binding, code=producer_info()))
        drop_partials(L; keep=k)
        _memlog("partial_L$(L)_k$k")
        println("STAGEB_PARTIAL_SAVED L=$L states=$(length(E)) k=$k next_k=$next_k path=$path"); flush(stdout)
    end
    # Identity of a partial window: L, dimension, g, segment basis, raw cut and input binding.
    function check_partial(p, L, path)
        p.L == L && p.dim == cs[L].dim && p.g == g && p.basis_sha == bsha && p.raw_dmax == raw_dmax && p.input_binding == binding &&
            size(p.vectors) == (p.dim, length(p.energy)) && issorted(p.energy) && length(p.c2) == length(p.energy) ||
            error("Partial StageB window identity mismatch: $path")
        p
    end
    function load_partial(L)
        ks = partial_ks(L); isempty(ks) && return nothing
        path = partialpath(L, last(ks)); p = check_partial(_checked_load(path), L, path)
        p.next_k > p.k || error("Partial StageB window has no next step: $path")
        println("STAGEB_PARTIAL_RESUME L=$L states=$(length(p.energy)) k=$(p.k) next_k=$(p.next_k) frontier=$(p.frontier) path=$path"); flush(stdout)
        p
    end
    # prelim_top_window: the partial window that stands in for the raw window of the top L
    usepartial(L) = prelim && L == prelim_L
    load_window(L) = usepartial(L) ? check_partial(_checked_load(prelim_path), L, prelim_path) : _checked_load(windowpath(L))
    window_sha(L) = String(_ckpt_sha(usepartial(L) ? prelim_path : windowpath(L)))
    wmeta = Dict{Int,Any}()                                      # L => (energy, c2, frontier, maxrh, sha, k)

    # Refuse a solve step before allocating when the peak would exceed the limit.  Peak columns of D:
    # max(copy of the prior into the grown window, grown window + Krylov basis + Ritz vectors).
    function guard_mem(L, m, kadd)
        D = cs[L].dim; a = max(0, min(kadd, D - m)); kd = min(D, krylovdim(max(a, 1)))
        cols = max(2m + a, (m + a) + kd + 2a + 6)
        base = max(0.0, _memlog("guard_L$(L)_m$(m)").rss - 8.0 * D * m)
        estimate = base + 8.0 * D * cols
        println("STAGEB_MEMORY_GUARD L=$L m=$m add=$a peak_columns=$cols estimate_gb=$(estimate / 1e9) limit_gb=$(memory_limit / 1e9)"); flush(stdout)
        estimate <= memory_limit || error("STAGEB_MEMORY_BOUND L$L m=$m add=$a requires estimated $(estimate / 1e9) GB; no unsafe allocation attempted")
    end
    # Prior window: the run's own half window (ope3) or (1,1) window, used only if it is gated and a prefix.
    function prior_window(L, H, C)
        w = nothing
        if sector == "half"
            path = joinpath(o3dir, "half", "blocks", "L$L", "eig.ckpt")
            if isdir(path)
                old = _checked_load(path)
                old.g == g && old.segment_sha == basis_source && old.dim == cs[L].dim || error("Existing half-window identity mismatch")
                w = (energy=old.energy, vectors=old.vectors)
            end
        else
            a = source_window(run, 1, 1, L); b = target_window(run, 1, 1, L)
            w0 = a === nothing ? b : b === nothing || length(a.E) >= length(b.E) ? a : b
            if w0 !== nothing
                w0.basis_sha == basis_source && w0.dim == cs[L].dim || error("Existing integer-window identity mismatch")
                w = (energy=w0.E, vectors=w0.V)
            end
        end
        w === nothing && return nothing
        rh = post_h(H, w.vectors, w.energy; workers=act, chunk=postchunk); cc, rc = post_c(C, w.vectors; workers=act, chunk=postchunk)
        try
            window_gate("StageB prior L$L", w.energy, w.vectors; rh, rc, c2=cc)
            # with extra states the first grown step deflates against the prior and stops if a new state
            # lies below it (lean_target_window!), so the separate complement solve is skipped
            nextra == 0 && (lean_complement_check(H, cs[L].dim, w.energy, w.vectors, act, ctol, 77100 + L).min_margin > guard || return nothing)
            return w
        catch err
            println("STAGEB_PRIOR_NOT_REUSED L=$L reason=$(sprint(showerror, err))"); flush(stdout)
            return nothing
        end
    end
    # Grow, gate, PH-resolve and save the raw window of L.
    function solve_window(L)
        path = windowpath(L); D = cs[L].dim
        H = absorb!(set_g!(BuildCompOperator(cs[L], tpl.template, sH; num_th=act), tpl, g), fH)
        C = absorb!(BuildCompOperator(cs[L], tpl.c2, sC; num_th=act), fC)
        pw = partial_ckpt ? load_partial(L) : nothing
        if pw === nothing
            box = Ref{Any}(prior_window(L, H, C))
            k = min(D, cap, max(24, box[] === nothing ? 0 : length(box[].energy)))
        else                                                  # the state of the loop after step pw.k, as saved
            box = Ref{Any}((energy=pw.energy, vectors=pw.vectors)); k = pw.next_k; pw = nothing
        end
        while true
            retry_fresh = false
            m = box[] === nothing ? 0 : length(box[].energy)
            if m >= k
                E = box[].energy; V = box[].vectors
            else
                kx = min(D, k + nextra)                       # the seed still depends on k only
                guard_mem(L, m, kx - m)
                w = lean_target_window!(box, H, D, kx, 88000 + 100L + k, act, tol)
                E = w.energy; V = w.vectors; w = nothing
            end
            box[] = nothing
            zeroP = zeros(Int, length(E))
            resolve_c2!(E, V, zeroP, C; tol=1e-6, workers=act); lean_polish_c2!(E, V, zeroP, H, C; workers=act, chunk=postchunk)
            order = sortperm(E); E = E[order]; permute_cols!(V, order)
            rh = post_h(H, V, E; workers=act, chunk=postchunk); cc, rc = post_c(C, V; workers=act, chunk=postchunk)
            window_gate("StageB half/raw L$L", E, V; rh, rc, c2=cc)
            n = length(E)
            if nextra > 0 && n > k && n < D                   # certificate from the solved states above the top
                kc = certified_top(E, k, guard); cert = "extra_states"
                frontier = kc == 0 ? -Inf : E[kc]; margin = kc == 0 ? -Inf : E[kc+1] - E[kc]
            else
                kc = n; cert = n >= D ? "whole_sector" : "deflated_complement"
                frontier = window_frontier(E, zeros(Int, n), Dict(0 => D); guard); margin = Inf
            end
            println("STAGEB_WINDOW N=$N sector=$sector L=$L dim=$D states=$n certified=$kc DeltaTop=$(Delta(E[max(kc, 1)])) frontier=$frontier need=$need"); flush(stdout)
            if frontier > need
                cert == "deflated_complement" && (margin = lean_complement_check(H, D, E, V, act, ctol, 99000 + L).min_margin)
                println("STAGEB_CERTIFICATE L=$L kind=$cert margin=$margin guard=$guard"); flush(stdout)
                if margin > guard
                    if kc < n                                 # drop the extra states above the certified top
                        W = V[:, 1:kc]; V = nothing; GC.gc(); V = W; W = nothing
                        E = E[1:kc]; cc = cc[1:kc]
                    end
                    # near-degenerate pairs of different C2 (gap ~1e-6) keep a C2 admixture ~tol/gap that depends
                    # on the Lanczos path; polish the certified window below the per-step trigger
                    if final_polish > 0
                        lean_polish_c2!(E, V, zeros(Int, length(E)), H, C; trigger=final_polish, workers=act, chunk=postchunk)
                        cc, rc = post_c(C, V; workers=act, chunk=postchunk)
                    end
                    plan = comp_parity(cs[L], maps, maps); pseg, pres = ph_resolve!(E, V, cc, plan)
                    rh = post_h(H, V, E; workers=act, chunk=postchunk); cc, rc = post_c(C, V; workers=act, chunk=postchunk)
                    gate = window_gate("StageB parity-resolved L$L", E, V; rh, rc, c2=cc)
                    # PH diagonalization rotates within the same retained span;
                    # the complement-prefix certificate is invariant under this unitary.
                    frontier = window_frontier(E, zeros(Int, length(E)), Dict(0 => D); guard)
                    frontier > need || error("Post-PH window frontier no longer guards requested basis")
                    _atomic_save(path, (energy=E, vectors=V, c2=cc, rh=rh, rc=rc, dim=D, g=g, basis_sha=bsha,
                        qn=qn, sector=sector, N=N, raw_dmax=raw_dmax, frontier=frontier, required_energy=need,
                        pseg=pseg, parity_residual=pres, independent_min_margin=margin, certificate=cert, complement_tol=ctol,
                        colsha=column_shas(V), gate=gate, calibration=fit, input_binding=binding, code=producer_info()))
                    _memlog("saved_L$L")
                    drop_partials(L)
                    return nothing
                end
                println("STAGEB_EXTEND_TERMINAL_CLUSTER L=$L margin=$margin"); flush(stdout)   # only the deflated check can fail
                retry_fresh = true
            end
            k >= min(cap, D) && error("INCOMPLETE_STAGEB_WINDOW L$L cap=$cap reached")
            next_k = min(cap, D, max(k + 8, ceil(Int, 1.5k)))
            partial_ckpt && (retry_fresh ? drop_partials(L) : save_partial(L, E, V, k, next_k, frontier, cc, rh, rc))
            nstates = length(E)
            box[] = retry_fresh ? nothing : (energy=E, vectors=V)
            E = nothing; V = nothing; cc = nothing; rh = nothing; rc = nothing
            k = next_k
            stop_k > 0 && !retry_fresh && nstates >= stop_k && throw(StageBPartialStop(L, nstates, k))
        end
    end

    stopped = nothing
    for L in 0:raw_lmax
        step == "windows" && !(L in window_ls) && continue           # another job owns this L
        path = windowpath(L)
        if usepartial(L)
            ispath(path) && error("prelim_top_window: a window L$L exists in $out; a partial window may not replace it")
            p = load_window(L)
            window_gate("StageB prelim partial L$L", p.energy, p.vectors; rh=p.rh, rc=p.rc, c2=p.c2)
            wmeta[L] = (energy=copy(p.energy), c2=copy(p.c2), frontier=p.frontier, maxrh=maximum(p.rh), sha=window_sha(L), k=p.k)
            println("STAGEB_PRELIM_PARTIAL_WINDOW N=$N sector=$sector L=$L states=$(length(p.energy)) k=$(p.k) frontier=$(p.frontier) ",
                "frontier_Delta=$(Delta(p.frontier)) DeltaTop=$(Delta(last(p.energy))) need=$need sha=$(wmeta[L].sha) path=$prelim_path"); flush(stdout)
            p = nothing; GC.gc(); continue
        end
        if isdir(path)
            w = _checked_load(path)
            w.g == g && w.basis_sha == bsha && w.raw_dmax == raw_dmax && w.frontier > need && w.input_binding == binding ||
                error("Cached StageB window identity/frontier mismatch")
            window_gate("StageB saved L$L", w.energy, w.vectors; rh=w.rh, rc=w.rc, c2=w.c2)
            println("STAGEB_WINDOW_REUSE N=$N sector=$sector L=$L states=$(length(w.energy))"); flush(stdout)
            w = nothing; GC.gc(); continue
        end
        L in window_ls || error("STAGEB_MISSING_WINDOW L$L is neither in stageb.windows nor saved in $out")
        lockdir = joinpath(out, "windows", ".solving_L$L"); mkpath(dirname(lockdir))
        try
            mkdir(lockdir)
        catch
            error("Another job is solving L$L ($lockdir exists; remove it only if that job is dead)")
        end
        try
            solve_window(L)
        catch err
            err isa StageBPartialStop || rethrow()
            stopped = err
        finally
            rm(lockdir; force=true, recursive=true)
        end
        GC.gc()
        stopped === nothing || break
    end
    if stopped !== nothing
        println("STAGEB_PARTIAL_STOP N=$N sector=$sector L=$(stopped.L) states=$(stopped.states) next_k=$(stopped.next_k) stop_after_k=$stop_k"); flush(stdout)
        return nothing
    end
    step == "windows" && (println("STAGEB_WINDOWS_DONE N=$N sector=$sector L=$window_ls"); flush(stdout); return nothing)
    for L in 0:raw_lmax                                           # prelim: metadata of the complete windows
        (!prelim || haskey(wmeta, L)) && continue
        w = _checked_load(windowpath(L))
        wmeta[L] = (energy=copy(w.energy), c2=copy(w.c2), frontier=w.frontier, maxrh=maximum(w.rh), sha=window_sha(L), k=length(w.energy))
        w = nothing; GC.gc()
    end

    # Calibrate only the overall PH-sector sign against a known raw multiplet;
    # the lowering-generator coefficients remain exactly the integer-run fit.
    w0 = _checked_load(windowpath(0)); anchor_c2 = sector == "half" ? 2 : 5; anchor_P = sector == "half" ? -1 : 1
    ix = findall(x -> abs(x - anchor_c2) < 1e-5, w0.c2); isempty(ix) && error("Missing parity anchor")
    j = ix[argmin(w0.energy[ix])]; integer0 = source_window(run, 0, 0, 0)
    ii = findall(x -> abs(x - anchor_c2) < 1e-5, integer0.C2); isempty(ii) && error("Missing integer parity anchor")
    abs(w0.energy[j] - minimum(integer0.E[ii])) <= 1e-6 || error("Half/integer parity-anchor energy mismatch")
    vacfactor = anchor_P * w0.pseg[j]; integer0 = nothing; w0 = nothing; GC.gc()
    raw_rows = Any[]; certificates = Dict{String,Any}()
    for L in 0:raw_lmax
        if usepartial(L)                                          # a partial window has no PH labels: pseg = P = 0
            m = wmeta[L]; certificates["L$L"] = "prelim_partial_window"
            for i in eachindex(m.energy)
                push!(raw_rows, (N, sector, L, i, m.energy[i], Delta(m.energy[i]), m.c2[i], 0, m.sha, 0))
            end
            continue
        end
        w = _checked_load(windowpath(L)); wsha = String(_ckpt_sha(windowpath(L)))
        # windows of v0.3.0 and earlier were certified by the deflated complement check at the solver tolerance
        certificates["L$L"] = hasproperty(w, :certificate) ? "$(w.certificate) (complement_tol $(w.complement_tol))" : "deflated_complement (complement_tol $tol)"
        for i in eachindex(w.energy)
            push!(raw_rows, (N, sector, L, i, w.energy[i], Delta(w.energy[i]), w.c2[i], w.pseg[i], wsha, w.pseg[i] * vacfactor * (-1)^L))
        end
        w = nothing; GC.gc()
    end
    _csv(joinpath(out, "raw_spectrum_Lle$(raw_lmax).csv"), ["N","sector","L","raw_index","energy","Delta","C2","pseg","window_sha","P"], raw_rows)
    control = Dict{String,Any}()
    if sector == "half"
        # the current J of the half sector must sit at Delta = 2 with P = +1 and reproduce ope3's phi-J element
        pick(E, C, cc) = (ii = findall(x -> abs(x - cc) < 1e-5, C); isempty(ii) && error("Missing control irrep"); ii[argmin(E[ii])])
        n0 = _checked_load(windowpath(0)); np = pick(n0.energy, n0.c2, 2)
        s = (V=n0.vectors[:, np:np], sha=String(_ckpt_sha(windowpath(0))) * "|phi$np"); phiE = n0.energy[np]; n0 = nothing; GC.gc()
        n1 = _checked_load(windowpath(1)); nj = pick(n1.energy, n1.c2, 3)
        current_P = n1.pseg[nj] * vacfactor * (-1); current_Delta = Delta(n1.energy[nj])
        current_P == 1 && abs(current_Delta - 2.0) < 1e-6 || error("StageB current parity/Ward energy-scale anchor failed")
        t = (V=n1.vectors[:, nj:nj], sha=String(_ckpt_sha(windowpath(1))) * "|J$nj"); curE = n1.energy[nj]; n1 = nothing; GC.gc()
        o0 = _checked_load(joinpath(o3dir, "half", "blocks", "L0", "eig.ckpt")); op = pick(o0.energy, o0.c2, 2); o0 = nothing
        o1 = _checked_load(joinpath(o3dir, "half", "blocks", "L1", "eig.ckpt")); oj = pick(o1.energy, o1.c2, 3); o1E = o1.energy; o1 = nothing; GC.gc()
        refp = joinpath(o3dir, "half", "edges", "half_gamma2_r1_phi_L1.ckpt"); oldedge = _checked_load(refp)
        oldedge.scols == [op] && size(oldedge.overlaps[1], 1) == length(o1E) || error("Old current-control edge/window identity mismatch")
        # v0.2.8 ope3 edges name their windows by checkpoint SHA and g (A3); older edges carry neither
        oid = oldedge.identity
        !hasproperty(oid, :g) || (oid.g == g && oid.src == String(_ckpt_sha(joinpath(o3dir, "half", "blocks", "L0", "eig.ckpt"))) &&
            oid.tgt == String(_ckpt_sha(joinpath(o3dir, "half", "blocks", "L1", "eig.ckpt")))) || error("Old current-control edge built from other half windows or g")
        abs(curE - o1E[oj]) < 1e-6 || error("Current-control energy changed")
        id = (tag="StageB_independent_raw_phi_J_regression", source=s.sha, target=t.sha, g=g, input_binding=binding,
              reference_sha=String(_ckpt_sha(refp)))
        edge = _ope3_edge(out, "raw_phi_J_control", lib, ss, cs[0], cs[1], ope3_terms(N, :gamma2, 1; cache), 2, s, [1], t, id; workers=act, image_bytes)
        oldabs = abs(oldedge.overlaps[1][oj, 1]); newabs = abs(edge.overlaps[1][1, 1]); bound = 1e-7 * max(1, oldabs)
        abs(newabs - oldabs) <= bound || error("STAGEB_RAW_CURRENT_REGRESSION_FAILED old=$oldabs new=$newabs")
        control = Dict{String,Any}("status" => "pass", "old_abs_ME" => oldabs, "new_abs_ME" => newabs,
            "absolute_difference" => abs(newabs - oldabs), "tolerance" => bound, "current_P" => current_P, "current_Delta" => current_Delta,
            "old_edge_sha" => id.reference_sha, "new_phi_energy" => phiE, "new_current_energy" => curE)
        open(io -> TOML.print(io, control), joinpath(out, "raw_current_control.toml"), "w")
        println("STAGEB_RAW_CURRENT_REGRESSION_PASS N=$N diff=$(abs(newabs - oldabs))"); flush(stdout)
        s = nothing; t = nothing; oldedge = nothing; edge = nothing; GC.gc()
    end

    # ---- KK: lowering edges L -> L-1, L, L+1 of the raw windows, per P_seg sector -----------------
    kt = kk_terms(N; cache); catalog = Any[]; coverage = Any[]; truncated_rows = Any[]
    for li in 0:kk_lmax
        w = _checked_load(windowpath(li)); ids = [i for i in eachindex(w.energy) if w.energy[i] <= cut && round(Int, w.c2[i]) in STAGEB_REPS]
        tls = li == 0 ? [1] : [li - 1, li, li + 1]
        if prelim
            # keep only the sources the frontier guard below accepts for every target window
            lim_of = lf -> wmeta[lf].frontier - max(1e-8, 10 * (maximum(w.rh) + wmeta[lf].maxrh))
            lim = minimum(lim_of(lf) for lf in tls); kept = [i for i in ids if w.energy[i] < lim]
            for i in setdiff(ids, kept)
                push!(truncated_rows, (N, sector, li, i, w.energy[i], Delta(w.energy[i]), w.c2[i], w.pseg[i], lim, Delta(lim),
                    join([lf for lf in tls if !(w.energy[i] < lim_of(lf))], ";")))
            end
            length(kept) < length(ids) && (println("STAGEB_PRELIM_KK_SOURCES_TRUNCATED N=$N L=$li kept=$(length(kept)) removed=$(length(ids) - length(kept)) limit_Delta=$(Delta(lim))"); flush(stdout))
            ids = kept
        end
        isempty(ids) && error("No StageB raw KK sources L$li")
        S = (E=w.energy[ids], V=view(w.vectors, :, ids), C2=w.c2[ids], parity=zeros(Int, length(ids)))
        source_sha = String(_ckpt_sha(windowpath(li))); edges = Any[]
        for lf in tls
            t = lf == li ? w : load_window(lf); atol = max(1e-8, 10 * (maximum(w.rh) + maximum(t.rh)))
            maximum(S.E) < t.frontier - atol || error("StageB KK target frontier insufficient")
            epath = joinpath(out, "kk", "edges", "L$(li)_L$(lf).ckpt"); target_sha = window_sha(lf)
            if isdir(epath)                                  # A2: reuse only an edge of exactly these windows
                old = _checked_load(epath)
                if old.source_sha == source_sha && old.target_sha == target_sha && old.input_binding == binding && old.source_E == S.E
                    push!(edges, old); println("STAGEB_KK_EDGE_REUSE N=$N sector=$sector L$li->$lf"); flush(stdout); t = nothing; continue
                end
                println("STAGEB_KK_EDGE_SUPERSEDED $(_supersede!(epath))"); flush(stdout)
            end
            A = edge_arrays(lib, ss, cs[li], cs[lf], kt, 2, S.V, t.vectors; workers=act, chunk=1)
            mask = [abs(a - b) > 1e-5 for a in t.c2, b in S.C2]
            all(norm(M .* mask) <= 1e-6 * max(1, norm(M)) for M in A.overlaps) || error("StageB generator changes C2")
            edge = (li=li, lf=lf, source_E=S.E, source_C2=S.C2, target_E=t.energy, target_C2=t.c2,
                    overlaps=A.overlaps, image_gram=A.image_gram, energy_atol=atol, target_frontier=t.frontier,
                    target_parities=[0], source_sha=source_sha, target_sha=target_sha, input_binding=binding)
            _atomic_save(epath, edge); push!(edges, edge)
            println("STAGEB_KK_EDGE N=$N sector=$sector L$li->$lf"); flush(stdout); t = nothing; A = nothing; GC.gc()
        end
        modes = Any[]
        for ps in (-1, 1)
            loc = findall(==(ps), w.pseg[ids]); isempty(loc) && continue
            sub = (E=S.E[loc], C2=S.C2[loc], parity=zeros(Int, length(loc)))
            es = [merge(e, (overlaps=[M[:, loc] for M in e.overlaps], image_gram=e.image_gram[:, :, loc])) for e in edges]
            for ch in kk_classify(es, sub, li, fit.coefficients; threshold=thr)
                push!(coverage, (sector, li, ch.C2, ps, ch.status, length(ch.source_ids), length(ch.excluded_ids), join(ch.missing_L, ";")))
                ch.Q === nothing && continue
                globalids = ids[loc[ch.source_ids]]
                for k in axes(ch.Q, 2)
                    q = zeros(length(w.energy)); q[globalids] = ch.Q[:, k]
                    pphys = ps * vacfactor * (-1)^li
                    push!(modes, (label="$(sector)_KK_C2_$(ch.C2)_P$(pphys)_m$(k)", c2=ch.C2, P=pphys, pseg=ps, energy=ch.energies[k],
                                  sigmaE=ch.energy_mixing_residual[k], K2=dot(ch.Q[:, k], ch.K2 * ch.Q[:, k]), coeff=q,
                                  source_sha=source_sha, coverage=ch.status))
                end
            end
        end
        ppath = projectionpath(li)
        saved = isdir(ppath) ? _checked_load(ppath) : nothing
        if saved !== nothing && !(saved.source_sha == source_sha && saved.input_binding == binding && saved.raw_dmax == raw_dmax && saved.threshold == thr)
            println("STAGEB_PROJECTION_SUPERSEDED $(_supersede!(ppath))"); flush(stdout); saved = nothing
        end
        if saved === nothing
            _atomic_save(ppath, (N=N, L=li, sector=sector, modes=modes, source_sha=source_sha,
                calibration=fit, raw_dmax=raw_dmax, threshold=thr, vacuum_raw_factor=vacfactor, input_binding=binding))
        else
            # the saved modes stay authoritative (the OPE below reads them); the reclassification must reproduce them
            [m.label for m in saved.modes] == [m.label for m in modes] && all(abs(a.energy - b.energy) <= 1e-9 for (a, b) in zip(saved.modes, modes)) ||
                error("StageB saved projection L$li differs from the reclassification of its edges")
            modes = saved.modes; println("STAGEB_PROJECTION_REUSE N=$N sector=$sector L$li modes=$(length(modes))"); flush(stdout)
        end
        for m in modes
            push!(catalog, (N, sector, li, m.label, m.c2, m.P, Delta(m.energy), m.sigmaE, m.K2, source_sha, m.coverage))
        end
        w = nothing; S = nothing; edges = nothing; saved = nothing; GC.gc()
    end
    _memlog("kk_done")

    # ---- full-Q OPE: projected scalar sources (Delta <= 5) x projected spin-l targets (Delta <= 6.6) -----
    function projected(L, dmax)
        w = _checked_load(windowpath(L)); p = _checked_load(projectionpath(L))
        p.source_sha == String(_ckpt_sha(windowpath(L))) && p.input_binding == binding || error("Projected/source SHA or input binding mismatch")
        modes = [m for m in p.modes if Delta(m.energy) <= dmax]
        Q = isempty(modes) ? zeros(length(w.energy), 0) : hcat([m.coeff for m in modes]...)
        V = w.vectors * Q; w = nothing; GC.gc()
        norm(V' * V - I) <= 1e-6 || error("StageB projected physical vectors not orthonormal")
        (V=V, sha=bytes2hex(sha256(join(column_shas(V), "|"))), modes=modes)
    end
    src = projected(0, src_dmax); isempty(src.modes) && error("No projected scalar source below Delta$(src_dmax)")
    rows = Any[]; norm_sha = file_digest(joinpath(o3dir, "ope3.toml"))
    for l in 0:kk_lmax
        tgt = projected(l, tgt_dmax)
        isempty(tgt.modes) && continue
        for ins in inss
            id = (tag="stageB_fullQ", N=N, g=g, sector=sector, source_sha=src.sha, target_sha=tgt.sha, input_binding=binding,
                  raw_dmax=raw_dmax, normalization_sha=norm_sha, insertion=ins, rank=l, version=STAGEB_VERSION)
            A = _ope3_edge(out, "$(sector)_$(ins)_L$l", lib, ss, cs[0], cs[l], insertion_terms(N, ins, l; cache), 2l, src, collect(eachindex(src.modes)), tgt, id;
                           workers=act, image_bytes)
            for (i, x) in enumerate(src.modes), (j, z) in enumerate(tgt.modes)
                me1 = A.overlaps[1][j, i]; me2 = length(A.overlaps) > 1 ? A.overlaps[2][j, i] : 0.0
                pf = ins == :singlet ? (1 + x.P * z.P) / 2 : 1.0
                push!(rows, (N, sector, String(ins), l, x.label, x.c2, Delta(x.energy), x.K2, x.sigmaE, x.P, z.label, z.c2, Delta(z.energy), z.K2, z.sigmaE, z.P,
                             me1, me2, pf, src.sha, tgt.sha, "exact_local_fullQ_no_cross_sector_identity_claim"))
            end
            println("STAGEB_OPE N=$N sector=$sector L=$l insertion=$ins sources=$(length(src.modes)) targets=$(length(tgt.modes))"); flush(stdout)
        end
        tgt = nothing; GC.gc()
    end
    _memlog("ope_done")
    _csv(joinpath(out, "xy_blocks_raw.csv"), ["N","sector","insertion","rank","X_label","X_C2","X_Delta","X_K2","X_sigmaE","X_P","Z_label","Z_C2","Z_Delta","Z_K2","Z_sigmaE","Z_P","me_1","me_2","S_even_projection_factor","source_projected_sha","target_projected_sha","identity_status"], rows)
    _csv(joinpath(out, "projected_catalog.csv"), ["N","sector","L","label","C2","P","Delta","sigmaE","K2","source_sha","coverage"], catalog)
    _csv(joinpath(out, "kk_coverage.csv"), ["sector","L","C2","pseg","status","source_count","excluded_count","missing_L"], coverage)
    prelim && _csv(joinpath(out, "prelim_truncated_sources.csv"),
        ["N","sector","L","raw_index","energy","Delta","C2","pseg","limit_E","limit_Delta","short_targets"], truncated_rows)
    status = prelim ? "preliminary_partial_top_window_truncated_KK_sources" : "complete_exact_local_candidate_components"
    man = joinpath(out, "stageB_manifest.toml")
    md = Dict{String,Any}("N" => N, "g" => g, "sector" => sector, "source_run" => run, "ope3_source" => o3dir, "version" => STAGEB_VERSION,
        "revision" => STAGEB_REVISION,
        "raw_basis_Delta_max" => raw_dmax, "projected_source_Delta_max" => src_dmax, "projected_target_Delta_max" => tgt_dmax,
        "spin_max" => kk_lmax, "KK_target_spin_max" => raw_lmax, "insertions" => String.(inss), "threshold" => thr,
        "raw_windows" => Dict("L$L" => count(r -> r[3] == L, raw_rows) for L in 0:raw_lmax), "window_certificate" => certificates,
        "max_nev" => cap, "calibration_coefficients" => fit.coefficients, "alpha" => fit.alpha, "E0" => fit.E0, "basis_sha" => bsha,
        "rows" => length(rows), "status" => status,
        "identity_limit" => "No automatic nearest-energy join to existing integer candidates or previous raw half states",
        "code" => Dict(string(k) => v for (k, v) in pairs(producer_info())))
    if prelim
        m = wmeta[prelim_L]
        md["prelim"] = Dict("partial_window_L" => prelim_L, "partial_path" => prelim_path, "partial_sha" => m.sha, "partial_k" => m.k,
            "partial_states" => length(m.energy), "partial_frontier" => m.frontier, "partial_frontier_Delta" => Delta(m.frontier),
            "need" => need, "need_Delta" => Delta(need), "truncated_KK_sources" => length(truncated_rows),
            "window_frontier_Delta" => Dict("L$L" => Delta(wmeta[L].frontier) for L in 0:raw_lmax),
            "limits" => "partial top window: no PH labels (raw_spectrum pseg = P = 0), no complement certificate unless prelim_complement_L$(prelim_L).toml passes; KK sources above its guard are removed (prelim_truncated_sources.csv), so their projected modes and OPE rows are absent",
            "complement_check" => prelim_complement ? "prelim_complement_L$(prelim_L).toml" : "not run")
    end
    open(io -> TOML.print(io, md), man, "w")
    println("STAGEB_DONE N=$N sector=$sector rows=$(length(rows)) status=$status"); flush(stdout)
    if prelim && prelim_complement                               # after every output is written
        L = prelim_L; D = cs[L].dim; rows = nothing; catalog = nothing; coverage = nothing; GC.gc()
        H = absorb!(set_g!(BuildCompOperator(cs[L], tpl.template, sH; num_th=act), tpl, g), fH)
        p = load_window(L); t0 = time()
        chk = lean_complement_check(H, D, p.energy, p.vectors, act, ctol, 99000 + L); okc = chk.min_margin > guard
        open(joinpath(out, "prelim_complement_L$L.toml"), "w") do io
            TOML.print(io, Dict("L" => L, "states" => length(p.energy), "top_Delta" => Delta(last(p.energy)), "frontier" => p.frontier,
                "min_margin" => chk.min_margin, "guard" => guard, "complement_tol" => ctol, "status" => okc ? "pass" : "FAIL",
                "seed" => 99000 + L, "seconds" => time() - t0, "partial_sha" => wmeta[L].sha))
        end
        println(okc ? "STAGEB_PRELIM_COMPLEMENT_PASS" : "STAGEB_PRELIM_COMPLEMENT_FAIL", " L=$L margin=$(chk.min_margin) guard=$guard"); flush(stdout)
        okc || error("STAGEB_PRELIM_COMPLEMENT_FAIL L$L margin=$(chk.min_margin) (outputs written, marked preliminary)")
    end
    get(o, "rawtarget", false) && run_rawtarget(c, o)
    man
end

"""
    run_rawtarget(c, o)

OPE edges of the Stage B scalar sources onto the FULL raw target windows (every saved raw eigenstate of
L = 0..lmax), so that the analysis can project onto any target basis (rawtarget-v1).
Writes below `o["rawtarget_out"]` (default <stageb out>/rawtarget).  Gates: projecting the raw-target matrix
onto the production target modes reproduces the Stage B edge and its image Gram (rel <= 1e-10); captured
norm never exceeds the full image norm.
"""
function run_rawtarget(c, o)
    N = Int(c["N"]); g = Float64(c["g"]); stage = realpath(o["out"]); sector = String(get(o, "sector", "half"))
    out = abspath(get(o, "rawtarget_out", joinpath(stage, "rawtarget")))
    kk_lmax = Int(get(o, "lmax", 4)); src_dmax = Float64(get(o, "source_delta_max", 5.0)); tgt_dmax = Float64(get(o, "target_delta_max", 6.6))
    inss = String.(get(o, "insertions", ["gamma2", "tensorG", "singlet"]))
    mkpath(joinpath(out, "edges")); mkpath(joinpath(out, "export", "bin"))
    act = Threads.nthreads(); image_bytes = Float64(get(o, "image_bytes", 4e9))
    binding = TOML.parsefile(joinpath(stage, "input_binding.toml"))
    Int(binding["N"]) == N && binding["sector"] == sector && abs(Float64(binding["g"]) - g) < 1e-14 || error("Stage B binding is for another N/sector/g")
    fit = _checked_load(joinpath(binding["source_run"], "kk", "calibration.ckpt"))
    Delta(E) = fit.alpha * (E - fit.E0)
    windowpath(L) = joinpath(stage, "windows", "L$L.ckpt"); projectionpath(L) = joinpath(stage, "kk", "L$L.ckpt")
    segpath = joinpath(stage, "segment.ckpt"); seg = _checked_load(segpath).space; bsha = String(_ckpt_sha(segpath))
    ss = [seg, seg]; qn = sector == "half" ? [2N, 0, 2] : [2N, 0, 0]
    cs = Dict(L => BuildCompSpace(ss, qn, 2L) for L in 0:kk_lmax)
    lib = SegLibrary(builder=:tiled, tile_width=get(c, "tile", 64), workers=get(c, "workers", act), opmat_threads=get(c, "native", 8))
    cache = joinpath(out, "operator_terms")
    function window(L)
        w = _checked_load(windowpath(L)); sha = String(_ckpt_sha(windowpath(L)))
        w.g == g && w.basis_sha == bsha || error("window L$L identity mismatch (g/basis)")
        size(w.vectors, 1) == cs[L].dim || error("window L$L dimension mismatch")
        norm(w.vectors' * w.vectors - I) <= 1e-8 || error("window L$L not orthonormal")
        w, sha
    end
    function projected(w, wsha, L, dmax)
        p = _checked_load(projectionpath(L))
        p.source_sha == wsha || error("projection/window SHA mismatch L$L")
        modes = [m for m in p.modes if Delta(m.energy) <= dmax]
        Q = isempty(modes) ? zeros(length(w.energy), 0) : hcat([m.coeff for m in modes]...)
        V = w.vectors * Q
        norm(V' * V - I) <= 1e-6 || error("projected vectors not orthonormal")
        (V=V, Q=Q, sha=bytes2hex(sha256(join(column_shas(V), "|"))), modes=modes)
    end
    w0, w0sha = window(0); src = projected(w0, w0sha, 0, src_dmax); isempty(src.modes) && error("no scalar sources")
    src_ident = (window_sha=w0sha, projection_sha=String(_ckpt_sha(projectionpath(0))), labels=Tuple(m.label for m in src.modes))
    src = (V=src.V, sha=src.sha, modes=src.modes); w0 = nothing; GC.gc()
    println("RAWTARGET_START N=$N sector=$sector g=$g sources=$(length(src.modes)) source_sha=$(src.sha) threads=$act"); flush(stdout)
    meta_edges = Any[]; worst_proj = 0.0; worst_gram = 0.0
    for l in 0:kk_lmax
        w, wsha = window(l); tgt = projected(w, wsha, l, tgt_dmax); Tw = (V=w.vectors, sha=wsha)
        for ins in inss
            name = "$(sector)_$(ins)_L$(l)"
            old = _checked_load(joinpath(stage, "edges", name * ".ckpt"))
            src_bitwise = old.identity.source_sha == src.sha; tgt_bitwise = old.identity.target_sha == tgt.sha
            size(old.overlaps[1]) == (size(tgt.Q, 2), length(src.modes)) || error("$name: original edge shape differs")
            ident = (tag="stageB_rawtarget", version="rawtarget-v2", N=N, g=g, sector=sector, insertion=ins, rank=l,
                     source=src_ident, target_window_sha=wsha, target_count=size(w.vectors, 2), segment_sha=bsha,
                     terms_def=(ins == "tensorG" ? "Tcc.tensor_g_parts" : "Tcc.ope3_terms") * "|$(ins)|r$(l)|N$(N)")
            e = _ope3_edge(out, name, lib, ss, cs[0], cs[l], insertion_terms(N, Symbol(ins), l; cache), 2l, src, collect(eachindex(src.modes)), Tw, ident;
                           workers=act, image_bytes)
            dp = maximum(norm(tgt.Q' * e.overlaps[p] - old.overlaps[p]) / max(norm(old.overlaps[p]), 1e-300) for p in eachindex(e.overlaps))
            dg = norm(e.image_gram - old.image_gram) / max(norm(old.image_gram), 1e-300)
            (dp <= 1e-10 && dg <= 1e-10) || error("$name: regression vs original Stage B edge failed (proj $dp, gram $dg)")
            for p in eachindex(e.overlaps), s in axes(e.overlaps[p], 2)
                sum(abs2, view(e.overlaps[p], :, s)) <= e.image_gram[p, p, s] * (1 + 1e-10) + 1e-14 || error("$name: captured > image norm")
            end
            worst_proj = max(worst_proj, dp); worst_gram = max(worst_gram, dg)
            bin = joinpath("bin", "rawtarget_$(name).f64"); path = joinpath(out, "export", bin)
            open(path * ".partial", "w") do io
                for M in e.overlaps; write(io, htol.(vec(Matrix{Float64}(M)))); end
            end
            mv(path * ".partial", path; force=true)
            push!(meta_edges, Dict("name" => name, "insertion" => ins, "rank" => l, "nt" => size(w.vectors, 2), "ns" => length(src.modes),
                "parts" => length(e.overlaps), "bin" => bin, "bin_sha256" => bytes2hex(open(sha256, path)),
                "target_window_sha" => wsha, "target_E" => collect(Float64, w.energy), "target_C2" => collect(Float64, w.c2),
                "source_sha" => src.sha, "source_labels" => [m.label for m in src.modes],
                "regression_projection_rel" => dp, "regression_gram_rel" => dg,
                "source_sha_bitwise_equal_original" => src_bitwise, "target_sha_bitwise_equal_original" => tgt_bitwise))
            println("RAWTARGET_EDGE N=$N $name nt=$(size(w.vectors, 2)) ns=$(length(src.modes)) proj_rel=$(dp) gram_rel=$(dg)"); flush(stdout)
        end
        w = nothing; Tw = nothing; tgt = nothing; GC.gc()
    end
    meta = Dict("exporter" => "Tcc.run_rawtarget rawtarget-v2", "N" => N, "sector" => sector, "g" => g, "segment_sha" => bsha,
                "tcc_version" => string(TCC_VERSION), "stageB_dir" => stage, "edges" => meta_edges)
    open(io -> write(io, _json(meta)), joinpath(out, "export", "rawtarget_meta.json"), "w")
    open(io -> TOML.print(io, Dict("status" => "done", "N" => N, "sector" => sector, "g" => g, "edges" => length(meta_edges),
        "worst_projection_regression_rel" => worst_proj, "worst_gram_regression_rel" => worst_gram, "stageB_dir" => stage,
        "code" => Dict(string(k) => v for (k, v) in pairs(producer_info())))), joinpath(out, "rawtarget_manifest.toml"), "w")
    println("RAWTARGET_DONE N=$N edges=$(length(meta_edges)) worst_proj=$(worst_proj) worst_gram=$(worst_gram)")
    out
end

_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
_json(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
_json(x::AbstractVector) = "[" * join(_json.(x), ",") * "]"
_json(x::AbstractDict) = "{" * join([_json(string(k)) * ":" * _json(v) for (k, v) in x], ",") * "}"
_json(x::NamedTuple) = _json(Dict(string(k) => v for (k, v) in pairs(x)))
_json(x::Symbol) = _json(string(x))
