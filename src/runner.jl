# General-N spectrum runner (Tcc_v0.0.1).
#   * one segment-operator library per run: H, C2 and all (sA,sB) pairs share operators;
#   * diagonal blocks (s,s,L) can be solved per segment-exchange parity in half-length space;
#   * every block result is an atomic, checksummed checkpoint; a restart skips finished blocks.
# Physics (H, C2, phase convention, basis) is the frozen Zhou v0.0.5 definition.

function _atomic_save(path, payload)
    ispath(path) && error("refusing to overwrite $path")
    mkpath(dirname(path)); tmp = mktempdir(dirname(path); prefix=".pending-")
    try
        f = joinpath(tmp, "payload.jls"); open(io -> serialize(io, payload), f, "w")
        write(joinpath(tmp, "sha256.txt"), open(io -> bytes2hex(sha256(io)), f) * "\n")
        mv(tmp, path)
    finally
        ispath(tmp) && rm(tmp; recursive=true, force=true)
    end
    path
end

function _checked_load(path)
    f = joinpath(path, "payload.jls")
    strip(read(joinpath(path, "sha256.txt"), String)) == open(io -> bytes2hex(sha256(io)), f) ||
        error("checksum mismatch $path")
    open(deserialize, f)
end

"""Order pairs so that every segment space is released right after its last use."""
function pair_schedule(blocks)
    pairs = unique([(b[1], b[2]) for b in blocks])
    sort!(pairs)
    lastuse = Dict{Int,Int}()
    for (k, (a, b)) in enumerate(pairs); lastuse[a] = k; lastuse[b] = k; end
    pairs, lastuse
end

"""Column chunk so that three D x chunk Float64 work arrays stay within `bytes`."""
post_chunk(D::Int, m::Int, bytes::Integer) = clamp(Int(fld(bytes, 24 * max(D, 1))), 1, max(m, 1))

"""H residual norms ||H v - E v|| for the columns of `V` (block action, column chunks)."""
function post_h(H, V::Matrix{Float64}, E::Vector{Float64}; workers::Int=Threads.nthreads(), chunk::Int=size(V, 2))
    _post(H, V; workers, chunk) do j, x, y
        s2 = 0.0
        @inbounds @simd for i in eachindex(x); s2 += (y[i] - E[j] * x[i])^2; end
        sqrt(s2)
    end
end

"""C2 labels <v|C|v> and residual norms ||C v - c2 v||."""
function post_c(C, V::Matrix{Float64}; workers::Int=Threads.nthreads(), chunk::Int=size(V, 2))
    r = _post(C, V; workers, chunk) do j, x, y
        c2 = dot(x, y); s2 = 0.0
        @inbounds @simd for i in eachindex(x); s2 += (y[i] - c2 * x[i])^2; end
        (c2, sqrt(s2))
    end
    first.(r), last.(r)
end

function _post(f, A, V; workers, chunk)
    m = size(V, 2); D = size(V, 1); res = Vector{Any}(undef, m)
    Y = zeros(D, min(chunk, m))
    for c0 in 1:chunk:m
        cols = c0:min(c0 + chunk - 1, m); w = length(cols)
        X = V[:, cols]; y = w == size(Y, 2) ? Y : zeros(D, w)
        act_block!(y, A, X; workers)                 # v0.1.17: kernel chosen by size
        Threads.@threads for k in 1:w
            res[cols[k]] = f(cols[k], view(X, :, k), view(y, :, k))
        end
    end
    [r for r in res]
end

"""C2 labels and H/C2 residual norms for the columns of `V` (energies `E`) using the block action."""
function block_postprocess(H, C, V::Matrix{Float64}, E::Vector{Float64}; workers::Int=Threads.nthreads(), chunk::Int=size(V, 2))
    rh = post_h(H, V, E; workers, chunk); c2, rc = post_c(C, V; workers, chunk)
    c2, rc, Float64.(rh)
end

"""Write checkpoints in a background task (at most one in flight); `async=false` writes inline."""
function save_async!(slot::Ref{Union{Task,Nothing}}, jobs; async::Bool=true)
    slot[] === nothing || wait(slot[])     # rethrows a failed earlier write
    slot[] = nothing
    run() = foreach(((p, x),) -> _atomic_save(p, x), jobs)
    async ? (slot[] = Threads.@spawn run()) : run()
    nothing
end

"""States per exchange parity for a diagonal block: (k_plus, k_minus)."""
function sector_want(parity::AbstractString, k::Int)
    parity == "both" && return (k, k)
    parity == "plus" && return (k, 0)
    parity == "minus" && return (0, k)
    error("parity must be both | plus | minus, got $parity")
end

# Same Krylov dimension rule as FuzzifiED.GetEigensystem (ncv = min(max(2k, k+10), 100)).
# v0.1.12: high windows (k > 90) need krylovdim > k; the frozen rule is kept for k <= 90.
krylovdim(k) = k <= 90 ? min(max(2k, k + 10), 100) : k + max(30, k ÷ 2)

"""Lowest `k` states of a Hermitian action `f` on vectors of length `n` (ordinary Lanczos)."""
function lowest(f, n::Int, k::Int, seed::Int; tol=1e-10, maxiter=500)
    k = min(k, n)
    if n <= 400   # small: exact dense diagonalisation from columns of f
        M = zeros(n, n); e = zeros(n)
        for j in 1:n; fill!(e, 0); e[j] = 1; M[:, j] = f(e); end
        E, V = eigen(Symmetric((M + M') / 2))
        return E[1:k], [V[:, j] for j in 1:k], (converged=k, numops=n, dense=true)
    end
    alg = KrylovKit.Lanczos(krylovdim=min(n, krylovdim(k)), tol=tol, maxiter=maxiter, verbosity=0)
    E, V, info = KrylovKit.eigsolve(f, normalize(randn(MersenneTwister(seed), n)), k, :SR, alg)
    # KrylovKit may return more than k converged pairs; keep exactly the lowest k (deterministic row count).
    o = sortperm(E)[1:min(k, length(E))]
    Float64.(E[o]), [Vector{Float64}(V[i]) for i in o], (converged=info.converged, numops=info.numops, dense=false)
end

"""
    lowest_deflated(f, n, k, V0, seed; tol) -> (E, V, info)

v0.1.14: the next `k` states of the Hermitian action `f` above an existing orthonormal window
`V0` (n x m, columns = already computed eigenvectors).  Solves g = Q f Q + sigma P with
P = V0 V0', Q = 1 - P: span(V0) is lifted to sigma (far above the window), the complement keeps
H's spectrum, so the lowest k eigenpairs of g are the k states that follow V0.  Old vectors are
never recomputed; the union is the lowest m + k states when V0 is a complete lowest prefix
(callers check that no new state lies below the old window).
"""
function lowest_deflated(f, n::Int, k::Int, V0::AbstractMatrix{Float64}, seed::Int; tol=1e-10, maxiter=500, Eref=nothing)
    m = size(V0, 2); k = min(k, n - m)
    k <= 0 && return Float64[], Vector{Float64}[], (converged=0, numops=0, dense=false, deflated=m)
    m == 0 && return lowest(f, n, k, seed; tol, maxiter)
    scale = Eref === nothing ? 1.0 : maximum(abs, Eref; init=1.0)
    sigma = 10 * scale + 100
    P(v) = V0 * (V0' * v)
    # v0.1.17: P(v) is computed once per action (two projections instead of three; same operator).
    # The projections are BLAS-2 passes over V0 (n x m) and dominate deflated actions for large m.
    buf = zeros(m); pv = zeros(n)
    function g(v)
        mul!(buf, transpose(V0), v); mul!(pv, V0, buf)
        y = f(v - pv)
        mul!(buf, transpose(V0), y); mul!(y, V0, buf, -1.0, 1.0)
        y .+= sigma .* pv
        y
    end
    E, V, info = lowest(g, n, k, seed; tol, maxiter)
    # remove the numerical leak into span(V0) and re-orthonormalise the new block
    W = reduce(hcat, [v - P(v) for v in V])
    Qw = Matrix(qr(W).Q)[:, 1:size(W, 2)]
    Hs = Qw' * reduce(hcat, [f(Qw[:, j]) for j in axes(Qw, 2)])     # Rayleigh-Ritz in the new block
    F = eigen(Symmetric((Hs + Hs') / 2))
    U = Qw * F.vectors
    maximum(abs.(F.values .- E)) < 1e-6 * max(1, maximum(abs, E)) ||
        @warn "deflated Ritz values moved: $(maximum(abs.(F.values .- E)))"
    Float64.(F.values), [U[:, j] for j in axes(U, 2)], (converged=info.converged, numops=info.numops, dense=info.dense, deflated=m)
end

# Default residency ("auto"): src/resources.jl, resolve_residency (pair above 16e9 bytes of segment basis).

# ---------------------------------------------------------------------------
# Split segment-space store (residency "pair" / "minimal", v0.1.5).
# segment_split/bases holds the configuration bases shared by the three spaces, and
# segment_split/space_s the multiplet states of space s only; a pair loads <= 2 spaces.

"""Write `spaces` as one bases file plus one file per space (no shared Basis duplicated)."""
function split_segment_spaces(spaces, dir::AbstractString; source_sha::AbstractString="")
    b0 = spaces[1]
    for sp in spaces
        length(sp.bs) == length(b0.bs) && all(sp.bs[i] === b0.bs[i] || sp.bs[i].cfs.conf == b0.bs[i].cfs.conf
                                              for i in eachindex(b0.bs)) || error("segment spaces do not share bases")
    end
    isdir(joinpath(dir, "bases")) || _atomic_save(joinpath(dir, "bases"), (bs=b0.bs, bs1=b0.bs1, source_sha=source_sha))
    for (k, sp) in enumerate(spaces)
        d = joinpath(dir, "space_$(k-1)"); isdir(d) && continue
        T = eltype(sp.sts[1])
        light = SegSpace{T}(sp.sec, sp.sec_modul, sp.l_rng, sp.l_lookup, sp.ptr_st, Basis[], Basis[], sp.sts, sp.sts1)
        _atomic_save(d, (s=k - 1, space=light))
    end
    dir
end

mutable struct SpaceStore
    dir::String
    bases::Any
    live::Dict{Int,Any}
    loads::Int
end
SpaceStore(dir) = SpaceStore(dir, nothing, Dict{Int,Any}(), 0)

function getspace!(st::SpaceStore, s::Int)
    haskey(st.live, s) && return st.live[s]
    st.bases === nothing && (st.bases = _checked_load(joinpath(st.dir, "bases")))
    sp = _checked_load(joinpath(st.dir, "space_$s")).space
    sp.bs = st.bases.bs; sp.bs1 = st.bases.bs1
    st.loads += 1
    st.live[s] = sp
end

function dropspace!(st::SpaceStore, lib, s::Int)
    haskey(st.live, s) || return
    release!(lib, st.live[s]); delete!(st.live, s)
    isempty(st.live) && (st.bases = nothing)
    nothing
end

# ---------------------------------------------------------------------------

"""
    run_spectrum(cfg)

cfg keys: N, g, blocks (vector of (sA,sB,L)), nev, out, segment_checkpoint ("" = build),
exchange (Bool), tol, threads policy (native, tile, workers, action, build_blas, solver_blas),
post_bytes (work-array budget of the batched residual pass), async_io (background checkpoint writes),
parity (both | plus | minus), parity_nev (key => [k_plus, k_minus]), block_nev (key => k),
residency (all | pair | minimal | auto; auto = pair when the segment basis exceeds 16e9 bytes, src/resources.jl),
concurrency (blocks solved at once, residency all only; default 1), segment_split (directory of a
split store; default out/segment_split when residency != all).

Residency:
  all      every segment space and every built operator stays resident; blocks of a pair may run
           concurrently (`concurrency` > 1, each with action ÷ concurrency workers, BLAS = 1);
  pair     at most the two spaces of the current pair (and their operators) are resident;
  minimal  as pair, and the C2-only segment operators and the C2 CompOperator are not resident
           during the Lanczos phase: H-phase (solve, H residuals, eig.ckpt), then C2-phase (labels,
           C2 residuals) which adds only the operators C2 does not share with H.
A finished eig.ckpt is reused on restart (only post-processing is redone).
"""
function run_spectrum(cfg)
    N = cfg["N"]; g = cfg["g"]; out = cfg["out"]; nev = cfg["nev"]; tol = get(cfg, "tol", 1e-10)
    blocks = [Tuple(b) for b in cfg["blocks"]]
    useX = get(cfg, "exchange", true)
    native = get(cfg, "native", 8); action = get(cfg, "action", 64)
    mode = resolve_residency(cfg)
    mode in ("all", "pair", "minimal") || error("residency must be all | pair | minimal")
    conc = mode == "all" ? max(1, get(cfg, "concurrency", 1)) : 1
    mkpath(out); st = Stage(joinpath(out, "stages.tsv"))
    # v0.1.16 (V15-8): finish or roll back checkpoint replacements interrupted by a crash
    isdir(joinpath(out, "blocks")) && foreach(d -> _recover_replace!(joinpath(out, "blocks", d)),
                                             filter(d -> isdir(joinpath(out, "blocks", d)), readdir(joinpath(out, "blocks"))))
    FuzzifiED.NumThreads = native; BLAS.set_num_threads(get(cfg, "build_blas", 16))
    seg = get(cfg, "segment_checkpoint", "")
    split = get(cfg, "segment_split", "")
    # ---- segment spaces -------------------------------------------------------
    spaces = nothing; store = nothing; basis_sha = ""
    split_ready(d) = !isempty(d) && all(isdir(joinpath(d, x)) for x in ("bases", "space_0", "space_1", "space_2"))
    mode != "all" && isempty(split) && split_ready(joinpath(out, "segment_split")) && (split = joinpath(out, "segment_split"))
    if mode != "all" && split_ready(split)
        store = SpaceStore(split); basis_sha = _checked_load(joinpath(split, "bases")).source_sha
    else
        sp, basis_sha = if isempty(seg)
            own = joinpath(out, "segment_spaces.ckpt")
            # vectors are tied to this exact basis (segment-state gauge is not reproducible between
            # builds), so eig.ckpt records the SHA of the saved basis, not a placeholder.
            isdir(own) || stage!(st, "segment_build") do
                ph = _use_ph(cfg, N)
                _atomic_save(own, (meta=(N=N, tcc=string(TCC_VERSION), ph_half=ph), spaces=build_segment_spaces(N; ph)))
            end
            (_checked_load(own).spaces, strip(read(joinpath(own, "sha256.txt"), String)))
        else
            s = stage!(st, "segment_load") do; load_segment_spaces(seg); end
            (s.spaces, s.sha)
        end
        if mode == "all"
            spaces = sp
        else
            d = joinpath(out, "segment_split")
            stage!(st, "segment_split") do; split_segment_spaces(sp, d; source_sha=basis_sha); end
            sp = nothing; GC.gc(true)
            store = SpaceStore(d)
        end
    end
    space(s) = spaces === nothing ? stage!(st, "space_load_$s") do; getspace!(store, s); end : spaces[s+1]
    tpl = stage!(st, "terms") do; hamiltonian_template(N); end
    lib = SegLibrary(builder=Symbol(get(cfg, "builder", "tiled")), tile_width=get(cfg, "tile", 64),
                     workers=get(cfg, "workers", Threads.nthreads()), opmat_threads=native)
    pairs, lastuse = pair_schedule(blocks)
    rows = NamedTuple[]; io_task = Ref{Union{Task,Nothing}}(nothing)
    chunkof(D, m) = post_chunk(D, m, get(cfg, "post_bytes", 2^31))

    # ---- one block: solve (or reuse eig.ckpt) --------------------------------
    wanted_sectors(key, a, b) = begin
        kblk = get(get(cfg, "block_nev", Dict()), key, nev)
        if useX && a == b
            pk = get(get(cfg, "parity_nev", Dict()), key, nothing)
            want = pk === nothing ? sector_want(get(cfg, "parity", "both"), kblk) : (pk[1], pk[2])
            [(1, want[1]), (-1, want[2])]
        else
            [(0, kblk)]
        end
    end
    # prior (v0.1.14): an existing eig checkpoint of this block; its vectors are kept bit-identical and
    # only the missing states of each sector are computed, deflated against them (lowest_deflated).
    function solve!(H, cp, a, b, L, key, dir, act; prior=nothing)
        D = cp.dim
        sectors = if useX && a == b
            perm, sgn = exchange_map(cp)
            [(sg, ExchangeSector(perm, sgn, sg), k) for (sg, k) in wanted_sectors(key, a, b) if k > 0]
        else
            [(0, nothing, wanted_sectors(key, a, b)[1][2])]
        end
        Es = Float64[]; sigs = Int[]; convs = Int[]; ndims = Int[]; vecs = Vector{Float64}[]; isold = Bool[]
        if prior !== nothing                      # every old column is kept, whatever the new request
            for i in eachindex(prior.energy)
                push!(vecs, prior.vectors[:, i]); push!(Es, prior.energy[i]); push!(sigs, prior.parity[i])
                push!(convs, prior.converged[i]); push!(ndims, prior.sector_dim[i]); push!(isold, true)
            end
        end
        for (sig, x, kk) in sectors
            n = x === nothing ? D : length(x); n == 0 && continue
            f = x === nothing ? (v -> *(H, v; num_th=act)) : reduced_action(H, x; num_th=act)
            oldc = prior === nothing ? Int[] : findall(==(sig), prior.parity)
            kk - length(oldc) <= 0 && continue
            V0 = zeros(n, length(oldc))
            for (j, c) in enumerate(oldc)
                x === nothing ? (V0[:, j] .= prior.vectors[:, c]) : sector_reduce!(view(V0, :, j), x, prior.vectors[:, c])
            end
            E, V, info = stage!(st, "eig_$(key)_x$sig"; detail="n=$n k=$(min(kk,n)) old=$(length(oldc))") do
                isempty(oldc) ? lowest(f, n, kk, 1000a + 100b + 10L + (sig + 1); tol) :
                    lowest_deflated(f, n, kk - length(oldc), V0, 1000a + 100b + 10L + (sig + 1) + 7919; tol, Eref=prior.energy[oldc])
            end
            if !isempty(oldc) && !isempty(E) && minimum(E) < maximum(prior.energy[oldc]) - 1e-8
                error("block $key sector $sig: a deflated state lies below the stored window (stored window was not a complete lowest prefix)")
            end
            for (i, v) in enumerate(V)
                push!(vecs, x === nothing ? v : sector_expand!(zeros(D), x, v))
                push!(Es, E[i]); push!(sigs, sig); push!(convs, info.converged); push!(ndims, n); push!(isold, false)
            end
        end
        o = sortperm(Es)
        (energy=Es[o], parity=sigs[o], converged=convs[o], sector_dim=ndims[o], vectors=reduce(hcat, vecs[o]), isold=isold[o])
    end
    # v0.1.14: with cfg["extend"] a finished block is reopened when it holds fewer states than requested
    function needs_extension(a, b, L)
        get(cfg, "extend", false) || return false
        key = "s$(a)_s$(b)_L$(L)"; r = _checked_load(joinpath(out, "blocks", key, "result.ckpt"))
        r.dim == 0 && return false
        any(wanted_sectors(key, a, b)) do (sig, k)
            rs = [x for x in r.rows if x.x == sig]
            k > 0 && (isempty(rs) ? true : length(rs) < min(k, rs[1].sector_dim))
        end
    end
    eigdir(dir) = joinpath(dir, "eig.ckpt")
    function load_eig(dir)
        isdir(eigdir(dir)) || return nothing
        e = _checked_load(eigdir(dir))
        (e.N == N && e.g == g && hasproperty(e, :converged)) || error("$(eigdir(dir)) belongs to a different run")
        # v0.1.16 (V15-1): stored vectors are meaningful only in the basis they were computed in
        e.basis_sha == basis_sha || gatefail("$(eigdir(dir)) was computed in another segment basis")
        e
    end
    # v0.1.16: every eig checkpoint records exact column SHAs, the producing code and its gate record
    eig_payload(key, D, e, rh; extra=NamedTuple(), colsha=nothing) = merge((key=key, N=N, g=g, dim=D, energy=e.energy,
        parity=e.parity, converged=e.converged, sector_dim=e.sector_dim, rh=rh, vectors=e.vectors, basis_sha=basis_sha,
        colsha=colsha === nothing ? column_shas(e.vectors) : colsha, producer=producer()), extra)
    save_eig!(dir, key, D, e, rh; async, extra=NamedTuple()) = save_async!(io_task, [(eigdir(dir), eig_payload(key, D, e, rh; extra))]; async)
    reorder(e, o) = merge(e, (energy=e.energy[o], parity=e.parity[o], converged=e.converged[o], sector_dim=e.sector_dim[o],
                              vectors=e.vectors[:, o]), hasproperty(e, :isold) ? (isold=e.isold[o],) : NamedTuple(),
                          hasproperty(e, :rh) ? (rh=e.rh[o],) : NamedTuple())
    function rows_of(e, a, b, L, c2, rc, rh)
        [(energy=e.energy[i], L=L, sa=a, sb=b, x=e.parity[i], c2=c2[i], rc=rc[i], rh=rh[i],
          norm=norm(view(e.vectors, :, i)), converged=e.converged[i], sector_dim=e.sector_dim[i]) for i in eachindex(e.energy)]
    end
    function finish!(dir, key, D, brows; async, gate=nothing)
        save_async!(io_task, [(joinpath(dir, "result.ckpt"), result_payload(key, D, brows, gate))]; async)
    end
    result_payload(key, D, brows, gate) = (key=key, dim=D, rows=brows, gate=gate, producer=producer())
    done(a, b, L) = isdir(joinpath(out, "blocks", "s$(a)_s$(b)_L$(L)", "result.ckpt"))

    # ---- residency all / pair: H and C2 of a pair together --------------------
    function block_all!(ss, sH, fH, sC, fC, a, b, L, act; async)
        key = "s$(a)_s$(b)_L$(L)"; dir = joinpath(out, "blocks", key)
        _recover_replace!(dir)                                      # v0.1.16 (V15-8)
        cp = BuildCompSpace(ss, [2N, 0, 0], 2L); D = cp.dim
        if D == 0
            finish!(dir, key, 0, NamedTuple[]; async=false); return NamedTuple[]
        end
        H = stage!(st, "asm_H_$key"; detail="D=$D") do
            absorb!(set_g!(BuildCompOperator(cp, tpl.template, sH; num_th=act), tpl, g), fH); end
        e = load_eig(dir); fresh = e === nothing
        extend = !fresh && isdir(joinpath(dir, "result.ckpt"))         # reached only when needs_extension
        oldsha = String[]; oldrows = NamedTuple[]
        if extend
            oldrows = sort(_checked_load(joinpath(dir, "result.ckpt")).rows; by=x -> x.energy)
            # v0.1.16 (V15-1): the stored window must be coherent and its columns are identified exactly
            [x.energy for x in oldrows] == e.energy || gatefail("block $key: stored eig and result windows differ")
            oldsha = column_shas(e.vectors)
            hasproperty(e, :colsha) && e.colsha != oldsha && gatefail("block $key: stored column SHAs do not match the stored vectors")
            e = solve!(H, cp, a, b, L, key, dir, act; prior=e)
        elseif fresh
            e = solve!(H, cp, a, b, L, key, dir, act)
        end
        hasproperty(e, :isold) || (e = merge(e, (isold=fill(false, length(e.energy)),)))
        m = length(e.energy)
        newc = findall(!, e.isold)                                  # post-processing only for new states
        rh = zeros(m); c2 = zeros(m); rc = zeros(m); C = nothing
        if !isempty(newc)
            Vn = e.vectors[:, newc]
            rh[newc] = stage!(st, "postH_$key"; detail="m=$(length(newc))") do; post_h(H, Vn, e.energy[newc]; workers=act, chunk=chunkof(D, length(newc))); end
            C = stage!(st, "asm_C2_$key"; detail="D=$D") do; absorb!(BuildCompOperator(cp, tpl.c2, sC; num_th=act), fC); end
            cc, rr = stage!(st, "postC_$key"; detail="m=$(length(newc))") do; post_c(C, Vn; workers=act, chunk=chunkof(D, length(newc))); end
            c2[newc] = cc; rc[newc] = rr; Vn = nothing
        end
        # v0.1.16: accidental near-degeneracies between SO(5) irreps (N9 (1,1) L4: C2 = 10 / 5 states 9e-7
        # apart, C2 residual 1.8e-4) are resolved in the spectrum as in the target windows; old columns
        # stay frozen.  Only blocks that would otherwise fail the C2 gate are touched.
        npol = 0
        if !isempty(newc) && maximum(rc[newc]) > GATE.rc
            E2 = copy(e.energy); V2 = copy(e.vectors)
            npol = stage!(st, "polish_$key"; detail="m=$m") do
                polish_c2!(E2, V2, e.parity, H, C; workers=act, frozen=e.isold, trigger=GATE.rc); end
            if npol > 0
                e = merge(e, (energy=E2, vectors=V2))
                Vn = e.vectors[:, newc]
                rh[newc] = post_h(H, Vn, e.energy[newc]; workers=act, chunk=chunkof(D, length(newc)))
                cc, rr = post_c(C, Vn; workers=act, chunk=chunkof(D, length(newc))); c2[newc] = cc; rc[newc] = rr; Vn = nothing
                o = _sortperm_stable(e.energy)
                if o != 1:m
                    e = reorder(e, o); rh = rh[o]; c2 = c2[o]; rc = rc[o]
                end
            end
        end
        H = nothing; C = nothing
        if extend                                                   # old rows reused verbatim
            oc = findall(e.isold)
            length(oc) == length(oldrows) || error("old window/result mismatch in $key")
            for (j, i) in enumerate(oc)
                e.energy[i] == oldrows[j].energy || gatefail("block $key: old state $j moved during extension")
                rh[i] = oldrows[j].rh; c2[i] = oldrows[j].c2; rc[i] = oldrows[j].rc
            end
        end
        colsha = column_shas(e.vectors)
        extend && sort(colsha[e.isold]) != sort(oldsha) && gatefail("block $key: an old column changed during extension")
        gd = window_gate("block $key", e.energy, e.vectors; rh, rc, c2)
        brows = rows_of(e, a, b, L, c2, rc, rh)
        eigp = eig_payload(key, D, e, rh; extra=(gate=gd, polished_groups=npol), colsha)
        if extend                                                   # previous checkpoints kept, superseded
            _replace!([eigdir(dir) => eigp, joinpath(dir, "result.ckpt") => result_payload(key, D, brows, gd)];
                      tag="k$(count(e.isold))")
        elseif fresh
            save_async!(io_task, [(eigdir(dir), eigp)]; async)
            finish!(dir, key, D, brows; async, gate=gd)
        else                                                        # eig.ckpt from an interrupted run
            npol > 0 && _replace!([eigdir(dir) => eigp]; tag="prepolish")
            finish!(dir, key, D, brows; async, gate=gd)
        end
        brows
    end

    for (k, (a, b)) in enumerate(pairs)
        mine = [blk for blk in blocks if blk[1] == a && blk[2] == b]
        todo = [blk[3] for blk in mine if !done(a, b, blk[3]) || needs_extension(a, b, blk[3])]
        mode == "minimal" && any(L -> done(a, b, L), todo) && error("extend is supported with residency all / pair")
        if !isempty(todo)
            if store !== nothing                     # pair / minimal: <= 2 spaces resident
                for s in collect(keys(store.live)); s in (a, b) || dropspace!(store, lib, s); end
                GC.gc(true)
            end
            ss = [space(a), space(b)]
            if mode != "minimal"
                sH, fH = stage!(st, "seg_H_$a$b") do; segops_shared!(lib, ss, tpl.template); end
                sC, fC = stage!(st, "seg_C2_$a$b") do; segops_shared!(lib, ss, tpl.c2); end
                BLAS.set_num_threads(get(cfg, "solver_blas", 16))
                if conc == 1 || length(todo) == 1
                    for L in todo
                        append!(rows, block_all!(ss, sH, fH, sC, fC, a, b, L, action; async=get(cfg, "async_io", true)))
                        GC.gc(false)
                    end
                else
                    # v0.1.5 multi-block concurrency (residency all): largest blocks first,
                    # `conc` blocks at a time, workers split evenly, BLAS single-threaded.
                    BLAS.set_num_threads(1)
                    io_task[] === nothing || wait(io_task[]); io_task[] = nothing
                    order = sort(todo; by=L -> -BuildCompSpace(ss, [2N, 0, 0], 2L).dim)
                    act = max(1, action ÷ min(conc, length(order)))
                    lk = ReentrantLock(); nxt = Threads.Atomic{Int}(0)
                    @sync for _ in 1:min(conc, length(order))
                        Threads.@spawn while true
                            i = Threads.atomic_add!(nxt, 1) + 1
                            i > length(order) && break
                            r = block_all!(ss, sH, fH, sC, fC, a, b, order[i], act; async=false)
                            lock(() -> append!(rows, r), lk)
                        end
                    end
                end
                BLAS.set_num_threads(get(cfg, "build_blas", 16))
            else
                # minimal: H phase, empty library, C2 phase
                sH, fH = stage!(st, "seg_H_$a$b") do; segops_shared!(lib, ss, tpl.template); end
                BLAS.set_num_threads(get(cfg, "solver_blas", 16))
                for L in todo
                    key = "s$(a)_s$(b)_L$(L)"; dir = joinpath(out, "blocks", key)
                    isdir(eigdir(dir)) && continue          # finished H phase (atomic checkpoint)
                    cp = BuildCompSpace(ss, [2N, 0, 0], 2L); D = cp.dim; D == 0 && continue
                    H = stage!(st, "asm_H_$key"; detail="D=$D") do
                        absorb!(set_g!(BuildCompOperator(cp, tpl.template, sH; num_th=action), tpl, g), fH); end
                    e = solve!(H, cp, a, b, L, key, dir, action)
                    rh = stage!(st, "postH_$key"; detail="m=$(length(e.energy))") do
                        post_h(H, e.vectors, e.energy; workers=action, chunk=chunkof(D, length(e.energy))); end
                    H = nothing
                    save_eig!(dir, key, D, e, rh; async=false); e = nothing; GC.gc(false)   # minimal: never two vector sets
                end
                io_task[] === nothing || wait(io_task[]); io_task[] = nothing
                # v0.1.11: keep the H operators — C2 shares most of them (N12 (1,1): H 35 GiB, C2 alone 55 GiB,
                # union 59 GiB); only the C2-specific operators are added now, after the Lanczos phase.
                sH = nothing; GC.gc(true)
                BLAS.set_num_threads(get(cfg, "build_blas", 16))
                sC, fC = stage!(st, "seg_C2_$a$b") do; segops_shared!(lib, ss, tpl.c2); end
                for L in todo
                    key = "s$(a)_s$(b)_L$(L)"; dir = joinpath(out, "blocks", key)
                    cp = BuildCompSpace(ss, [2N, 0, 0], 2L); D = cp.dim
                    if D == 0
                        finish!(dir, key, 0, NamedTuple[]; async=false); continue
                    end
                    _recover_replace!(dir)
                    e = load_eig(dir)
                    C = stage!(st, "asm_C2_$key"; detail="D=$D") do; absorb!(BuildCompOperator(cp, tpl.c2, sC; num_th=action), fC); end
                    c2, rc = stage!(st, "postC_$key"; detail="m=$(length(e.energy))") do
                        post_c(C, e.vectors; workers=action, chunk=chunkof(D, length(e.energy))); end
                    rh = e.rh; npol = 0
                    if maximum(rc; init=0.0) > GATE.rc              # v0.1.16: polish as in block_all!
                        sH2, fH2 = segops_shared!(lib, ss, tpl.template)   # H operators stay in the library (v0.1.11)
                        Hm = absorb!(set_g!(BuildCompOperator(cp, tpl.template, sH2; num_th=action), tpl, g), fH2)
                        E2 = copy(e.energy); V2 = copy(e.vectors)
                        npol = stage!(st, "polish_$key"; detail="m=$(length(E2))") do
                            polish_c2!(E2, V2, e.parity, Hm, C; workers=action, trigger=GATE.rc); end
                        if npol > 0
                            rh = post_h(Hm, V2, E2; workers=action, chunk=chunkof(D, length(E2)))
                            c2, rc = post_c(C, V2; workers=action, chunk=chunkof(D, length(E2)))
                            e = merge(e, (energy=E2, vectors=V2, rh=rh))
                            o = _sortperm_stable(e.energy)
                            o != eachindex(o) && (e = reorder(e, o); rh = e.rh; c2 = c2[o]; rc = rc[o])
                        end
                        Hm = nothing; sH2 = nothing
                    end
                    C = nothing
                    gd = window_gate("block $key", e.energy, e.vectors; rh, rc, c2)
                    npol > 0 && _replace!([eigdir(dir) => eig_payload(key, D, e, rh; extra=(gate=gd, polished_groups=npol))]; tag="prepolish")
                    brows = rows_of(e, a, b, L, c2, rc, rh); e = nothing
                    finish!(dir, key, D, brows; async=false, gate=gd); append!(rows, brows); GC.gc(false)
                end
                sC = nothing; for s in (a, b); release!(lib, ss[s == a ? 1 : 2]); end
            end
        end
        for blk in mine
            blk[3] in todo && continue
            append!(rows, _checked_load(joinpath(out, "blocks", "s$(a)_s$(b)_L$(blk[3])", "result.ckpt")).rows)
        end
        if store === nothing
            for s in (a, b); lastuse[s] == k && release!(lib, spaces[s+1]); end
        end
        GC.gc(true)
    end
    io_task[] === nothing || wait(io_task[])
    store === nothing || for s in collect(keys(store.live)); dropspace!(store, lib, s); end
    sort!(rows; by=r -> r.energy)
    open(joinpath(out, "spectrum.tsv"), "w") do io
        println(io, "energy\tL\tsA\tsB\tX\tC2\tC2_res\tH_res\tconverged\tsector_dim")
        for r in rows
            println(io, join((r.energy, r.L, r.sa, r.sb, r.x, r.c2, r.rc, r.rh, r.converged, r.sector_dim), "\t"))
        end
    end
    (rows=rows, version=string(TCC_VERSION), residency=mode, concurrency=conc,
     library=(builds=lib.builds, hits=lib.hits, seconds=lib.build_seconds),
     space_loads=store === nothing ? 0 : store.loads)
end
