# Matrix-element "edges" for KK and OPE.
#
# An edge (source block L_i -> target block L_f, local operator candidates t_1..t_n of rank r)
# produces the arrays
#   overlaps[p][:, j] = T' * (O_p * S[:, j])      image_gram[p, q, j] = <O_p S_j, O_q S_j>
# Compared with the plain FuzzifiED construction (edge_arrays_reference below), only the way the
# same numbers are produced differs:
#   * segment operators come from the shared SegLibrary (no rebuild per edge / per term),
#   * each O_p acts on a block of source columns at once (tcc_mul_block!),
#   * source columns are processed in chunks to bound image memory (Dt * chunk * 8 bytes per term).

"""Rank-`rank` Gamma^2 density insertion used by the OPE pipeline (n_A - n_B component)."""
function gamma2_terms(N::Int, rank::Int)
    n = FuzzifiED.AngModes(GetDensityObs(N, 2, Matrix{Int}(I, 2, 2); norm_r2=Float64(N)))
    part = FilterL2(n, rank); q = [0, 0, 0]
    [SingleSegCouple(2, 1, part, 2rank, q) - SingleSegCouple(2, 2, part, 2rank, q)]
end

"""
    kk_terms(N; cache="")

KK generator candidates (density^2, symplectic pair^2) of rank 1.  With `cache` a directory,
the CoupleDecomps are stored once per (N, KKGenerator source, FuzzifiED source tree, Julia
version) and reloaded (v0.1.3: 23/38/85 s at N = 10/11/12 become a file read).
"""
function kk_terms(N::Int; cache::AbstractString="")
    build() = (c = KKGenerator.generator_candidates(N); [c.density_squared, c.symplectic_pair_squared])
    isempty(cache) && return build()
    key = bytes2hex(sha256(string("kk_terms-v1|N=", N, "|gen=", file_digest(joinpath(@__DIR__, "kk_generator.jl")),
                                  "|fz=", tree_digest(dirname(pathof(FuzzifiED))), "|jl=", VERSION)))
    dir = joinpath(cache, "kk_terms_N$(N)_$(key[1:16])")
    isdir(dir) && return _checked_load(dir).terms
    t = build()
    try
        _atomic_save(dir, (key=key, N=N, terms=t))
    catch err
        isdir(dir) || rethrow()      # a concurrent writer won the race: fine
    end
    t
end

function edge_arrays(lib::SegLibrary, ss::AbstractVector, sp, tp, terms, ltot2::Int,
                     S::AbstractMatrix{Float64}, T::AbstractMatrix{Float64};
                     workers::Int=Threads.nthreads(), chunk::Int=size(S, 2), st=nothing)
    tm(f, label) = st === nothing ? f() : stage!(f, st, label)
    sgs = tm("edge_segops") do; [segops_shared!(lib, ss, t) for t in terms]; end
    ops = tm("edge_assembly") do
        [absorb!(BuildCompOperator(sp, tp, t, sg, ltot2; disp_std=false, num_th=workers), fac)
         for (t, (sg, fac)) in zip(terms, sgs)]
    end
    big = max(sp.dim, tp.dim) > NATIVE_DIM_MAX[]          # v0.1.17: native kernel below, column split above
    plans = big ? [matvec_plan(op; workers=workers, granularity=1) for op in ops] : fill(nothing, length(ops))
    nS = size(S, 2); nT = size(T, 2); np = length(ops)
    Ms = [zeros(nT, nS) for _ in 1:np]; G = zeros(np, np, nS)
    nb0 = BLAS.get_num_threads()
    for c0 in 1:chunk:nS
        cols = c0:min(c0 + chunk - 1, nS)
        Xs = Matrix(view(S, :, cols))
        Ys = tm("edge_action_$(first(cols))") do
            [act_block!(zeros(tp.dim, length(cols)), op, Xs; workers, plan=pl) for (op, pl) in zip(ops, plans)]
        end
        BLAS.set_num_threads(min(32, Threads.nthreads()))
        tm("edge_projection_$(first(cols))") do
            for p in 1:np
                mul!(view(Ms[p], :, cols), transpose(T), Ys[p])
                for q in 1:np, (k, j) in enumerate(cols)
                    G[p, q, j] = dot(view(Ys[p], :, k), view(Ys[q], :, k))
                end
            end
        end
        BLAS.set_num_threads(nb0)
    end
    (overlaps=Ms, image_gram=G)
end

"""
    column_fingerprints(V) -> nV x 2 matrix

v0.1.14 cheap per-column summaries.  Deprecated in v0.1.16: two weighted sums cannot establish
that two vectors are identical (review finding V15-6); incremental edge reuse now matches columns
by exact energy *and* SHA-256 of the column bytes (`column_shas`).  Kept for old scripts only.
"""
function column_fingerprints(V::AbstractMatrix{Float64})
    F = zeros(size(V, 2), 2)
    Threads.@threads for j in axes(V, 2)
        s1 = 0.0; s2 = 0.0
        @inbounds for i in axes(V, 1)
            h = (UInt64(i) * 0x9E3779B97F4A7C15) >> 40           # 24-bit hash of the row index
            s1 += V[i, j] * (Float64(h & 0xFFF) / 4096 - 0.5)
            s2 += V[i, j] * (Float64(h >> 12) / 4096 - 0.5)
        end
        F[j, 1] = s1; F[j, 2] = s2
    end
    F
end

# old column -> new column, only for the *same vector*: identical energy and identical SHA-256 of the
# column bytes (v0.1.16).  Edges without stored column SHAs (written before v0.1.16) are never
# reused incrementally.
function _match_columns(oldE, oldS, newE, newS)
    m = Vector{Union{Nothing,Int}}(nothing, length(oldE))
    for (k, x) in enumerate(oldE)
        for j in findall(==(x), newE)
            if oldS[k] == newS[j]
                m[k] = j; break
            end
        end
    end
    m
end

"""
    edge_arrays_incremental(lib, ss, sp, tp, terms, ltot2, S, SE, T, TE, old; workers, chunk, SS, TS)

v0.1.14: edge arrays for enlarged windows (S, T with energies SE, TE) re-using an earlier edge
`old` computed on sub-windows.  v0.1.16: old columns are recognised as the same vectors only by
exact energy and identical column SHA-256 (`SS`, `TS`: column SHAs of S and T; `old` must carry
`source_colsha` / `target_colsha`, otherwise `nothing` is returned and the caller recomputes).
Only the missing blocks are computed:
  new sources  x all targets : the operator acts on the new source vectors only;
  old sources  x new targets : the operator acts on the old sources again, projected on the new
                               targets only (no eigen-solve, a few matvecs per old source);
  old sources  x old targets and the old image Gram are copied.
Returns (overlaps, image_gram, reused) or `nothing` when nothing can be reused.
"""
function edge_arrays_incremental(lib::SegLibrary, ss::AbstractVector, sp, tp, terms, ltot2::Int,
                                 S::AbstractMatrix{Float64}, SE::AbstractVector, T::AbstractMatrix{Float64},
                                 TE::AbstractVector, old; workers::Int=Threads.nthreads(), chunk::Int=size(S, 2),
                                 SS=column_shas(S), TS=column_shas(T))
    (hasproperty(old, :source_colsha) && hasproperty(old, :target_colsha)) || return nothing
    (old.source_colsha === nothing || old.target_colsha === nothing) && return nothing
    ms0 = _match_columns(old.source_E, old.source_colsha, SE, SS)
    mt0 = _match_columns(old.target_E, old.target_colsha, TE, TS)
    os = findall(!isnothing, ms0); ot = findall(!isnothing, mt0)     # usable old columns
    (isempty(os) || isempty(ot)) && return nothing
    ms = Int.(ms0[os]); mt = Int.(mt0[ot])
    (allunique(ms) && allunique(mt)) || return nothing
    nS = size(S, 2); nT = size(T, 2); np = length(old.overlaps)
    Ms = [zeros(nT, nS) for _ in 1:np]; G = zeros(np, np, nS)
    for p in 1:np; Ms[p][mt, ms] = old.overlaps[p][ot, os]; end
    G[:, :, ms] = old.image_gram[:, :, os]
    ns = setdiff(1:nS, ms); nt = setdiff(1:nT, mt)
    if !isempty(ns)
        A = edge_arrays(lib, ss, sp, tp, terms, ltot2, S[:, ns], T; workers, chunk=min(chunk, length(ns)))
        for p in 1:np; Ms[p][:, ns] = A.overlaps[p]; end
        G[:, :, ns] = A.image_gram
    end
    if !isempty(nt)
        B = edge_arrays(lib, ss, sp, tp, terms, ltot2, S[:, ms], T[:, nt]; workers, chunk=min(chunk, length(ms)))
        for p in 1:np; Ms[p][nt, ms] = B.overlaps[p]; end
    end
    (overlaps=Ms, image_gram=G, source_colsha=SS, target_colsha=TS,
     reused=(sources=length(ms), targets=length(mt), new_sources=length(ns), new_targets=length(nt)))
end

"""Reference construction: per-edge BuildSegOperators and one source vector at a time (plain FuzzifiED)."""
function edge_arrays_reference(ss::AbstractVector, sp, tp, terms, ltot2::Int,
                               S::AbstractMatrix{Float64}, T::AbstractMatrix{Float64};
                               native::Int=8, action::Int=64, st=nothing)
    tm(f, label) = st === nothing ? f() : stage!(f, st, label)
    ops = tm("ref_edge_build") do
        [BuildCompOperator(sp, tp, t, BuildSegOperators(ss, t; num_th=native), ltot2; disp_std=false, num_th=native) for t in terms]
    end
    np = length(ops); nS = size(S, 2)
    Ms = [zeros(size(T, 2), nS) for _ in 1:np]; G = zeros(np, np, nS)
    tm("ref_edge_actions") do
        for j in 1:nS
            imgs = [*(op, S[:, j]; num_th=action) for op in ops]
            for p in 1:np
                Ms[p][:, j] = T' * imgs[p]
                for q in 1:np; G[p, q, j] = dot(imgs[p], imgs[q]); end
            end
        end
    end
    (overlaps=Ms, image_gram=G)
end
