# Built-in acceptance gates and exact column identities (v0.1.16).
#
# Until v0.1.15 several acceptance checks lived only in external job scripts (review of 2026-10-12,
# findings V15-1 ... V15-9).  They are cheap once the vectors are in memory (V'V is one GEMM of
# D x m, a column SHA-256 one pass over the column, H/C2 residuals are computed anyway), so every
# window is now checked where it is produced or reused:
#   * window_gate      finite, sorted, normalised, orthonormal (full V'V), H and C2 residuals and
#                      integer C2 labels within GATE; a failure throws GateError ("GATE_FAIL ...");
#   * column_shas      SHA-256 of the bytes of every column: exact vector identity (replaces the
#                      two-number column fingerprints for incremental edge reuse and proves that
#                      frozen / old columns are bit-preserved);
#   * _replace!        multi-checkpoint replacement with a commit marker, so an interrupted
#                      extension can always be completed or rolled back (V15-8);
#   * producer_info    code identity recorded with every checkpoint and stage.
# The thresholds are constants on purpose: they are the acceptance contract of a run, not knobs.

const GATE = (rh=1e-6, rc=1e-6, orth=1e-6, norm=1e-6, c2int=1e-5, independent_de=1e-7, independent_sv=1 - 1e-5)

struct GateError <: Exception
    msg::String
end
Base.showerror(io::IO, e::GateError) = print(io, "GATE_FAIL ", e.msg)
gatefail(msg) = throw(GateError(msg))

"""SHA-256 (hex) of the raw bytes of every column of `V` (threaded; exact vector identity)."""
function column_shas(V::Matrix{Float64})
    D = size(V, 1); m = size(V, 2); out = Vector{String}(undef, m)
    D == 0 && (fill!(out, bytes2hex(sha256(UInt8[]))); return out)
    GC.@preserve V begin
        Threads.@threads for j in 1:m
            bytes = unsafe_wrap(Array, Ptr{UInt8}(pointer(V, (j - 1) * D + 1)), 8D)
            out[j] = bytes2hex(sha256(bytes))
        end
    end
    out
end
column_shas(V::AbstractMatrix) = column_shas(Matrix{Float64}(V))

"""Stored column SHAs of a checkpoint payload when present (and consistent in length), else computed."""
stored_or_column_shas(x, V) = (hasproperty(x, :colsha) && x.colsha !== nothing && length(x.colsha) == size(V, 2)) ?
                              Vector{String}(x.colsha) : column_shas(V)

"""Code identity of the running Tcc (version, canonical source-tree digest, git commit)."""
function producer_info()
    tree = try; tree_digest(dirname(@__DIR__)); catch; ""; end
    (tcc=string(TCC_VERSION), code_tree=tree, git_commit=get(ENV, "TCC_GIT_COMMIT", git_commit(@__DIR__)))
end
const _PRODUCER = Ref{Any}(nothing)
producer() = (_PRODUCER[] === nothing && (_PRODUCER[] = producer_info()); _PRODUCER[])

"""
    window_gate(what, E, V; rh, rc, c2, sorted=true) -> diagnostics NamedTuple

Hard acceptance of a window (energies `E`, columns `V`, optional H residuals `rh`, C2 residuals
`rc`, C2 labels `c2`).  Throws GateError unless every number is finite, `E` is ascending, every
column has unit norm and V'V = 1 (Frobenius) within GATE, and residuals / C2 integrality are
within GATE.
"""
function window_gate(what, E::AbstractVector, V::AbstractMatrix; rh=nothing, rc=nothing, c2=nothing, sorted::Bool=true)
    m = length(E)
    size(V, 2) == m || gatefail("$what: $(size(V, 2)) vectors for $m energies")
    for (nm, x) in (("H residuals", rh), ("C2 residuals", rc), ("C2 labels", c2))
        x === nothing || length(x) == m || gatefail("$what: $(length(x)) $nm for $m states")
    end
    all(isfinite, E) || gatefail("$what: non-finite energy")
    sorted && !issorted(E) && gatefail("$what: energies not ascending")
    orth = 0.0; nerr = 0.0
    if m > 0
        G = Matrix{Float64}(undef, m, m); mul!(G, transpose(V), V)
        all(isfinite, G) || gatefail("$what: non-finite vector entries")
        nerr = maximum(abs(sqrt(max(G[j, j], 0.0)) - 1) for j in 1:m)
        for j in 1:m; G[j, j] -= 1; end
        orth = norm(G)
    end
    mx(x) = x === nothing || isempty(x) ? 0.0 : maximum(x)
    all(x -> x === nothing || all(isfinite, x), (rh, rc, c2)) || gatefail("$what: non-finite residual or C2 label")
    mh = mx(rh); mc = mx(rc); ci = c2 === nothing || isempty(c2) ? 0.0 : maximum(abs.(c2 .- round.(c2)))
    nerr <= GATE.norm || gatefail("$what: column norm error $nerr > $(GATE.norm)")
    orth <= GATE.orth || gatefail("$what: orthogonality ||V'V - 1|| = $orth > $(GATE.orth)")
    mh <= GATE.rh || gatefail("$what: max H residual $mh > $(GATE.rh)")
    mc <= GATE.rc || gatefail("$what: max C2 residual $mc > $(GATE.rc)")
    ci <= GATE.c2int || gatefail("$what: C2 label off an integer by $ci > $(GATE.c2int)")
    (states=m, orthogonality=orth, max_norm_error=nerr, max_H_residual=mh, max_C2_residual=mc, max_C2_integer_error=ci)
end

"""Stable permutation that sorts `E` (identity when already sorted)."""
_sortperm_stable(E) = sortperm(E; alg=Base.Sort.MergeSort)

# ---------------------------------------------------------------------------------------------
# Atomic multi-checkpoint replacement (V15-8).
#   1. every new payload is written to <path>.next (atomic, checksummed);
#   2. a commit marker <dir>/.replace-<tag> listing the paths is written (atomic rename);
#   3. for each path: active <path> -> superseded name (never deleted), <path>.next -> <path>;
#   4. the marker is removed.
# _recover_replace!(dir) at the start of the next run: with a marker the commit is completed;
# without one, stray .next checkpoints (written before the marker) are moved aside, leaving the
# previous, coherent checkpoints active.

_superseded_name(p, tag) = (b = endswith(p, ".ckpt") ? p[1:end-5] : p; d = "$b.$tag.ckpt"; n = 1;
                            while ispath(d); d = "$b.$tag.ckpt.$n"; n += 1; end; d)

function _replace!(items::AbstractVector; tag::AbstractString)
    isempty(items) && return nothing
    paths = [String(first(x)) for x in items]
    dir = dirname(first(paths)); all(p -> dirname(p) == dir, paths) || error("_replace!: paths must share a directory")
    for (p, x) in items
        nx = p * ".next"; ispath(nx) && error("_replace!: stale $nx (run _recover_replace! first)")
        _atomic_save(nx, x)
    end
    mk = joinpath(dir, ".replace-" * tag); tmp = mk * ".tmp"
    open(io -> (println(io, tag); foreach(p -> println(io, basename(p)), paths)), tmp, "w"); mv(tmp, mk; force=true)
    _commit_replace!(dir, mk)
end

function _commit_replace!(dir, mk)
    L = readlines(mk); tag = L[1]
    for b in L[2:end]
        p = joinpath(dir, b); nx = p * ".next"
        isdir(nx) || continue                                   # already committed
        isdir(p) && mv(p, _superseded_name(p, tag))
        mv(nx, p)
    end
    rm(mk; force=true)
    nothing
end

"""Complete (marker present) or roll back (no marker) an interrupted _replace! in `dir`."""
function _recover_replace!(dir)
    isdir(dir) || return 0
    n = 0
    for f in readdir(dir)
        startswith(f, ".replace-") && !endswith(f, ".tmp") || continue
        println("REPLACE_RECOVER commit $(joinpath(dir, f))"); _commit_replace!(dir, joinpath(dir, f)); n += 1
    end
    for f in readdir(dir)
        endswith(f, ".next") || continue
        p = joinpath(dir, f); d = p * ".abandoned"; k = 1
        while ispath(d); d = p * ".abandoned.$k"; k += 1; end
        println("REPLACE_RECOVER abandon $p"); mv(p, d); n += 1
    end
    n
end
