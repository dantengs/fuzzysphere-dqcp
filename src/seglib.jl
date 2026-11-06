# Unified segment-operator library (Tcc C1) and tiled, task-parallel builder (Tcc B1).
#
# A SegOperator depends only on (initial segment space, final segment space,
# local angular modes restricted to rank ll, rank ll, sector shift).  It does not
# depend on which physical part it sits in, on the channel coefficient, on g, or on
# whether it belongs to H, C2, a KK generator or an OPE insertion.  The library keys
# operators by exactly these inputs (operator content hashed from the actual Terms of
# every m-component that BuildSegOperator may read), so H and C2 share the density
# and J=2s pair operators, parts A/B share operators on a common space, and every
# (sA,sB) pair reuses the operators of its spaces.

const IDENTITY_KEY = "identity"

function _terms_digest!(io::IO, tms)
    rows = [(Vector{Int}(t.cstr), reinterpret(UInt64, real(ComplexF64(t.coeff))),
             reinterpret(UInt64, imag(ComplexF64(t.coeff)))) for t in tms]
    sort!(rows)
    write(io, Int64(length(rows)))
    for (c, re, im) in rows
        write(io, Int64(length(c))); for x in c; write(io, Int64(x)); end
        write(io, re); write(io, im)
    end
end

"""Content key of the rank-`ll` part of `amd` (all m components, exact coefficient bits)."""
function amd_key(amd, ll::Int)
    amd === :Identity && return IDENTITY_KEY
    io = IOBuffer()
    write(io, Int64(ll))
    for m2 in -ll:2:ll
        write(io, Int64(m2))
        _terms_digest!(io, GetComponent(amd, ll / 2, m2 / 2))
    end
    bytes2hex(sha256(take!(io)))
end

# v0.1.3: normalisation-aware key.  Operators whose terms differ only by one overall scalar
# (density modes with a different norm_r2 or 1/(N sqrt(4pi)) factor, H vs Gamma^2 vs KK
# insertions) share one library entry; the scalar is returned and absorbed into the
# CompOperator channel coefficient.  The normalised coefficients are rounded to 12
# significant digits before hashing (a different operator would have to agree to 1e-12).
_q(x) = abs(x) < 1e-13 ? 0.0 : round(x; sigdigits=12)
function amd_key_normalised(amd, ll::Int)
    amd === :Identity && return (IDENTITY_KEY, 1.0 + 0im)
    comps = [GetComponent(amd, ll / 2, m2 / 2) for m2 in -ll:2:ll]
    lam = 0.0 + 0im
    for c in comps, t in c
        abs(t.coeff) > abs(lam) * (1 + 1e-12) && (lam = ComplexF64(t.coeff))
    end
    lam == 0 && return (amd_key(amd, ll), 1.0 + 0im)
    io = IOBuffer(); write(io, "normalised-v1"); write(io, Int64(ll))
    for (k, c) in enumerate(comps)
        write(io, Int64(-ll + 2(k - 1)))
        rows = sort([(Vector{Int}(t.cstr), _q(real(t.coeff / lam)), _q(imag(t.coeff / lam))) for t in c])
        write(io, Int64(length(rows)))
        for (cs, re, im) in rows
            write(io, Int64(length(cs))); for x in cs; write(io, Int64(x)); end
            write(io, re); write(io, im)
        end
    end
    (bytes2hex(sha256(take!(io))), lam)
end

struct SegOpKey
    space_d::UInt64
    space_f::UInt64
    amd::String
    ll::Int
    secop::Vector{Int}
end
Base.:(==)(a::SegOpKey, b::SegOpKey) = a.space_d == b.space_d && a.space_f == b.space_f &&
    a.amd == b.amd && a.ll == b.ll && a.secop == b.secop
Base.hash(k::SegOpKey, h::UInt) = hash((k.space_d, k.space_f, k.amd, k.ll, k.secop), h)

mutable struct SegLibrary
    spaces::Dict{UInt64,Any}          # keep referenced spaces alive while keyed by objectid
    ops::Dict{SegOpKey,Any}           # key => (operator, scale of the terms it was built from)
    bytes::Dict{SegOpKey,Int}
    builder::Symbol                   # :tiled (Tcc) or :upstream (FuzzifiED BuildSegOperator)
    tile_width::Int
    workers::Int
    opmat_threads::Int
    builds::Int
    hits::Int
    build_seconds::Float64
    log::Vector{Any}
    normalise::Bool
    scaled_hits::Int
end

SegLibrary(; builder::Symbol=:tiled, tile_width::Int=64, workers::Int=Threads.nthreads(),
           opmat_threads::Int=8, normalise::Bool=true) =
    SegLibrary(Dict{UInt64,Any}(), Dict{SegOpKey,Any}(), Dict{SegOpKey,Int}(), builder,
               tile_width, workers, opmat_threads, 0, 0, 0.0, Any[], normalise, 0)

_sid!(lib::SegLibrary, sp) = (id = objectid(sp); lib.spaces[id] = sp; id)

"""
    segop_shared!(lib, spd, spf, amd, ll, secop) -> (op, f)

Library operator `op` and real factor `f` with  exact operator == f * op.
"""
function segop_shared!(lib::SegLibrary, spd, spf, amd, ll::Int, secop::AbstractVector{<:Integer})
    akey, lam = lib.normalise ? amd_key_normalised(amd, ll) : (amd_key(amd, ll), 1.0 + 0im)
    key = SegOpKey(_sid!(lib, spd), _sid!(lib, spf), akey, ll, Vector{Int}(secop))
    if haskey(lib.ops, key)
        op, lam0 = lib.ops[key]
        f = lam / lam0
        if abs(imag(f)) <= 1e-12 * abs(f)
            lib.hits += 1; real(f) != 1 && (lib.scaled_hits += 1)
            return (op, real(f))
        end
        key = SegOpKey(key.space_d, key.space_f, akey * ":phase", ll, key.secop)   # never absorb a complex factor
        haskey(lib.ops, key) && (lib.hits += 1; return (lib.ops[key][1], 1.0))
        lam = 1.0 + 0im
    end
    t = time()
    op = if lib.builder === :tiled
        tiled_seg_operator(spd, spf, amd, ll, Vector{Int}(secop);
                           tile_width=lib.tile_width, workers=lib.workers, opmat_threads=lib.opmat_threads)
    else
        BuildSegOperator(spd, spf, amd, ll, Vector{Int}(secop); num_th=lib.opmat_threads)
    end
    dt = time() - t
    lib.ops[key] = (op, lam)
    lib.bytes[key] = Base.summarysize(op)
    lib.builds += 1
    lib.build_seconds += dt
    push!(lib.log, (amd=key.amd[1:min(12, end)], ll=ll, secop=copy(key.secop), seconds=dt, bytes=lib.bytes[key]))
    (op, 1.0)
end

"""Copy of a SegOperator with every block multiplied by `f`."""
function scaled_segop(op::SegOperator{T}, f::Real) where T
    el = [begin
              B = similar(A)
              for k in eachindex(A); isassigned(A, k) && (B[k] = T(f) .* A[k]); end
              B
          end for A in op.elmat]
    SegOperator{T}(copy(op.colptr), copy(op.rowid), el)
end

"""Fetch or build one segment operator (exact values; a scaled hit returns a scaled copy)."""
function segop!(lib::SegLibrary, spd, spf, amd, ll::Int, secop::AbstractVector{<:Integer})
    op, f = segop_shared!(lib, spd, spf, amd, ll, secop)
    f == 1 ? op : scaled_segop(op, f)
end

"""Segment-operator matrix (2 × nd) for `cpd` on (space_A, space_B), as BuildSegOperators returns."""
function segops!(lib::SegLibrary, spaces::AbstractVector, cpd; spaces_f::AbstractVector=spaces)
    np = length(spaces)
    out = Matrix{SegOperator}(undef, np, length(cpd))
    for (d, c) in enumerate(cpd), p in 1:np
        out[p, d] = segop!(lib, spaces[p], spaces_f[p], c.amd[p], c.ch[1, p], c.sec[:, p])
    end
    out
end

"""
    segops_shared!(lib, spaces, cpd) -> (ops, fac)

Library operators without copies; channel d of a CompOperator built from `ops` must have
its coefficient multiplied by `fac[d]` (see `absorb!`).
"""
function segops_shared!(lib::SegLibrary, spaces::AbstractVector, cpd; spaces_f::AbstractVector=spaces)
    np = length(spaces)
    out = Matrix{SegOperator}(undef, np, length(cpd)); fac = ones(length(cpd))
    for (d, c) in enumerate(cpd), p in 1:np
        out[p, d], f = segop_shared!(lib, spaces[p], spaces_f[p], c.amd[p], c.ch[1, p], c.sec[:, p])
        fac[d] *= f
    end
    out, fac
end

"""Absorb the library scale factors into the channel coefficients of a CompOperator."""
absorb!(op, fac) = (all(==(1), fac) || (op.coeff .*= fac); op)

"""Drop every operator acting on `space` (call after its last (sA,sB) use)."""
function release!(lib::SegLibrary, space)
    id = objectid(space)
    for k in collect(keys(lib.ops))
        (k.space_d == id || k.space_f == id) || continue
        delete!(lib.ops, k); delete!(lib.bytes, k)
    end
    delete!(lib.spaces, id)
    nothing
end

resident_bytes(lib::SegLibrary) = sum(values(lib.bytes); init=0)

# ---------------------------------------------------------------------------
# Tiled, task-parallel segment-operator builder.  Mathematically identical to
# FuzzifiED 2.0.1 BuildSegOperator: for each allowed sector pair it forms
#   (stf' * O * std)[rows(lf), cols(ld)] / CG     (or the m=1 fallback),
# but (i) only for angular blocks allowed by the triangle rule, (ii) one column
# tile at a time, never materialising O*std or stf'*O*std for a whole sector, and
# (iii) with tiles distributed over Julia tasks running BLAS single-threaded.
# Live temporaries are bounded by workers*(dim_d+dim_f)*tile_width*8 bytes.

function _csc(op_mat)
    op_mat.sym_q == 0 || return nothing
    SparseMatrixCSC(op_mat.dimf, op_mat.dimd, op_mat.colptr, op_mat.rowid, op_mat.elval)
end

function _tile_tasks(factors, ptrd, tile_width)
    tasks = Tuple{Int,Int,Int}[]
    for jl in axes(factors, 2)
        any(!iszero, view(factors, :, jl)) || continue
        first_col = ptrd[jl] + 1 - ptrd[1]
        last_col = ptrd[jl+1] - ptrd[1]
        for c0 in first_col:tile_width:last_col
            push!(tasks, (jl, c0, min(c0 + tile_width - 1, last_col)))
        end
    end
    tasks
end

function _run_tiles!(blocks, factors, stf, std, op_mat, ptrf, ptrd, tile_width, workers)
    tasks = _tile_tasks(factors, ptrd, tile_width)
    isempty(tasks) && return
    A = _csc(op_mat)
    lk = ReentrantLock()
    nw = max(1, min(workers, length(tasks)))
    next = Threads.Atomic{Int}(0)
    T = eltype(std)
    @sync for _ in 1:nw
        Threads.@spawn begin
            xin = Matrix{T}(undef, size(std, 1), tile_width)
            prod = Matrix{T}(undef, A === nothing ? op_mat.dimf : size(A, 1), tile_width)
            while true
                it = Threads.atomic_add!(next, 1) + 1
                it > length(tasks) && break
                jl, c0, c1 = tasks[it]
                w = c1 - c0 + 1
                xi = view(xin, :, 1:w); copyto!(xi, view(std, :, c0:c1))
                pr = view(prod, :, 1:w)
                if A === nothing      # symmetric storage: serialised native product
                    lock(lk) do
                        copyto!(pr, *(op_mat, Matrix(xi); num_th=1))
                    end
                else
                    mul!(pr, A, xi)
                end
                first_col = ptrd[jl] + 1 - ptrd[1]
                dest_cols = (c0 - first_col + 1):(c1 - first_col + 1)
                for il in axes(blocks, 1)
                    f = factors[il, jl]
                    iszero(f) && continue
                    r0 = ptrf[il] + 1 - ptrf[1]; r1 = ptrf[il+1] - ptrf[1]
                    dest = view(blocks[il, jl], :, dest_cols)
                    mul!(dest, adjoint(view(stf, :, r0:r1)), pr)
                    dest ./= f
                end
            end
        end
    end
    nothing
end

function tiled_seg_operator(sgspd, sgspf, amd, ll::Int, secop::Vector{Int};
                            tile_width::Int=64, workers::Int=Threads.nthreads(), opmat_threads::Int=8)
    T = eltype(sgspd.sts[1])
    (amd === :Identity) && (amd = one(FuzzifiED.AngModes))
    index = 0
    colptr = zeros(Int64, size(sgspd.sec, 2) + 1); colptr[1] = 1
    rowid = Int64[]
    elmat = Matrix{Matrix{T}}[]
    modul = sgspd.sec_modul
    blas0 = BLAS.get_num_threads()
    try
        for j in axes(sgspd.sec, 2)
            secd = sgspd.sec[:, j]; std = sgspd.sts[j]; md = secd[2]
            std1 = md == 0 ? sgspd.sts1[j] : nothing
            for i in axes(sgspf.sec, 2)
                secf = sgspf.sec[:, i]
                FuzzifiED.SO3lver.EquivSec(secd .+ secop, secf, modul) || continue
                index += 1; push!(rowid, i)
                mf = secf[2]; mm = mf - md
                stf = sgspf.sts[i]
                hmt = Matrix{Matrix{T}}(undef, length(sgspf.l_rng[i]), length(sgspd.l_rng[j]))
                fac = zeros(Float64, size(hmt)); fac1 = zeros(Float64, size(hmt))
                for jl in eachindex(sgspd.l_rng[j])
                    ld = sgspd.l_rng[j][jl]
                    ncol = sgspd.ptr_st[j][jl+1] - sgspd.ptr_st[j][jl]
                    for il in eachindex(sgspf.l_rng[i])
                        lf = sgspf.l_rng[i][il]
                        (ll < abs(ld - lf) || ll > ld + lf) && continue
                        nrow = sgspf.ptr_st[i][il+1] - sgspf.ptr_st[i][il]
                        hmt[il, jl] = Matrix{T}(undef, nrow, ncol)
                        f3 = FuzzifiED.SO3lver.clebschgordan(ld / 2, md / 2, ll / 2, mm / 2, lf / 2)
                        if f3 != 0
                            fac[il, jl] = f3
                        else
                            f31 = FuzzifiED.SO3lver.clebschgordan(ld / 2, md / 2 + 1, ll / 2, mm / 2 - 1, lf / 2)
                            fac1[il, jl] = sqrt(ld / 2 * (ld / 2 + 1)) * f31
                        end
                    end
                end
                BLAS.set_num_threads(blas0)
                if any(!iszero, fac)
                    op = Operator(sgspd.bs[j], sgspf.bs[i], GetComponent(amd, ll / 2, mm / 2))
                    om = OpMat(op; num_th=opmat_threads, disp_std=false)
                    BLAS.set_num_threads(1)
                    _run_tiles!(hmt, fac, stf, std, om, sgspf.ptr_st[i], sgspd.ptr_st[j], tile_width, workers)
                end
                if md == 0 && mf == 0 && ll > 0 && any(!iszero, fac1)
                    BLAS.set_num_threads(blas0)
                    op1 = Operator(sgspd.bs1[j], sgspf.bs[i], GetComponent(amd, ll / 2, mm / 2 - 1))
                    om1 = OpMat(op1; num_th=opmat_threads, disp_std=false)
                    BLAS.set_num_threads(1)
                    _run_tiles!(hmt, fac1, stf, std1, om1, sgspf.ptr_st[i], sgspd.ptr_st[j], tile_width, workers)
                end
                push!(elmat, hmt)
            end
            colptr[j+1] = index + 1
        end
    finally
        BLAS.set_num_threads(blas0)
    end
    SegOperator{T}(colptr, rowid, elmat)
end

"""Max |a-b| / max(1,|a|) over every assigned reduced block of two SegOperators."""
function segop_difference(a, b)
    bad = (worst=Inf, blocks=-1)
    a.colptr == b.colptr && a.rowid == b.rowid || return bad
    worst = 0.0; nblk = 0
    for e in eachindex(a.elmat)
        A = a.elmat[e]; B = b.elmat[e]
        size(A) == size(B) || return bad
        for k in eachindex(A)
            ia = isassigned(A, k); ib = isassigned(B, k)
            ia == ib || return bad
            ia || continue
            size(A[k]) == size(B[k]) || return bad
            isempty(A[k]) && continue
            nblk += 1
            scale = max(1.0, maximum(abs, A[k]))
            worst = max(worst, maximum(abs.(A[k] .- B[k])) / scale)
        end
    end
    (worst=worst, blocks=nblk)
end
