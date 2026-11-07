# Load-balanced composite-operator action for two-part (2+2 segment) CompOperators.
#
# FuzzifiED's (v0.0.4 typed, channel-owned) action gives each destination channel to one
# task.  On large blocks a few destination channels dominate, so the critical path stops
# shrinking beyond ~64-128 workers.  Here every destination channel (m2 x m1 after
# reshaping) is additionally split into column ranges J of its part-1 index:
#     Y[:, J] += c * A2 * (X * A1[J, :]^T)
# Every incoming term contributes to every column range with the same A1/A2 blocks, so the
# split is exact, needs no reduction, and output ranges stay disjoint (column-major).
# The CompOperator object and its data are only read.

struct _MVTask
    isec::Int
    ich::Int
    j0::Int
    j1::Int
    cost::Float64
end

mutable struct MatvecPlan
    tasks::Vector{_MVTask}
    incoming_ptr::Vector{Int}
    incoming_wk::Vector{Int}
    maxscr::Int
    max_task_share::Float64
end

function _dims_part(cpsp, isec, ich, p)
    sec = cpsp.idsec[p, isec]; l = cpsp.chs[isec][ich][1, p]
    il = cpsp.sgsp[p].l_lookup[sec][l]
    cpsp.sgsp[p].ptr_st[sec][il+1] - cpsp.sgsp[p].ptr_st[sec][il]
end

"""Build (once per operator) a cost-balanced task list with `granularity` tasks per worker."""
function matvec_plan(cpop; workers::Int=Threads.nthreads(), granularity::Int=4)
    cpop.cpspd.np == 2 || error("matvec_plan supports two-part composite operators")
    fsp = cpop.cpspf; dsp = cpop.cpspd
    nsec = size(fsp.idsec, 2)
    ptr = zeros(Int, nsec + 1); ptr[1] = 1
    cnt = zeros(Int, nsec)
    for (_, d, e) in cpop.wklist; cnt[cpop.rowid[d][e]] += 1; end
    for s in 1:nsec; ptr[s+1] = ptr[s] + cnt[s]; end
    inc = Vector{Int}(undef, length(cpop.wklist)); nxt = copy(ptr[1:nsec])
    for (iw, (_, d, e)) in enumerate(cpop.wklist)
        s = cpop.rowid[d][e]; inc[nxt[s]] = iw; nxt[s] += 1
    end
    # per destination channel cost = sum over incoming (work, jch) of n2*n1*m1 + m2*n2*m1
    chans = Tuple{Int,Int,Int,Float64}[]   # isec, ich, m1, cost
    maxscr = 1
    for isec in 1:nsec
        ptr[isec] == ptr[isec+1] && continue
        for ich in eachindex(fsp.chs[isec])
            fsp.ptr_st[isec][ich] == fsp.ptr_st[isec][ich+1] && continue
            m1 = _dims_part(fsp, isec, ich, 1); m2 = _dims_part(fsp, isec, ich, 2)
            c = 0.0
            for k in ptr[isec]:ptr[isec+1]-1
                jsec, d, e = cpop.wklist[inc[k]]
                m9 = cpop.mat9j[d][e]
                for jch in eachindex(dsp.chs[jsec])
                    abs(m9[ich, jch]) < sqrt(eps(Float64)) && continue
                    n1 = _dims_part(dsp, jsec, jch, 1); n2 = _dims_part(dsp, jsec, jch, 2)
                    c += n2 * n1 * m1 + m2 * n2 * m1
                    maxscr = max(maxscr, n2 * m1)
                end
            end
            push!(chans, (isec, ich, m1, c))
        end
    end
    total = sum(x[4] for x in chans; init=0.0)
    target = max(total / (granularity * max(workers, 1)), 1.0)
    tasks = _MVTask[]
    for (isec, ich, m1, c) in chans
        nsplit = clamp(ceil(Int, c / target), 1, m1)
        step = cld(m1, nsplit)
        for j0 in 1:step:m1
            j1 = min(j0 + step - 1, m1)
            push!(tasks, _MVTask(isec, ich, j0, j1, c * (j1 - j0 + 1) / m1))
        end
    end
    sort!(tasks; by=t -> -t.cost)          # longest-processing-time first
    share = isempty(tasks) ? 0.0 : tasks[1].cost / total
    MatvecPlan(tasks, ptr, inc, maxscr, share)
end

"""y = cpop * x with the column-split plan (BLAS forced to one thread)."""
function tcc_mul!(y::Vector{T}, cpop, x::Vector{T}, plan::MatvecPlan; workers::Int=Threads.nthreads()) where T
    fsp = cpop.cpspf; dsp = cpop.cpspd
    fill!(y, zero(T))
    tasks = plan.tasks; isempty(tasks) && return y
    nb = BLAS.get_num_threads(); BLAS.set_num_threads(1)
    nextt = Threads.Atomic{Int}(0)
    try
        @sync for _ in 1:min(workers, length(tasks))
            Threads.@spawn begin
                scr = Vector{T}(undef, plan.maxscr)
                while true
                    it = Threads.atomic_add!(nextt, 1) + 1
                    it > length(tasks) && break
                    t = tasks[it]
                    isec = t.isec; ich = t.ich
                    li1 = fsp.chs[isec][ich][1, 1]; li2 = fsp.chs[isec][ich][1, 2]
                    il1 = fsp.sgsp[1].l_lookup[fsp.idsec[1, isec]][li1]
                    il2 = fsp.sgsp[2].l_lookup[fsp.idsec[2, isec]][li2]
                    m1 = _dims_part(fsp, isec, ich, 1); m2 = _dims_part(fsp, isec, ich, 2)
                    off = fsp.ptr_st[isec][ich] - 1
                    J = t.j0:t.j1; w = length(J)
                    Y = reshape(view(y, off + (t.j0 - 1) * m2 + 1 : off + t.j1 * m2), m2, w)
                    for k in plan.incoming_ptr[isec]:plan.incoming_ptr[isec+1]-1
                        jsec, d, e = cpop.wklist[plan.incoming_wk[k]]
                        coeff = cpop.coeff[d]
                        id1 = cpop.idel[d][1, e]; id2 = cpop.idel[d][2, e]
                        m9 = cpop.mat9j[d][e]
                        for jch in eachindex(dsp.chs[jsec])
                            f = m9[ich, jch]
                            abs(f) < sqrt(eps(Float64)) && continue
                            lj1 = dsp.chs[jsec][jch][1, 1]; lj2 = dsp.chs[jsec][jch][1, 2]
                            jl1 = dsp.sgsp[1].l_lookup[dsp.idsec[1, jsec]][lj1]
                            jl2 = dsp.sgsp[2].l_lookup[dsp.idsec[2, jsec]][lj2]
                            A1 = cpop.sgop[1, d].elmat[id1][il1, jl1]
                            A2 = cpop.sgop[2, d].elmat[id2][il2, jl2]
                            n1 = size(A1, 2); n2 = size(A2, 2)
                            jr = dsp.ptr_st[jsec][jch] : dsp.ptr_st[jsec][jch+1] - 1
                            X = reshape(view(x, jr), n2, n1)
                            W = reshape(view(scr, 1:n2*w), n2, w)
                            mul!(W, X, transpose(view(A1, J, :)))
                            mul!(Y, A2, W, T(coeff * f), one(T))
                        end
                    end
                end
            end
        end
    finally
        BLAS.set_num_threads(nb)
    end
    y
end

"""
    tcc_mul_block!(Y, cpop, X, plan; workers)

Y[:, k] = cpop * X[:, k] for all columns k at once.  Every (destination task, incoming term)
reads its two segment blocks A1, A2 once and applies them to all b vectors while they are
cache-resident, so operator data are streamed once per block instead of once per vector.
Exact: the same contraction as the single-vector kernel, column by column.
"""
function tcc_mul_block!(Y::Matrix{T}, cpop, X::Matrix{T}, plan::MatvecPlan;
                        workers::Int=Threads.nthreads()) where T
    fsp = cpop.cpspf; dsp = cpop.cpspd
    b = size(X, 2); size(Y, 2) == b || error("block width mismatch")
    fill!(Y, zero(T))
    tasks = plan.tasks; isempty(tasks) && return Y
    nb = BLAS.get_num_threads(); BLAS.set_num_threads(1)
    nextt = Threads.Atomic{Int}(0)
    try
        @sync for _ in 1:min(workers, length(tasks))
            Threads.@spawn begin
                scr = Vector{T}(undef, plan.maxscr)
                while true
                    it = Threads.atomic_add!(nextt, 1) + 1
                    it > length(tasks) && break
                    t = tasks[it]
                    isec = t.isec; ich = t.ich
                    il1 = fsp.sgsp[1].l_lookup[fsp.idsec[1, isec]][fsp.chs[isec][ich][1, 1]]
                    il2 = fsp.sgsp[2].l_lookup[fsp.idsec[2, isec]][fsp.chs[isec][ich][1, 2]]
                    m2 = _dims_part(fsp, isec, ich, 2)
                    off = fsp.ptr_st[isec][ich] - 1
                    J = t.j0:t.j1; w = length(J)
                    yr = off + (t.j0 - 1) * m2 + 1 : off + t.j1 * m2
                    for k in plan.incoming_ptr[isec]:plan.incoming_ptr[isec+1]-1
                        jsec, d, e = cpop.wklist[plan.incoming_wk[k]]
                        coeff = cpop.coeff[d]
                        id1 = cpop.idel[d][1, e]; id2 = cpop.idel[d][2, e]
                        m9 = cpop.mat9j[d][e]
                        for jch in eachindex(dsp.chs[jsec])
                            f = m9[ich, jch]
                            abs(f) < sqrt(eps(Float64)) && continue
                            jl1 = dsp.sgsp[1].l_lookup[dsp.idsec[1, jsec]][dsp.chs[jsec][jch][1, 1]]
                            jl2 = dsp.sgsp[2].l_lookup[dsp.idsec[2, jsec]][dsp.chs[jsec][jch][1, 2]]
                            A1 = cpop.sgop[1, d].elmat[id1][il1, jl1]
                            A2 = cpop.sgop[2, d].elmat[id2][il2, jl2]
                            n1 = size(A1, 2); n2 = size(A2, 2)
                            jr = dsp.ptr_st[jsec][jch] : dsp.ptr_st[jsec][jch+1] - 1
                            A1J = view(A1, J, :)
                            W = reshape(view(scr, 1:n2*w), n2, w)
                            cf = T(coeff * f)
                            for c in 1:b
                                Xc = reshape(view(X, jr, c), n2, n1)
                                mul!(W, Xc, transpose(A1J))
                                mul!(reshape(view(Y, yr, c), m2, w), A2, W, cf, one(T))
                            end
                        end
                    end
                end
            end
        end
    finally
        BLAS.set_num_threads(nb)
    end
    Y
end


"""
    act_block!(Y, A, X; workers, plan=nothing) -> Y

v0.1.17: Y[:, j] = A * X[:, j] with the faster kernel for the size of A.  Up to `NATIVE_DIM_MAX`
(max of the two dimensions) every column uses FuzzifiED's channel-owned action with all workers --
B14/B16 at N9 (D 0.9-2.5e5): 4.5-8x faster per vector than the column-split `tcc_mul_block!`, and
at N11 (D 1.7e7) the single-vector kernels are at par.  Above it (N12 blocks, D ~1.6e8) the
column-split kernel was 1.3x faster at 128 workers and is kept.  The switch is by block dimension
(not by N: N12 also has many small blocks): default 2e7, just above the N11 point where the two
kernels are at par; `threads.native_dim_max` in run.toml overrides it.  Results are identical up to
the floating-point summation order.
"""
const NATIVE_DIM_MAX = Ref(20_000_000)
function act_block!(Y::Matrix{T}, A, X::AbstractMatrix{T}; workers::Int=Threads.nthreads(), plan=nothing) where T
    size(X, 2) == size(Y, 2) || error("block width mismatch")
    if max(size(X, 1), size(Y, 1)) <= NATIVE_DIM_MAX[]
        x = Vector{T}(undef, size(X, 1))
        for j in axes(X, 2)
            copyto!(x, view(X, :, j))
            Y[:, j] = *(A, x; num_th=workers)
        end
    else
        tcc_mul_block!(Y, A, Matrix(X), plan === nothing ? matvec_plan(A; workers=workers) : plan; workers)
    end
    Y
end
