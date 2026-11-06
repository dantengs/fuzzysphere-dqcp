# Optional runtime replacement of FuzzifiED.SimplifyTerms by the deterministic batched
# map/reduce version (a term-mapreduce patch of the earlier production code).  Package files stay untouched;
# call Tcc.enable_fast_simplify!() explicitly and record it in run metadata.
const FAST_SIMPLIFY_SRC = raw"""
const _SIMPLIFY_SERIAL_THRESHOLD = 16_384
const _SIMPLIFY_BATCH_SIZE = 4_096
const _SIMPLIFY_MAX_WORKERS = 8

function _normal_order_into!(dest :: Dict{Vector{Int64}, ComplexF64}, tms :: Terms,
                             first :: Int, last :: Int)
    for i in first : last
        for tm in NormalOrder(tms[i])
            dest[tm.cstr] = get(dest, tm.cstr, 0.0 + 0.0im) + tm.coeff
        end
    end
    return dest
end

function _merge_normal_order!(dest :: Dict{Vector{Int64}, ComplexF64},
                              src :: Dict{Vector{Int64}, ComplexF64})
    for (cstr, coeff) in src
        dest[cstr] = get(dest, cstr, 0.0 + 0.0im) + coeff
    end
    return dest
end

function _term_key_less(a :: Vector{Int64}, b :: Vector{Int64})
    for i in 1 : min(length(a), length(b))
        a[i] == b[i] || return a[i] < b[i]
    end
    return length(a) < length(b)
end

function SimplifyTerms(tms :: Terms ; cutoff :: Float64 = 1E-14) :: Terms
    combined = Dict{Vector{Int64}, ComplexF64}()
    n = length(tms)
    if n < _SIMPLIFY_SERIAL_THRESHOLD
        _normal_order_into!(combined, tms, 1, n)
    else
        # Fixed batches make floating-point reduction independent of thread count.
        nbatches = cld(n, _SIMPLIFY_BATCH_SIZE)
        nworkers = min(Threads.nthreads(), _SIMPLIFY_MAX_WORKERS, cld(n, 8_192))
        for wave_start in 1 : nworkers : nbatches
            wave_end = min(wave_start + nworkers - 1, nbatches)
            tasks = map(wave_start : wave_end) do batch
                Threads.@spawn begin
                    first = (batch - 1) * _SIMPLIFY_BATCH_SIZE + 1
                    last = min(batch * _SIMPLIFY_BATCH_SIZE, n)
                    _normal_order_into!(Dict{Vector{Int64}, ComplexF64}(), tms, first, last)
                end
            end
            # Fetch in batch order, so scheduling cannot change coefficient sums.
            for task in tasks
                _merge_normal_order!(combined, fetch(task))
            end
        end
    end
    keys_sorted = collect(keys(combined))
    sort!(keys_sorted; lt = _term_key_less)
    return Term[Term(combined[cstr], cstr) for cstr in keys_sorted
                if abs(combined[cstr]) > cutoff]
end
"""
const FAST_SIMPLIFY_ENABLED = Ref(false)
function enable_fast_simplify!()
    FAST_SIMPLIFY_ENABLED[] && return true
    Core.eval(FuzzifiED, Meta.parseall(FAST_SIMPLIFY_SRC))
    FAST_SIMPLIFY_ENABLED[] = true
end
