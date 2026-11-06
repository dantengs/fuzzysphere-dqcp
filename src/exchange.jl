# Segment-exchange (A<->B) parity for diagonal blocks (s,s,L)  (Tcc C2).
#
# X is taken from FuzzifiED's PermFirstSecondSegs, which permutes the two segment
# factors and applies the SO(3) recoupling sign.  In integer-spin blocks every
# segment has flavour S^z=0, hence even particle number, so the fermionic
# reordering sign (-1)^(nA nB) is +1 and X is exactly the physical exchange.
# (Half-integer (1/2,1/2) blocks carry an extra global (-1); not used here.)
#
# ExchangeSector stores the sigma=+-1 eigenbasis of X as an isometry
#   e_k = (|i> + c_k |j>)/sqrt(2)   for X|i> = p_i |j>, j != i, c_k = sigma p_i
#   e_k = |i>                        for X|i> = sigma |i>
# so Krylov vectors live in the half-length reduced space; H acts as
# reduce(H * expand(r)).  This is exact when [H,X]=0 and changes no kernel.

struct ExchangeSector
    dim::Int
    sigma::Int
    first::Vector{Int32}
    second::Vector{Int32}      # 0 for fixed points
    c::Vector{Int8}
end

"""Index/sign map of X on a CompSpace: X e_i = sgn[i] e_{perm[i]}."""
function exchange_map(cpsp)
    D = cpsp.dim
    D < 2^31 || error("dimension too large for Int32 maps")
    y = PermFirstSecondSegs(cpsp, collect(1.0:Float64(D)))   # y[j] = p_i * i  where X e_i = p_i e_j
    perm = zeros(Int32, D); sgn = zeros(Int8, D)
    for j in 1:D
        v = y[j]; i = Int(abs(v))
        (abs(v) == i && 1 <= i <= D && perm[i] == 0) || error("exchange map is not a signed permutation")
        perm[i] = j; sgn[i] = v > 0 ? 1 : -1
    end
    for i in 1:D   # involution check X^2 = 1
        j = perm[i]
        (perm[j] == i && sgn[i] * sgn[j] == 1) || error("exchange map is not an involution")
    end
    perm, sgn
end

function ExchangeSector(perm::Vector{Int32}, sgn::Vector{Int8}, sigma::Int)
    sigma in (1, -1) || error("sigma must be +-1")
    D = length(perm)
    f = Int32[]; s = Int32[]; c = Int8[]
    for i in 1:D
        j = perm[i]
        if j == i
            sgn[i] == sigma && (push!(f, i); push!(s, 0); push!(c, 1))
        elseif i < j
            push!(f, i); push!(s, j); push!(c, Int8(sigma * sgn[i]))
        end
    end
    ExchangeSector(D, sigma, f, s, c)
end

Base.length(x::ExchangeSector) = length(x.first)

function sector_reduce!(r::AbstractVector{T}, x::ExchangeSector, v::AbstractVector{T}) where T
    h = T(1 / sqrt(2))
    @inbounds Threads.@threads for k in eachindex(x.first)
        i = x.first[k]; j = x.second[k]
        r[k] = j == 0 ? v[i] : h * (v[i] + x.c[k] * v[j])
    end
    r
end

function sector_expand!(v::AbstractVector{T}, x::ExchangeSector, r::AbstractVector{T}) where T
    h = T(1 / sqrt(2))
    fill!(v, zero(T))
    @inbounds Threads.@threads for k in eachindex(x.first)
        i = x.first[k]; j = x.second[k]
        if j == 0
            v[i] = r[k]
        else
            v[i] = h * r[k]; v[j] = x.c[k] * h * r[k]
        end
    end
    v
end

"""Return a closure r -> reduce(H*expand(r)) with persistent full-length buffers."""
function reduced_action(H, x::ExchangeSector; num_th::Int=FuzzifiED.NumThreads)
    buf = zeros(Float64, x.dim)
    function act(r)
        sector_expand!(buf, x, r)
        y = *(H, buf; num_th)
        out = similar(r)
        sector_reduce!(out, x, y)
    end
    act
end
