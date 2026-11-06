# T2: exchange-parity-resolved Lanczos vs full-block Lanczos on (s,s,L) blocks.
# usage: julia --threads=T t2_exchange_solve.jl N segckpt|fresh g k s Ls outdir
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, FuzzifiED.SO3lver, LinearAlgebra, Random, TOML, KrylovKit
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); src = ARGS[2]; g = parse(Float64, ARGS[3]); k = parse(Int, ARGS[4])
s = parse(Int, ARGS[5]); Ls = parse.(Int, split(ARGS[6], ',')); out = ARGS[7]
action = parse(Int, get(ENV, "T2_ACTION", "64"))
mkpath(out); st = Stage(joinpath(out, "stages.tsv"))
FuzzifiED.NumThreads = 8; BLAS.set_num_threads(16)
spaces = src == "fresh" ? build_segment_spaces(N) : load_segment_spaces(src).spaces
tpl = hamiltonian_template(N)
lib = SegLibrary(builder=:tiled, tile_width=64, workers=Threads.nthreads(), opmat_threads=8)
ss = [spaces[s+1], spaces[s+1]]
sH = stage!(st, "seg_H") do; segops!(lib, ss, tpl.template); end
sC = stage!(st, "seg_C2") do; segops!(lib, ss, tpl.c2); end
res = Dict{String,Any}[]
for L in Ls
    cp = BuildCompSpace(ss, [2N, 0, 0], 2L); D = cp.dim
    H = set_g!(BuildCompOperator(cp, tpl.template, sH; num_th=action), tpl, g)
    C = BuildCompOperator(cp, tpl.c2, sC; num_th=action)
    BLAS.set_num_threads(1)
    perm, sgn = exchange_map(cp)
    kf = min(k, D)
    alg(kk, d) = Lanczos(krylovdim=min(d, max(2kk + 8, 32)), tol=1e-10, maxiter=500, verbosity=0)
    calls = Ref(0)
    Ef, Vf, inf = stage!(st, "full_L$L"; detail="D=$D k=$kf") do
        eigsolve(v -> (calls[] += 1; *(H, v; num_th=action)), normalize(randn(MersenneTwister(7), D)), kf, :SR, alg(kf, D))
    end
    full_calls = calls[]
    sect = Dict{Int,Any}()
    for sig in (1, -1)
        x = ExchangeSector(perm, sgn, sig); n = length(x); n == 0 && continue
        act = reduced_action(H, x; num_th=action); kk = min(k, n); calls[] = 0
        E, V, info = stage!(st, "sector$(sig)_L$L"; detail="D=$n k=$kk") do
            eigsolve(r -> (calls[] += 1; act(r)), normalize(randn(MersenneTwister(8 + sig), n)), kk, :SR, alg(kk, n))
        end
        buf = zeros(D)
        c2 = Float64[]; rh = Float64[]
        for i in 1:kk
            sector_expand!(buf, x, V[i]); push!(c2, dot(buf, *(C, buf; num_th=action)))
            push!(rh, norm(*(H, buf; num_th=action) - E[i] * buf))
        end
        sect[sig] = (E=E[1:kk], c2=c2, rh=rh, calls=calls[], dim=n)
    end
    merged = sort(vcat([(e, sig) for sig in keys(sect) for e in sect[sig].E]...); by=first)
    m = min(kf, length(merged))
    # Compare only the window closed in both: energies below min(max returned in each sector)
    top = minimum(maximum(sect[sig].E) for sig in keys(sect))
    cmpE = [e for (e, _) in merged if e <= top]
    nc = min(length(cmpE), kf)
    dE = maximum(abs.(cmpE[1:nc] - sort(Ef)[1:nc]); init=0.0)
    row = Dict{String,Any}("L"=>L, "dim"=>D, "k"=>kf, "full_E"=>sort(Ef)[1:kf], "full_calls"=>full_calls,
                           "compared"=>nc, "max_dE"=>dE)
    for (sig, v) in sect
        row["sector$(sig)"] = Dict("dim"=>v.dim, "E"=>v.E, "C2"=>round.(v.c2; digits=6), "max_rH"=>maximum(v.rh), "calls"=>v.calls)
    end
    push!(res, row)
    println("L=$L D=$D compared=$nc max_dE=$dE full_calls=$full_calls")
end
open(io -> TOML.print(io, Dict("N"=>N, "g"=>g, "s"=>s, "blocks"=>res); sorted=true), joinpath(out, "summary.toml"), "w")
println("T2_DONE")
