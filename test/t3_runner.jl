# T3: end-to-end Tcc spectrum vs frozen v0.0.5 spectrum (raw_spectrum_h.tsv).
# usage: julia --threads=T t3_runner.jl N g nev source_root outdir exchange(0/1)
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, Printf, DelimitedFiles
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); g = parse(Float64, ARGS[2]); nev = parse(Int, ARGS[3]); src = ARGS[4]; out = ARGS[5]
useX = ARGS[6] == "1"
blocks = [[a, b, L] for a in 0:2 for b in a:2 for L in 0:2]
cfg = Dict{String,Any}("N"=>N, "g"=>g, "nev"=>nev, "out"=>out, "blocks"=>blocks, "exchange"=>useX,
    "segment_checkpoint"=>joinpath(src, "segment_spaces.ckpt"), "native"=>8, "action"=>parse(Int, get(ENV, "TCC_ACTION", "64")), "tile"=>64, "build_blas"=>16,
    "tol"=>parse(Float64, get(ENV, "TCC_TOL", "1e-10")))
t = time(); res = run_spectrum(cfg); wall = time() - t
ref = readdlm(joinpath(src, "raw_spectrum_h.tsv"), '\t'; header=true)[1]
worstE = 0.0; worstC = 0.0; ncmp = 0
r_dim(mine, x) = first(r.sector_dim for r in mine if r.x == x)
for a in 0:2, b in a:2, L in 0:2
    R = sort([Float64(ref[i, 1]) for i in axes(ref, 1) if ref[i, 2] == L && ref[i, 3] == a && ref[i, 4] == b])
    mine = [r for r in res.rows if r.L == L && r.sa == a && r.sb == b]
    (isempty(R) || isempty(mine)) && (println("s$a$b L$L skipped ours=$(length(mine)) ref=$(length(R))"); continue)
    # closed window of ours: per sector the top returned state is unguarded
    tops = Float64[]
    for x in unique(r.x for r in mine)
        es = [r.energy for r in mine if r.x == x]; length(es) < r_dim(mine, x) && push!(tops, maximum(es))
    end
    lim = min(isempty(tops) ? Inf : minimum(tops), maximum(R))
    E = sort([r.energy for r in mine if r.energy < lim - 1e-9])
    Rr = [e for e in R if e < lim - 1e-9]
    n = min(length(E), length(Rr))
    d = n == 0 ? 0.0 : maximum(abs.(E[1:n] - Rr[1:n]))
    global worstE = max(worstE, d); global ncmp += n
    @printf("s%d%d L%d compared=%d max|dE|=%.2e  ours=%d ref=%d\n", a, b, L, n, d, length(mine), length(R))
end
maxrh = maximum(r.rh for r in res.rows); maxrc = maximum(r.rc for r in res.rows)
println(@sprintf("T3 N=%d exchange=%s wall=%.1f s compared=%d worst_dE=%.2e max_rH=%.2e max_rC=%.2e builds=%d hits=%d",
                 N, useX, wall, ncmp, worstE, maxrh, maxrc, res.library.builds, res.library.hits))
println(worstE < 1e-7 && maxrh < 1e-6 && maxrc < 1e-5 ? "T3_PASS" : "T3_FAIL")
