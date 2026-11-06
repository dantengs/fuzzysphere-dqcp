# High singlet tower (S, S', S'', ...) with the exchange-even (0,0) sector only, compared with a
# both-parity reference run: KK-projected C2=0 L0 states and phi x phi tower must agree.
# Both runs keep complete windows (small N, dense), so the KK subspaces are identical; for finite
# windows the projection depends on which sources are guarded and runs are not comparable state by state.
# usage: julia --threads=T t10_highscalar.jl N out        (N <= 6: every (0,0) block <= 400 states)
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, TOML, Printf
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); root = ARGS[2]; g = 0.95
th = Dict("native" => min(8, Threads.nthreads()), "tile" => 16)
function conf(out, hi::Bool)
    Dict("run" => Dict("N" => N, "g" => g, "out" => out, "stages" => ["basis", "spectrum", "targets", "kk", "ope"]),
         "spectrum" => Dict{String,Any}("blocks" => [[0, 0, 0], [0, 0, 1], [0, 1, 1]], "lmax" => 0, "nev" => 12, "exchange" => hi,
             "tol" => 1e-11, "block_nev" => Dict("s0_s0_L0" => 400, "s0_s0_L1" => 400),
             (hi ? ("parity_nev" => Dict("s0_s0_L0" => [400, 6], "s0_s0_L1" => [400, 0]),) : ("parity" => "both",))...),
         "threads" => th, "targets" => Dict{String,Any}("initial_nev" => 400, "max_nev" => 400, (hi ? ("target_parity" => "plus",) : ())...),
         "kk" => Dict("pairs" => [[0, 0]], "lmax" => 0), "ope" => Dict("half" => false))
end
dH = joinpath(root, "hi_N$N"); dR = joinpath(root, "ref_N$N"); rm(dH; recursive=true, force=true); rm(dR; recursive=true, force=true)
mH = run_pipeline(conf(dH, true)); mR = run_pipeline(conf(dR, false))
rd(d) = [split(l, ',') for l in readlines(joinpath(d, "kk", "projected_states.csv"))[2:end]]
sing(d, t) = sort([parse(Float64, strip(r[6], '"')) for r in rd(d) if strip(r[2], '"') == "0" && strip(r[3], '"') == "0" && strip(r[5], '"') == t])
ok = true
for t in ("0.05", "0.1", "0.25", "0.5")
    A = sing(dH, t); B = sing(dR, t); m = min(length(A), length(B))
    d = maximum(abs.(A[1:m] - B[1:m]); init=0.0)
    @printf("threshold %s: singlet primaries hi=%d ref=%d, common max|dE|=%.2e\n", t, length(A), length(B), d)
    global ok &= d < 1e-8 && m >= 2
end
tw(d) = [split(l, ',') for l in readlines(joinpath(d, "ope", "phi_phi_towers.csv"))[2:end] if startswith(l, "\"projected_t0.25") && occursin("C2_0", l)]
tH = sort([(parse(Float64, strip(r[5], '"')), parse(Float64, strip(r[9], '"'))) for r in tw(dH)])
tR = sort([(parse(Float64, strip(r[5], '"')), parse(Float64, strip(r[9], '"'))) for r in tw(dR)])
m = min(length(tH), length(tR)); dt = maximum(abs(tH[i][2] - tR[i][2]) for i in 1:m; init=0.0)
@printf("phi x phi -> projected singlets (t=0.25): hi=%d ref=%d max|df|=%.2e\n", length(tH), length(tR), dt)
for i in 1:min(6, m); @printf("   E=%.8f  f=%.6e\n", tH[i]...); end
global ok &= dt < 1e-8
println(ok ? "HIGHSCALAR_PASS N=$N" : "HIGHSCALAR_FAIL N=$N")
