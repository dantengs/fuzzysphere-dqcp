# OPE stage against the frozen low-N references of half_model.jl (g = 0.9437):
#   N4: f_phiphiJ = 0.7343328726999043, f_phiphiT = 1.1014564638818614
#   N6: f_phiphiJ = 0.7504299632109654, f_phiphiT = 1.1515489286980547
# usage: julia --threads=T t9_ope.jl out
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, TOML, Printf
FuzzifiED.SilentStd = true
REF = Dict(4 => Dict("f_J" => 0.7343328726999043, "f_T" => 1.1014564638818614),
           6 => Dict("f_J" => 0.7504299632109654, "f_T" => 1.1515489286980547))
ok = true
for N in (4, 6)
    d = joinpath(ARGS[1], "ope_N$N"); rm(d; recursive=true, force=true)
    conf = Dict("run" => Dict("N" => N, "g" => 0.9437, "out" => d, "stages" => ["basis", "spectrum", "targets", "kk", "ope", "summary"]),
                "spectrum" => Dict("nev" => 12, "exchange" => true, "tol" => 1e-11),
                "threads" => Dict("native" => min(8, Threads.nthreads()), "tile" => 16),
                "targets" => Dict("initial_nev" => 12, "max_nev" => 80),
                "ope" => Dict("reference" => REF[N]))
    r = try
        run_pipeline(conf); TOML.parsefile(joinpath(d, "ope", "ope.toml"))
    catch err
        println("OPE error N$N: ", sprint(showerror, err)); nothing
    end
    pass = r !== nothing && get(r, "reference_check", "") == "passed"
    global ok &= pass
    r === nothing || @printf("N=%d f_J=%.13f (ref %.13f) f_T=%.13f (ref %.13f) phi-energy(half-int)=%.2e  phiphiS=%s T14=%s stress=%s\n", N,
        r["f_phi_phi_J"], REF[N]["f_J"], r["f_phi_phi_T_half"], REF[N]["f_T"], get(r, "half_vs_integer_phi_energy_difference", NaN),
        get(r, "raw:phi_phi_S", "-"), get(r, "raw:phi_phi_T14", "-"), get(r, "raw:phi_phi_stress", "-"))
end
println(ok ? "OPE_REFERENCE_PASS" : "OPE_REFERENCE_FAIL")
