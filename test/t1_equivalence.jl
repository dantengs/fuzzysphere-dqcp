# T1: library + tiled builder vs upstream BuildSegOperators, exchange parity checks.
# usage: julia --threads=T t1_equivalence.jl N g tile_width outdir
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, FuzzifiED.SO3lver, LinearAlgebra, Random, Printf, TOML
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); g = parse(Float64, ARGS[2]); tw = parse(Int, ARGS[3]); out = ARGS[4]
mkpath(out); st = Stage(joinpath(out, "stages.tsv"))
FuzzifiED.NumThreads = min(8, Threads.nthreads())
spaces = stage!(st, "segment_spaces") do; build_segment_spaces(N); end
tpl = stage!(st, "terms") do; hamiltonian_template(N); end
sec_tot = [2N, 0, 0]
lib = SegLibrary(builder=:tiled, tile_width=tw, workers=Threads.nthreads(), opmat_threads=FuzzifiED.NumThreads)
pairs = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
rows = Dict{String,Any}[]; worst = Dict("segop"=>0.0, "H"=>0.0, "C2"=>0.0, "commH"=>0.0, "commC2"=>0.0, "spec"=>0.0)
ref_builds = 0
rng = MersenneTwister(20261011)
for (a, b) in pairs
    ss = [spaces[a+1], spaces[b+1]]
    rH = stage!(st, "ref_seg_H_$a$b") do; BuildSegOperators(ss, tpl.template; num_th=FuzzifiED.NumThreads); end
    rC = stage!(st, "ref_seg_C2_$a$b") do; BuildSegOperators(ss, tpl.c2; num_th=FuzzifiED.NumThreads); end
    global ref_builds += 2 * (length(tpl.template) + length(tpl.c2))
    nH = stage!(st, "tcc_seg_H_$a$b") do; segops!(lib, ss, tpl.template); end
    nC = stage!(st, "tcc_seg_C2_$a$b") do; segops!(lib, ss, tpl.c2); end
    for (R, Nw) in ((rH, nH), (rC, nC)), k in eachindex(R)
        d = segop_difference(R[k], Nw[k]); worst["segop"] = max(worst["segop"], d.worst)
    end
    for L in 0:2
        cp = BuildCompSpace(ss, sec_tot, 2L); D = cp.dim; D == 0 && continue
        Hr = set_g!(BuildCompOperator(cp, tpl.template, rH), tpl, g)
        Hn = set_g!(BuildCompOperator(cp, tpl.template, nH), tpl, g)
        Cr = BuildCompOperator(cp, tpl.c2, rC); Cn = BuildCompOperator(cp, tpl.c2, nC)
        x = normalize(randn(rng, D))
        hr = *(Hr, x); hn = *(Hn, x); cr = *(Cr, x); cn = *(Cn, x)
        eH = norm(hr - hn) / norm(hr); eC = norm(cr - cn) / max(norm(cr), 1e-300)
        worst["H"] = max(worst["H"], eH); worst["C2"] = max(worst["C2"], eC)
        row = Dict{String,Any}("sa"=>a, "sb"=>b, "L"=>L, "dim"=>D, "rel_H"=>eH, "rel_C2"=>eC)
        if a == b
            perm, sgn = exchange_map(cp)
            Xv(v) = PermFirstSecondSegs(cp, v)
            cH = norm(*(Hn, Xv(x)) - Xv(*(Hn, x))) / norm(hn)
            cC = norm(*(Cn, Xv(x)) - Xv(*(Cn, x))) / max(norm(cn), 1e-300)
            worst["commH"] = max(worst["commH"], cH); worst["commC2"] = max(worst["commC2"], cC)
            ep = ExchangeSector(perm, sgn, 1); em = ExchangeSector(perm, sgn, -1)
            row["dim_even"] = length(ep); row["dim_odd"] = length(em)
            row["comm_H"] = cH; row["comm_C2"] = cC
            length(ep) + length(em) == D || error("exchange sectors do not partition the block")
            # isometry: expand(reduce(P v)) == P v, with P v = (v + sigma X v)/2
            for x_ in (ep, em)
                pv = (x .+ x_.sigma .* Xv(x)) ./ 2
                r = sector_reduce!(zeros(length(x_)), x_, pv); back = sector_expand!(zeros(D), x_, r)
                row["iso_err_$(x_.sigma)"] = norm(back - pv) / max(norm(pv), 1e-300)
                abs(norm(r) - norm(pv)) < 1e-12 * max(1, norm(pv)) || error("reduced map not isometric")
            end
            if D <= 1500
                Ef = eigvals(Symmetric(Matrix(Hn)))
                Es = Float64[]
                for x_ in (ep, em)
                    n = length(x_); n == 0 && continue
                    act = reduced_action(Hn, x_)
                    M = zeros(n, n); e = zeros(n)
                    for k in 1:n; fill!(e, 0); e[k] = 1; M[:, k] = act(e); end
                    append!(Es, eigvals(Symmetric((M + M') / 2)))
                end
                sort!(Es); dsp = maximum(abs.(Es - Ef)); worst["spec"] = max(worst["spec"], dsp)
                row["spec_full_vs_sectors"] = dsp
                # parity of lowest states
                vals, vecs = eigen(Symmetric(Matrix(Hn)))
                row["lowest_parities"] = [round(dot(vecs[:, k], Xv(vecs[:, k])); digits=6) for k in 1:min(6, D)]
            end
        end
        push!(rows, row)
        @printf("%d%d L%d D=%d relH=%.2e relC2=%.2e %s\n", a, b, L, D, eH, eC, a == b ? @sprintf("even/odd=%d/%d commH=%.1e", row["dim_even"], row["dim_odd"], row["comm_H"]) : "")
    end
    a == 0 && b == 2 && nothing
end
summary = Dict("N"=>N, "g"=>g, "tile_width"=>tw, "threads"=>Threads.nthreads(), "worst"=>worst,
               "library_builds"=>lib.builds, "library_hits"=>lib.hits, "reference_channel_builds"=>ref_builds,
               "library_build_seconds"=>lib.build_seconds, "library_resident_bytes"=>resident_bytes(lib), "blocks"=>rows)
open(io -> TOML.print(io, summary; sorted=true), joinpath(out, "summary.toml"), "w")
println("SUMMARY ", worst, " builds=", lib.builds, " hits=", lib.hits, " ref_channels=", ref_builds)
ok = worst["segop"] < 1e-10 && worst["H"] < 1e-12 && worst["C2"] < 1e-12 && worst["commH"] < 1e-12 && worst["commC2"] < 1e-12 && worst["spec"] < 1e-9
println(ok ? "T1_PASS" : "T1_FAIL"); exit(ok ? 0 : 1)
