# v0.2.8, external audit A2: Stage B can be restarted (v0.3.0: src/stageb.jl through bench/stageb_run.jl).
#   1. a finished output is rerun: windows, KK edges, projections and full-Q edges are reused; same export
#   2. a partly lost output (one projection, the KK edges of one L, the export) is completed; same export
#   3. (optional ORIGINAL = the production script 53d551e2) the same rerun of the original stops at
#      "refusing to overwrite" -- the defect this version removes
# usage: julia --project=<env> --threads=T t15_stageb_restart.jl N out [ORIGINAL]
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, TOML, Printf
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); root = abspath(ARGS[2]); original = length(ARGS) >= 3 ? abspath(ARGS[3]) : ""
g = 0.95; mkpath(root)
ok = true
check(c, msg) = (println(c ? "PASS " : "FAIL ", msg); global ok &= c)
th = Dict("native" => min(4, Threads.nthreads()), "tile" => 16)
script = joinpath(@__DIR__, "..", "bench", "stageb_run.jl")

run_dir = joinpath(root, "run_N$N"); o3 = joinpath(root, "ope3_N$N")
conf = Dict("run" => Dict("N" => N, "g" => g, "out" => run_dir),
            "spectrum" => Dict{String,Any}("nev" => 8, "exchange" => true, "tol" => 1e-11),
            "threads" => th, "targets" => Dict("initial_nev" => 12, "max_nev" => 120),
            "kk" => Dict("cache" => joinpath(root, "kk_cache")), "ope" => Dict("half" => false))
run_toml = joinpath(root, "run_N$N.toml"); open(io -> TOML.print(io, conf), run_toml, "w")
run_pipeline(conf)
if !isfile(joinpath(o3, "ope3.toml"))
    c = Tcc.pipeline_config(conf); c["g"] = Tcc.resolved_g(c); sp, sha = Tcc.ope3_spaces(c)
    Tcc.run_ope3(c, sp, sha, o3; opts=Dict{String,Any}("half_lmax" => 3))
end

"""Run a Stage B script in a fresh process; returns (exit code, combined log)."""
function stageb(scr, out)
    logf = joinpath(root, basename(out) * ".log.$(round(Int, time() * 1000))")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads()) $scr $run_toml $out $o3 half`
    cmd = addenv(cmd, "TCC_SOURCE_FILE" => joinpath(@__DIR__, "..", "src", "Tcc.jl"))
    p = open(io -> run(pipeline(ignorestatus(cmd); stdout=io, stderr=io)), logf, "w")
    s = read(logf, String); println("---- $(basename(scr)) $(basename(out)) exit=$(p.exitcode)"); print(last(s, 1500))
    p.exitcode, s
end
cells(l) = [strip(x, '"') for x in split(l, ',')]
"""Exports equal: text columns identical, numbers to 1e-9 (SHA columns of recomputed projections may differ)."""
function same_export(a, b)
    A = readlines(a); B = readlines(b); length(A) == length(B) || return false
    h = cells(A[1]); keep = [i for i in eachindex(h) if !endswith(h[i], "_sha")]
    for (x, y) in zip(A, B)
        p = cells(x); q = cells(y)
        for i in keep
            u = tryparse(Float64, p[i]); v = tryparse(Float64, q[i])
            (u === nothing || v === nothing) ? (p[i] == q[i] || return false) : (abs(u - v) <= 1e-9 * max(1, abs(u)) || return false)
        end
    end
    true
end

outA = joinpath(root, "stageb_N$N"); rm(outA; recursive=true, force=true)
code1, _ = stageb(script, outA); csv = joinpath(outA, "xy_blocks_raw.csv")
check(code1 == 0 && isfile(csv), "Stage B finishes on a fresh output")
ref = joinpath(root, "xy_blocks_raw.first.csv"); cp(csv, ref; force=true)

code2, log2 = stageb(script, outA)
check(code2 == 0 && occursin("STAGEB_KK_EDGE_REUSE", log2) && occursin("STAGEB_PROJECTION_REUSE", log2) &&
      !occursin(r"STAGEB_KK_EDGE N=", log2) && read(csv) == read(ref),
      "rerun of a finished output reuses windows, KK edges and projections; export byte-identical")

rm(joinpath(outA, "kk", "L2.ckpt"); recursive=true)
foreach(f -> rm(joinpath(outA, "kk", "edges", f); recursive=true), filter(startswith("L3_"), readdir(joinpath(outA, "kk", "edges"))))
rm(csv)
code3, log3 = stageb(script, outA)
check(code3 == 0 && occursin("STAGEB_KK_EDGE N=$N sector=half L3->", log3) && occursin("STAGEB_PROJECTION_REUSE N=$N sector=half L3", log3) &&
      isfile(csv) && same_export(csv, ref),
      "rerun after losing projection L2, the L3 KK edges and the export: recomputed, L3 projection re-verified, export equal")

if !isempty(original)
    outB = joinpath(root, "stageb_orig_N$N"); rm(outB; recursive=true, force=true)
    c1, _ = stageb(original, outB); c2, l2 = stageb(original, outB)
    check(c1 == 0 && c2 != 0 && occursin("refusing to overwrite", l2),
          "production script 53d551e2: fresh run finishes, the rerun stops at 'refusing to overwrite' (defect reproduced)")
end

println(ok ? "STAGEB_RESTART_PASS N=$N" : "STAGEB_RESTART_FAIL N=$N")
