# Build the Julia environment of Tcc (run once, from anywhere):
#
#   julia deps/setup_env.jl
#
# 1. installs the registered releases FuzzifiED 2.0.1 and KrylovKit 0.10.4 into env/;
# 2. copies FuzzifiED into deps/FuzzifiED.jl and replaces src/so3lver/core/comp_operator.jl by the production
#    version (deps/FuzzifiED-2.0.1/comp_operator.jl: multi-threaded assembly with a `num_th` keyword and a
#    lock-free H*v; Tcc calls BuildCompOperator(...; num_th=...), which the registered release does not accept);
#    env/ then uses this copy (Pkg.develop);
# 3. checks every source file of both packages against the SHA-256 of the copies used in production
#    (deps/production_sources.toml).  KrylovKit: the production copy additionally carries `LockedLanczos`
#    (src/eigsolve/lockedlanczos.jl, changes in src/KrylovKit.jl and src/algorithms.jl), an addition that Tcc
#    never calls; the registered release is used and these files are reported as the expected difference.
# Afterwards:  julia --project=env --threads=8 bin/tcc.jl run.toml
using Pkg, SHA, TOML
const ROOT = dirname(@__DIR__); const ENVDIR = joinpath(ROOT, "env"); const FZ = joinpath(@__DIR__, "FuzzifiED.jl")
const PATCH = joinpath(@__DIR__, "FuzzifiED-2.0.1", "comp_operator.jl"); const REL = "src/so3lver/core/comp_operator.jl"
const UUIDS = Dict("FuzzifiED" => "b45b6d2d-92f2-400f-a7df-d2d897927e5f", "KrylovKit" => "0b1a1467-8014-51b9-945f-bf0ae24f4b77")
VERSION.major == 1 && VERSION.minor == 11 || @warn "Tcc was validated with Julia 1.11.7; this is $(VERSION)"

filesha(p) = (b = read(p); 0x00 in b || (b = Vector{UInt8}(replace(String(b), "\r\n" => "\n"))); bytes2hex(sha256(b)))
pkgroot(name) = dirname(dirname(Base.locate_package(Base.PkgId(Base.UUID(UUIDS[name]), name))))

Pkg.activate(ENVDIR)
haskey(Pkg.project().dependencies, "FuzzifiED") && Pkg.rm("FuzzifiED")       # a re-run starts from the release again
Pkg.add([PackageSpec(name="FuzzifiED", version="2.0.1"), PackageSpec(name="KrylovKit", version="0.10.4")])
reg = pkgroot("FuzzifiED")                                                     # the registered 2.0.1 tree (read-only)
rm(FZ; recursive=true, force=true); cp(reg, FZ); chmod(FZ, 0o755; recursive=true)
cp(PATCH, joinpath(FZ, REL); force=true)
Pkg.rm("FuzzifiED"); Pkg.develop(path=FZ)
Pkg.compat("FuzzifiED", "=2.0.1")                                             # Pkg.rm dropped it; keeps env/Project.toml as tracked
Pkg.instantiate()

prod = TOML.parsefile(joinpath(@__DIR__, "production_sources.toml")); ok = true
for (pkg, dir) in (("FuzzifiED.jl", FZ), ("KrylovKit.jl", pkgroot("KrylovKit")))
    want = prod[pkg]["production_src"]; expected = pkg == "KrylovKit.jl" ? Set(vcat(prod[pkg]["registered_differs"], prod[pkg]["production_only"])) : Set{String}()
    bad = String[]; extra = String[]
    for (f, h) in want
        p = joinpath(dir, f)
        (isfile(p) && filesha(p) == h) || (f in expected ? push!(extra, f) : push!(bad, f))
    end
    println(pkg, ": ", length(want) - length(bad) - length(extra), "/", length(want), " source files identical to production",
            isempty(extra) ? "" : "; expected difference (production-only LockedLanczos, unused by Tcc): " * join(sort(extra), ", "))
    isempty(bad) || (println("  NOT AS IN PRODUCTION: ", join(sort(bad), ", ")); global ok = false)
end
ok || error("environment differs from the production sources")
Pkg.precompile()
println("TCC_ENV_READY  from ", ROOT, ":  julia --project=env --threads=8 bin/tcc.jl <run.toml>")
