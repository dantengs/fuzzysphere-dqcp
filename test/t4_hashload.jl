# T4: single-pass hashed load equals plain SHA + deserialize.
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, SHA, Serialization
d = ARGS[1]
t = time(); a = Tcc.load_segment_spaces(d); t1 = time() - t
println("single_pass sha=", a.sha, " mode=", a.sha_mode, " seconds=", round(t1; digits=1))
want = strip(read(joinpath(d, "sha256.txt"), String))
println(a.sha == want ? "T4_PASS" : "T4_FAIL")
