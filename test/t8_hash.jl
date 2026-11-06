# Canonical hash fixture (v0.1.8): Julia digests are independent of CRLF/LF checkouts and of the
# platform path separator, binary files are hashed raw; writes <dir>/fixture_digest.txt for the
# Python twin (tools/canonical_hash.py tree <dir>/lf must print the same digest).
# usage: julia t8_hash.jl DIR
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc
root = ARGS[1]; rm(root; recursive=true, force=true)
files = Dict("a.txt" => "alpha\nbeta\n", "sub/ü_name.jl" => "x = 1\n# comment\n", "sub/deeper/z.toml" => "k = \"v\"\n",
             "B.md" => "upper-case sorts first\n", ".git/HEAD" => "ignored\n")
bin = UInt8[0x00, 0x0d, 0x0a, 0x41, 0x0d, 0x0a]
for (variant, eol) in (("lf", "\n"), ("crlf", "\r\n"))
    for (rel, txt) in files
        p = joinpath(root, variant, split(rel, '/')...); mkpath(dirname(p)); write(p, replace(txt, "\n" => eol))
    end
    write(joinpath(root, variant, "blob.bin"), bin)
end
d1 = tree_digest(joinpath(root, "lf")); d2 = tree_digest(joinpath(root, "crlf"))
f1 = file_digest(joinpath(root, "lf", "a.txt")); f2 = file_digest(joinpath(root, "crlf", "a.txt"))
fb = file_digest(joinpath(root, "lf", "blob.bin"))
ok = d1 == d2 && f1 == f2 && fb == bytes2hex(Tcc.sha256(bin)) && f1 == bytes2hex(Tcc.sha256("alpha\nbeta\n"))
write(joinpath(root, "fixture_digest.txt"), d1 * "\n")
println("tree lf   ", d1); println("tree crlf ", d2); println("git_commit ", git_commit(@__DIR__))
println(ok ? "HASH_FIXTURE_PASS" : "HASH_FIXTURE_FAIL")
exit(ok ? 0 : 1)
