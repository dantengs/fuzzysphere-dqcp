# Canonical content digests (identical on Linux, macOS and Windows checkouts).
#   * file: SHA-256 of the bytes with CRLF -> LF unless the file contains a NUL byte (binary);
#   * tree: files under `dir` (excluding .git), POSIX relative paths sorted bytewise, each
#     contributing "<path>\0<file digest>\n" to one SHA-256.
# A Python twin (tools/canonical_hash.py) must produce the same strings.

function _lf(b::Vector{UInt8})
    0x00 in b && return b
    out = Vector{UInt8}(undef, length(b)); n = 0
    @inbounds for i in eachindex(b)
        (b[i] == 0x0d && i < length(b) && b[i+1] == 0x0a) && continue
        n += 1; out[n] = b[i]
    end
    resize!(out, n)
end

"""Canonical SHA-256 (hex) of one file."""
file_digest(path::AbstractString) = bytes2hex(sha256(_lf(read(path))))

"""Canonical SHA-256 (hex) of a directory tree."""
function tree_digest(dir::AbstractString)
    rels = String[]
    for (root, dirs, files) in walkdir(dir)
        filter!(d -> d != ".git", dirs)
        for f in files
            push!(rels, join(splitpath(relpath(joinpath(root, f), dir)), '/'))
        end
    end
    sort!(rels; by=codeunits)
    ctx = SHA.SHA256_CTX()
    for r in rels
        SHA.update!(ctx, Vector{UInt8}(codeunits(r))); SHA.update!(ctx, UInt8[0x00])
        SHA.update!(ctx, Vector{UInt8}(codeunits(file_digest(joinpath(dir, r))))); SHA.update!(ctx, UInt8[0x0a])
    end
    bytes2hex(SHA.digest!(ctx))
end

"""Commit of the git checkout containing `dir` (reads .git directly; "" when unavailable)."""
function git_commit(dir::AbstractString=@__DIR__)
    d = abspath(dir)
    while true
        g = joinpath(d, ".git")
        if isdir(g) || isfile(g)
            if isfile(g)                                   # worktree: "gitdir: <path>"
                m = match(r"gitdir:\s*(.+)", read(g, String)); m === nothing && return ""
                g = isabspath(strip(m[1])) ? strip(m[1]) : joinpath(d, strip(m[1]))
            end
            isfile(joinpath(g, "HEAD")) || return ""        # v0.2.8: worktree registered under another mount
            head = strip(read(joinpath(g, "HEAD"), String))
            startswith(head, "ref:") || return head
            ref = strip(head[5:end])
            for base in (g, joinpath(g, "..", ".."))       # refs may live in the common dir of a worktree
                f = joinpath(base, ref); isfile(f) && return strip(read(f, String))
            end
            for base in (g,)
                pf = joinpath(base, "packed-refs")
                isfile(pf) || continue
                for l in eachline(pf); endswith(l, " " * ref) && return split(l)[1]; end
            end
            return ""
        end
        p = dirname(d); p == d && return ""; d = p
    end
end
