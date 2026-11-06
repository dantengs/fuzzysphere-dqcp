# Physical model for the four-flavour SO(5) DQCP on the fuzzy sphere, general N.
# Expressions are byte-for-byte the frozen Zhou v0.0.5 formulas (run_so5_resume.jl,
# so5_g_scan.jl, n11 physical_model.jl); only the bookkeeping is generalised.
#   H(g) = H_den - g H_pair ;  C2 uses the parity-corrected cross-density coefficient.

"""Segment-space construction parameters used by every Zhou-lineage run."""
function segment_setup(nm::Int)
    ne = 2nm
    qnd_pt = [GetNeQNDiag(nm * 2), GetLz2QNDiag(nm, 2), GetFlavQNDiag(nm, 2, [1, -1])]
    tms_lzlp_pt = GetLzLpTerms(nm, 2)
    tms_c2 = GetC2Terms(nm, 2, :SU)
    sec_pt = stack([[ne1, ((nm + 1) * ne1) % 2, 0] for ne1 = 0:ne])
    s_rng = collect(0:2)
    c2_rng = [Float64[sx * (sx + 1)] for sx in s_rng]
    (; qnd_pt, tms_lzlp_pt, tms_c2, sec_pt, s_rng, c2_rng, sec_tot=[ne, 0, 0])
end

"""Build the three integer-spin segment spaces (s=0,1,2) exactly as the frozen runner.
`ph = true` (v0.1.17): diagonalise only ne <= N and generate ne > N by particle-hole (src/ph.jl);
same multiplet subspaces, different multiplet gauge."""
function build_segment_spaces(nm::Int; ph::Bool=false)
    p = segment_setup(nm)
    ph && return ph_expand(ph_build_half(nm, p), p)
    BuildSegSpaces(nm * 2, p.sec_pt, p.qnd_pt, p.tms_lzlp_pt, p.tms_c2, p.c2_rng)
end

"""Return (nn, dd, c2) CoupleDecomps; H(g) = nn - g*dd."""
function build_terms(nm::Int)
    nf = 4
    s = (nm - 1) / 2
    pair = GetPairingMod(nm, 2, [0 1; 0 0])
    den = GetDensityMod(nm, 2, [1 0; 0 1])
    nn = SingleSegCouple(2, 1, GetDenIntTerms(nm, 2), [0, 0, 0]) +
         SingleSegCouple(2, 2, GetDenIntTerms(nm, 2), [0, 0, 0]) +
         2 * CoupleDecomps([den, den], ConvPsPot(RecoupleAngMom(s, [1]))..., [0 0; 0 0; 0 0])
    dd = SingleSegCouple(2, 1, GetPairIntTerms(nm, 2, [0 1; 0 0]), [0, 0, 0]) +
         SingleSegCouple(2, 2, GetPairIntTerms(nm, 2, [0 1; 0 0]), [0, 0, 0]) +
         CoupleDecomps([pair', pair], ConvPsPot(s, [1])..., [2 -2; 0 0; 0 0]) +
         CoupleDecomps([pair, pair'], ConvPsPot(s, [1])..., [-2 2; 0 0; 0 0])
    c2sg = GetC2Terms(nm, 2, :SU) + (nf - 2) / 4 * GetPolTerms(nm, 2)
    c2 = SingleSegCouple(2, 1, c2sg, [0, 0, 0]) +
         SingleSegCouple(2, 2, c2sg, [0, 0, 0]) +
         CoupleDecomps([den, den], ConvPsPot(Dict([2s - l => SO5C2DensityPhase.cross_density_coefficient(nm) for l = 0:2s]))..., [0 0; 0 0; 0 0]) +
         CoupleDecomps([pair', pair], ConvPsPot(Dict([2s - l => -1 for l = 0:2:2s]))..., [2 -2; 0 0; 0 0]) +
         CoupleDecomps([pair, pair'], ConvPsPot(Dict([2s - l => -1 for l = 0:2:2s]))..., [-2 2; 0 0; 0 0])
    return nn, dd, c2
end

"""g-independent Hamiltonian template nn+dd plus the real coefficient vectors."""
function hamiltonian_template(nm::Int)
    nn, dd, c2 = build_terms(nm)
    template = nn + dd
    re(cpd) = (z = [x.coeff for x in cpd];
               maximum(abs, imag.(z); init=0.0) < 1e-11 || error("complex channel coefficient");
               real.(z))
    (; template, c2, n_density=length(nn), density=re(nn), pairing=re(dd))
end

"""Set H(g)=H_den - g H_pair on a CompOperator built from `hamiltonian_template`."""
function set_g!(H, tpl, g::Real)
    H.coeff[1:tpl.n_density] .= tpl.density
    H.coeff[tpl.n_density+1:end] .= -g .* tpl.pairing
    H
end

const SHA_RECEIPTS = Ref(get(ENV, "TCC_SHA_RECEIPTS", ""))

_statsig(path) = (s = stat(path); "$(s.size):$(s.inode):$(s.mtime):$(s.ctime)")

"""SHA-256 of a large immutable file; reuse a persistent receipt when the stat signature is unchanged."""
function file_sha(path::AbstractString)
    p = abspath(path); rec = SHA_RECEIPTS[]
    if !isempty(rec) && isfile(rec)
        for line in eachline(rec)
            f = split(line, '\t')
            length(f) == 3 && f[1] == p && f[2] == _statsig(p) && return (String(f[3]), :receipt)
        end
    end
    sig = _statsig(p)
    h = open(io -> bytes2hex(sha256(io)), p)
    _statsig(p) == sig || error("file changed while hashing: $p")
    if !isempty(rec)
        mkpath(dirname(rec)); open(io -> println(io, p, '\t', sig, '\t', h), rec, "a")
    end
    (h, :streamed)
end

# Single-pass verified deserialisation: the SHA-256 is accumulated while the
# deserializer consumes the stream, so a large NFS payload is read once, not twice.
mutable struct HashingReader <: IO
    io::IO
    ctx::SHA.SHA256_CTX
    buf::Vector{UInt8}
    n::Int
end
HashingReader(io) = HashingReader(io, SHA.SHA256_CTX(), Vector{UInt8}(undef, 1 << 16), 0)
function _flush!(h::HashingReader)
    h.n > 0 && SHA.update!(h.ctx, view(h.buf, 1:h.n)); h.n = 0; nothing
end
Base.eof(h::HashingReader) = eof(h.io)
function Base.read(h::HashingReader, ::Type{UInt8})
    b = read(h.io, UInt8)
    h.n == length(h.buf) && _flush!(h)
    h.n += 1; @inbounds h.buf[h.n] = b
    b
end
function Base.unsafe_read(h::HashingReader, p::Ptr{UInt8}, nb::UInt)
    unsafe_read(h.io, p, nb)
    _flush!(h); SHA.update!(h.ctx, unsafe_wrap(Array, p, Int(nb)))
    nothing
end
Base.readbytes!(h::HashingReader, b::AbstractVector{UInt8}, nb=length(b)) =
    (GC.@preserve b unsafe_read(h, pointer(b), UInt(nb)); nb)
function digest!(h::HashingReader)
    while !eof(h.io); read(h, UInt8); end
    _flush!(h); bytes2hex(SHA.digest!(h.ctx))
end

"""Load the `spaces` of a frozen `segment_spaces.ckpt` (SHA verified in the same read, receipt-cached)."""
function load_segment_spaces(ckpt_dir::AbstractString)
    data = abspath(joinpath(ckpt_dir, "payload.jls"))
    want = strip(read(joinpath(ckpt_dir, "sha256.txt"), String))
    rec = SHA_RECEIPTS[]
    sig = _statsig(data)
    if !isempty(rec) && isfile(rec) && any(l -> l == string(data, '\t', sig, '\t', want), eachline(rec))
        p = open(deserialize, data)
        return (spaces=p.spaces, meta=p.meta, sha=want, sha_mode=:receipt)
    end
    p, got = open(data) do io
        h = HashingReader(io); x = deserialize(h); (x, digest!(h))
    end
    _statsig(data) == sig || error("file changed while reading: $data")
    got == want || error("segment checkpoint SHA mismatch: $ckpt_dir")
    if !isempty(rec)
        mkpath(dirname(rec)); open(io -> println(io, data, '\t', sig, '\t', got), rec, "a")
    end
    (spaces=p.spaces, meta=p.meta, sha=got, sha_mode=:single_pass)
end
