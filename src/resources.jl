# Resource policies (v0.3.0).  None of these switches changes a physical number; they choose between
# algorithms that were validated against each other (README v0.1.5, v0.1.17).  Up to v0.2.8 the
# defaults were written as N thresholds; from v0.3.0 they are decided from the size of the problem,
# with thresholds placed so that every production run (N5-N12) gets exactly the choice it had.
#
#   basis.ph_half      = "auto": PH half construction when the largest segment sector that BuildSegSpaces
#                        diagonalises densely has at least `ph_half_dim_min` (2000) configurations.
#                        Largest sector: N9 1070, N10 3656, N11 10762, N12 37834 -> on from N10 (as before).
#   spectrum.residency = "auto": "pair" when the stored segment basis (segment_spaces.ckpt payload) is
#                        larger than `residency_pair_bytes` (16e9).  N11 4.6e9 -> all, N12 4.9e10 -> pair.
#   threads.native_dim_max (2e7, src/matvec.jl): per block, FuzzifiED's own action up to that dimension,
#                        the column-blocked Tcc action above it (unchanged, already dimension based).

const PH_HALF_DIM_MIN = Ref(2000)
const RESIDENCY_PAIR_BYTES = Ref(16e9)

"""
Largest flavour-Sz = 0, minimal-|Lz| sector of one 2-flavour segment (the sectors `segment_setup`
hands to BuildSegSpaces), counted combinatorially: two flavours of N orbitals with 2m = -(N-1), ..., N-1.
"""
function segment_sector_dim_max(N::Int)
    ms = [-(N - 1) + 2i for i in 0:N-1]
    cnt = Dict((0, 0) => 1)                                # (electrons, 2Lz) of one flavour -> configurations
    for m in ms
        nxt = copy(cnt)
        for ((k, lz), c) in cnt; nxt[(k + 1, lz + m)] = get(nxt, (k + 1, lz + m), 0) + c; end
        cnt = nxt
    end
    best = 0
    for ne in 0:2:2N
        k = ne ÷ 2; lz = ((N + 1) * ne) % 2
        d = sum((c1 * get(cnt, (k, lz - l1), 0) for ((k1, l1), c1) in cnt if k1 == k); init=0)
        best = max(best, d)
    end
    best
end

"""PH half construction by default when the dense segment diagonalisation is large (`basis.ph_half = "auto"`)."""
ph_half_default(N::Int) = segment_sector_dim_max(N) >= PH_HALF_DIM_MIN[]
_use_ph(cfg, N) = (v = get(cfg, "ph_half", "auto"); v == "auto" ? ph_half_default(N) : v === true)

"""Segment checkpoint used by a run: the external one, else out/segment_spaces.ckpt (may not exist yet)."""
_segment_ckpt(cfg) = (s = get(cfg, "segment_checkpoint", ""); isempty(s) ? joinpath(cfg["out"], "segment_spaces.ckpt") : s)

"""
Residency for "auto" (or a missing key): "pair" when the stored segment basis exceeds `residency_pair_bytes`,
"all" otherwise or while the basis does not exist yet (the basis stage itself does not use residency).
"""
function residency_auto(cfg)
    p = joinpath(_segment_ckpt(cfg), "payload.jls")
    isfile(p) || return "all"
    filesize(p) > get(cfg, "residency_pair_bytes", RESIDENCY_PAIR_BYTES[]) ? "pair" : "all"
end
resolve_residency(cfg) = (v = get(cfg, "residency", "auto"); v == "auto" ? residency_auto(cfg) : v)
