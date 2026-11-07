module KKGenerator

using FuzzifiED
using FuzzifiED.SO3lver
using LinearAlgebra

export generator_candidates, generator_operator, generator_integral_candidates

"""Two-term local candidates for the SO(5) DQCP conformal generator in the 2+2 segment basis.

Each candidate is the l=1 spherical component of a four-flavour local
observable, n†n or Δ†Δ. The four flavours are grouped into the two
symplectic pairs (1,2) and (3,4), one pair per segment.
The contact couplings integrate √(4π)Y* on a sphere of radius²=N;
division by N√(4π) restores the component convention of `GetComponent`.
The same construction with rank 0 permits an independent integral gate.
"""
function generator_candidates(N::Int; rank::Int=1)
    N >= 4 || throw(ArgumentError("N must be at least four"))
    rank >= 0 || throw(ArgumentError("negative operator rank"))
    FuzzifiED.ElementType = Float64
    FuzzifiED.ObsNormRadSq = Float64(N)
    omega = [0 1; 0 0]
    density = StoreComps(GetDensityObs(N, 2; norm_r2=Float64(N)))
    pairing = StoreComps(GetPairingObs(N, 2, omega;
                                          norm_r2=Float64(N), mom_incr=false))
    # The pinned GetPairingObs initializes positive monopole charge, so it
    # requires mom_incr=false, whose component m *lowers* orbital Lz by m.
    # SO3lver requires components labelled by their *increase* in Lz. Reverse
    # m and restore the i^(2m)=(-1)^m phase of the pinned electron angular
    # mode. The phase is invisible in the final m=0 Fock operator but is
    # essential when SO3lver extracts matrix elements from m≠0 components.
    # For two identical rank-J factors this reverses a rank-L component by
    # (-1)^(2J-L)=(-1)^L (J=N-1 is integer); restore the old m=0 component
    # with that rank parity after coupling.
    pair_up = StoreComps(SphereObs(pairing.s2,pairing.l2m,
                                   (l2,m2)->(isodd(m2÷2) ? -1 : 1) *
                                             pairing.get_comp(l2,-m2)))
    neutral = [0, 0, 0]
    d_local = FuzzifiED.SO3lver.AngModes(StoreComps(density' * density))
    p_local = FuzzifiED.SO3lver.AngModes(StoreComps(pair_up' * pair_up))
    # ContactCouple represents the integrated harmonic insertion. Both cross
    # orders are explicit for pairing, where the operators exchange charge.
    cross_scale = inv(N * sqrt(4π))
    d = SingleSegCouple(2, 1, d_local, 2rank, neutral) +
        SingleSegCouple(2, 2, d_local, 2rank, neutral) +
        2cross_scale * ContactCouple([density, density], [0 0; 0 0; 0 0], 2rank)
    p = SingleSegCouple(2, 1, p_local, 2rank, neutral) +
        SingleSegCouple(2, 2, p_local, 2rank, neutral) +
        cross_scale * ContactCouple([pair_up', pair_up], [2 -2; 0 0; 0 0], 2rank) +
        cross_scale * ContactCouple([pair_up, pair_up'], [-2 2; 0 0; 0 0], 2rank)
    isodd(rank) && (p = -p)
    (density_squared=d, symplectic_pair_squared=p)
end

function generator_integral_candidates(N::Int)
    candidates = generator_candidates(N; rank=0)
    # A spherical rank-zero component times N√(4π) is its spatial integral.
    (density_squared=N * sqrt(4π) * candidates.density_squared,
     symplectic_pair_squared=N * sqrt(4π) * candidates.symplectic_pair_squared)
end

"""Build the rank-one, source→target matrix-free SO3lver action."""
function generator_operator(source, target, N::Int, coefficients::AbstractVector)
    length(coefficients) == 2 && all(isfinite, coefficients) ||
        error("Need two finite generator coefficients")
    source.ltot == 0 && target.ltot == 2 ||
        error("This KK adapter handles only L=0→L=1; other components need explicit coverage")
    c = generator_candidates(N)
    terms = coefficients[1] * c.density_squared +
            coefficients[2] * c.symplectic_pair_squared
    BuildCompOperator(source, target, terms, 2)
end

end
