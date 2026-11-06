# Physical parity P on composite states (v0.2.2 research, 2026-10-13).
#
# Zhou-Hu-Zhu-He (arXiv:2306.16435) use the particle-hole transformation c -> J c^dagger (with i -> -i)
# as the improper O(3) element (phi is P-odd).  In the 2+2 segment form J pairs the two flavours of a
# segment, so P is a product of one particle-hole map per segment.  src/ph.jl already implements that
# map on one segment,
#     P c†_{m,f} P⁻¹ = (-1)^(s-m) ε_f c_{-m,f̄},   P|0> = |full>,
# which commutes with Lz, L±, the segment SU(2) Casimir and the segment Hamiltonian.  Hence P sends the
# multiplet |ne, l, α> of a segment space to a combination of the multiplets |2N-ne, l, β>; the
# orthogonal matrices U^(l) are computed here from the stored representative components (sts).
# On a composite channel (A sector, B sector, l_A, l_B; L) the coefficient block X[β_B, α_A] maps to
# U_B X U_A^T in the channel (P(A), P(B), l_A, l_B; L).  P is real on the real Zhou Hamiltonian, so its
# antiunitary part acts trivially on the stored real eigenvectors: <v|P v> = ±1 is the parity.
# Convention (fixed empirically, then verified):  the segment map above sends m -> -m, i.e. it is the
# paper's P composed with a rotation, and its sign on the vacuum depends on N (N5: -1, N10: +1).  The
# physical parity of a state of total angular momentum L is therefore
#     P_phys = <v|P_seg|v> * <vac|P_seg|vac> * (-1)^L .
# Checks (cluster, 2026-10-13): no sign is needed between composite sectors (every stored eigenstate of
# all 24 blocks at N5 and 72 blocks at N10 is a pure P_seg eigenstate, residual <= 3.9e-10; the
# alternative sector signs fail with residual ~1); multiplets seen in several (sA,sB) blocks get the same
# parity (86 at N5, all of N10); N5: phi -, T14 +, S +, S^- (5.353) -, as required by the phi x phi
# selection rule; N10: all 129 P/D rows of Table VI of arXiv:2306.16435 (L = 0..3, ten SO(5) irreps)
# reproduced, 129/129.

"""Segment PH maps: `maps[i] = (j, Dict(2l => U))` for every sector i of segment space `sp` (N orbitals)."""
function segment_ph_maps(sp, nm::Int; tol=1e-8)
    p = segment_setup(nm); qnd = p.qnd_pt
    lz2 = qnd[2].charge; fl = qnd[3].charge
    partner = ph_partner(lz2, fl); eta = ph_eta(nm, lz2, fl)
    no = 2nm; nsec = size(sp.sec, 2); maps = Dict{Int,Any}(); worst = 0.0
    loc(i) = sp.ptr_st[i] .- (sp.ptr_st[i][1] - 1)
    for i in 1:nsec
        ne = sp.sec[1, i]
        j = findfirst(k -> sp.sec[1, k] == no - ne && sp.sec[2:end, k] == sp.sec[2:end, i], 1:nsec)
        j === nothing && error("segment PH: no partner sector for ne = $ne")
        Pm = ph_matrix(sp.bs[i], sp.bs[j], partner, eta)
        li = loc(i); lj = loc(j); U = Dict{Int,Matrix{Float64}}()
        for (il, l2) in enumerate(sp.l_rng[i])
            haskey(sp.l_lookup[j], l2) || error("segment PH: l = $(l2/2) missing in the partner sector")
            jl = sp.l_lookup[j][l2]
            ci = li[il]:li[il+1]-1; cj = lj[jl]:lj[jl+1]-1
            length(ci) == length(cj) || error("segment PH: multiplicity differs for ne = $ne, 2l = $l2")
            img = Pm * sp.sts[i][:, ci]
            u = Matrix{Float64}(real.(sp.sts[j][:, cj]' * img))
            worst = max(worst, norm(u' * u - I), norm(img - sp.sts[j][:, cj] * u))
            U[l2] = u
        end
        maps[i] = (j, U)
    end
    worst <= tol || error("segment PH maps not orthogonal / not closed: $worst")
    maps, worst
end

"""Index structure of P on a composite space with two segment spaces (maps from segment_ph_maps)."""
function comp_parity(cpsp, mapsA, mapsB; sign_rule::Symbol=:none)
    cpsp.np == 2 || error("comp_parity: two parts only")
    nsec = size(cpsp.idsec, 2)
    secidx = Dict(Tuple(cpsp.idsec[:, k]) => k for k in 1:nsec)
    plan = Any[]
    for isec in 1:nsec
        iA, iB = cpsp.idsec[1, isec], cpsp.idsec[2, isec]
        jA, UA = mapsA[iA]; jB, UB = mapsB[iB]
        jsec = get(secidx, (jA, jB), 0); jsec == 0 && error("comp_parity: image sector missing")
        nA = cpsp.sgsp[1].sec[1, iA]
        sg = sign_rule === :none ? 1.0 : sign_rule === :halfA ? (iseven(nA ÷ 2) ? 1.0 : -1.0) :
             sign_rule === :halfAN ? (iseven((nA - cpsp.sgsp[1].sec[1, jA]) ÷ 4) ? 1.0 : -1.0) : error("sign_rule")
        for ich in eachindex(cpsp.chs[isec])
            ch = cpsp.chs[isec][ich]
            jch = findfirst(==(ch), cpsp.chs[jsec]); jch === nothing && error("comp_parity: image channel missing")
            ri = cpsp.ptr_st[isec][ich]:cpsp.ptr_st[isec][ich+1]-1
            rj = cpsp.ptr_st[jsec][jch]:cpsp.ptr_st[jsec][jch+1]-1
            ua = UA[ch[1, 1]]; ub = UB[ch[1, 2]]
            length(ri) == size(ua, 2) * size(ub, 2) || error("comp_parity: channel dimension mismatch")
            push!(plan, (ri=ri, rj=rj, ua=ua, ub=ub, sg=sg))
        end
    end
    plan
end

"""w = P v (columns of a matrix at once)."""
function apply_parity(plan, V::AbstractMatrix{Float64})
    W = zeros(size(V)); m = size(V, 2)
    Threads.@threads for k in eachindex(plan)
        q = plan[k]; dA = size(q.ua, 2); dB = size(q.ub, 2)
        for c in 1:m
            X = reshape(view(V, q.ri, c), dB, dA)
            W[q.rj, c] .= vec(q.sg .* (q.ub * X * transpose(q.ua)))
        end
    end
    W
end

"""Parity eigenvalues <v|P v> and purity residuals ||P v - p v|| of the columns of V."""
function parity_values(plan, V::AbstractMatrix{Float64})
    W = apply_parity(plan, V)
    p = [dot(view(V, :, c), view(W, :, c)) for c in axes(V, 2)]
    r = [norm(view(W, :, c) .- p[c] .* view(V, :, c)) for c in axes(V, 2)]
    p, r
end

"""Physical parity from the raw segment-map value: P_seg * P_seg(vacuum) * (-1)^L (see header)."""
physical_parity(praw, vac_raw, L) = round(Int, praw) * round(Int, vac_raw) * (isodd(L) ? -1 : 1)
