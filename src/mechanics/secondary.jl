# ──────────────────────────────────────────────────────────────────────────
# Secondary variables: stress and strain
# ──────────────────────────────────────────────────────────────────────────

"""
    ElasticStress <: VectorVariables

Cell-centred symmetric (Voigt) stress tensor as a secondary variable.

Storage convention (Voigt notation):
- 1-D: [σ_xx]
- 2-D: [σ_xx, σ_yy, σ_xy]
- 3-D: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

The values are computed from the cell-centred displacement gradient (finite-
difference approximation using the cell itself and all its face-neighbours).
"""
struct ElasticStress <: VectorVariables end

"""
    ElasticStrain <: VectorVariables

Cell-centred symmetric (Voigt) strain tensor as a secondary variable.
Same Voigt ordering as `ElasticStress`.
"""
struct ElasticStrain <: VectorVariables end

# Number of independent components in dim-D symmetric tensor (Voigt)
_voigt_size(dim) = dim == 1 ? 1 : dim == 2 ? 3 : 6

Jutul.values_per_entity(model, ::ElasticStress) = _voigt_size(model.system.dim)
Jutul.values_per_entity(model, ::ElasticStrain) = _voigt_size(model.system.dim)

function Jutul.select_secondary_variables!(
        S,
        ::LinearElasticitySystem,
        model::SimulationModel,
    )
    S[:ElasticStress] = ElasticStress()
    S[:ElasticStrain] = ElasticStrain()
end

# ──────────────────────────────────────────────────────────────────────────
# Evaluation: ElasticStrain
# ──────────────────────────────────────────────────────────────────────────

function Jutul.update_secondary_variable!(
        ε_out,                    # Voigt × nc
        ::ElasticStrain,
        model::SimulationModel,
        state,
        ix,
    )
    sys = model.system
    dim = sys.dim
    disc = model.domain.discretizations.tpsa

    U = state.Displacement    # dim × nc

    for cell in ix
        ldisc = disc(cell, Cells())
        _update_strain_cell!(ε_out, cell, U, dim, ldisc)
    end
end

function _update_strain_cell!(ε_out, cell, U, dim, ldisc)
    (; faces, face_signs, neighbors, normals, areas, dists) = ldisc
    T = eltype(U)
    u_self = SVector(ntuple(i -> U[i, cell], Val(dim)))

    # Accumulate weighted gradient: ε ≈ sym(∇u) estimated as sum_f t_f ⊗ n_f
    ε_sum = MMatrix{dim, dim, T}(zeros(T, dim, dim))
    vol_approx = 0.0
    for k in eachindex(faces)
        n_k  = normals[k]
        A_k  = areas[k]
        d_k  = dists[k]
        j    = neighbors[k]
        u_j  = SVector(ntuple(i -> U[i, j], Val(dim)))
        Δu   = u_j - u_self
        factor = A_k / d_k
        for α in 1:dim, β in 1:dim
            ε_sum[α, β] += 0.5 * factor * (Δu[α]*n_k[β] + n_k[α]*Δu[β])
        end
        vol_approx += A_k * d_k
    end
    # Normalize by approximate cell volume
    if vol_approx > 0
        ε_sum ./= (vol_approx / dim)
    end

    _write_voigt!(ε_out, cell, ε_sum, dim)
end

# ──────────────────────────────────────────────────────────────────────────
# Evaluation: ElasticStress
# ──────────────────────────────────────────────────────────────────────────

function Jutul.update_secondary_variable!(
        σ_out,
        ::ElasticStress,
        model::SimulationModel,
        state,
        ix,
    )
    sys = model.system
    dim = sys.dim

    ε   = state.ElasticStrain     # Voigt × nc  (already updated)
    E_a = state.YoungModulus
    ν_a = state.PoissonRatio

    for cell in ix
        E_c = E_a[cell]
        ν_c = ν_a[cell]
        λ_c, μ_c = lame_parameters(E_c, ν_c)
        _update_stress_cell!(σ_out, cell, ε, dim, λ_c, μ_c)
    end
end

function _update_stress_cell!(σ_out, cell, ε, dim, λ, μ)
    # Recover full strain tensor from Voigt storage
    ε_full = _read_voigt(ε, cell, dim)
    tr_ε   = sum(ε_full[i,i] for i in 1:dim)

    T = eltype(ε)
    σ_full = MMatrix{dim, dim, T}(zeros(T, dim, dim))
    for α in 1:dim, β in 1:dim
        σ_full[α, β] = 2μ * ε_full[α, β]
    end
    for α in 1:dim
        σ_full[α, α] += λ * tr_ε
    end

    _write_voigt!(σ_out, cell, σ_full, dim)
end

# ──────────────────────────────────────────────────────────────────────────
# Voigt helpers
# ──────────────────────────────────────────────────────────────────────────

# 3-D Voigt index pairs: (1,1),(2,2),(3,3),(2,3),(1,3),(1,2)
const _VOIGT_IDX_3D = ((1,1),(2,2),(3,3),(2,3),(1,3),(1,2))
# 2-D Voigt index pairs: (1,1),(2,2),(1,2)
const _VOIGT_IDX_2D = ((1,1),(2,2),(1,2))

function _write_voigt!(v, cell, m, dim)
    if dim == 1
        v[1, cell] = m[1,1]
    elseif dim == 2
        for (k, (α, β)) in enumerate(_VOIGT_IDX_2D)
            v[k, cell] = m[α, β]
        end
    else
        for (k, (α, β)) in enumerate(_VOIGT_IDX_3D)
            v[k, cell] = m[α, β]
        end
    end
end

function _read_voigt(v, cell, dim)
    T = eltype(v)
    if dim == 1
        return SMatrix{1,1,T,1}(v[1, cell])
    elseif dim == 2
        pairs = _VOIGT_IDX_2D
        m = MMatrix{2,2,T,4}(zeros(T, 4))
        for (k, (α, β)) in enumerate(pairs)
            m[α, β] = v[k, cell]
            m[β, α] = v[k, cell]
        end
        return SMatrix{2,2}(m)
    else
        pairs = _VOIGT_IDX_3D
        m = MMatrix{3,3,T,9}(zeros(T, 9))
        for (k, (α, β)) in enumerate(pairs)
            m[α, β] = v[k, cell]
            m[β, α] = v[k, cell]
        end
        return SMatrix{3,3}(m)
    end
end
