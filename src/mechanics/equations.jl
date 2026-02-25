# ──────────────────────────────────────────────────────────────────────────
# Mechanical equilibrium equation
# ──────────────────────────────────────────────────────────────────────────

"""
    MechanicalEquilibriumEquation

Residual form of the quasi-static force balance

    ∑_f t_f(u) + f_body = 0

discretised with the Two-Point Stress Approximation (TPSA).  In thermo-
poroelastic mode the total traction on each face also includes the Biot
effective-pressure contribution and the thermal-expansion contribution.
"""
struct MechanicalEquilibriumEquation{D} <: JutulEquation
    discretization::D
end

# Number of equations per cell = spatial dimension
function Jutul.number_of_equations_per_entity(
        model  ::SimulationModel,
        eq     ::MechanicalEquilibriumEquation,
    )
    return model.system.dim
end

Jutul.associated_entity(::MechanicalEquilibriumEquation) = Cells()

function Jutul.local_discretization(eq::MechanicalEquilibriumEquation, i)
    return eq.discretization(i, Cells())
end

function Jutul.select_equations!(
        eqs,
        ::LinearElasticitySystem,
        model::SimulationModel,
    )
    disc = model.domain.discretizations.tpsa
    eqs[:mechanical_equilibrium] = MechanicalEquilibriumEquation(disc)
end

# ──────────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────────

function Jutul.update_equation_in_entity!(
        eq_buf  ::AbstractVector,     # length = dim
        self_cell,
        state,
        state0,
        eq      ::MechanicalEquilibriumEquation,
        model   ::SimulationModel,
        dt,
        ldisc   = Jutul.local_discretization(eq, self_cell),
    )
    sys = model.system
    dim = sys.dim

    # ---- Retrieve cell fields -------------------------------------------
    U     = state.Displacement      # dim × nc matrix (or SMatrix view)
    E_arr = state.YoungModulus
    ν_arr = state.PoissonRatio

    E_i = E_arr[self_cell]
    ν_i = ν_arr[self_cell]
    λ_i, μ_i = lame_parameters(E_i, ν_i)

    u_self = SVector(ntuple(i -> U[i, self_cell], Val(dim)))

    # ---- Optional coupling fields ---------------------------------------
    coupled = sys.coupled
    if coupled
        p_arr    = state.PorePressure
        T_arr    = state.Temperature
        p0_arr   = state.ReferencePressure
        T0_arr   = state.ReferenceTemperature
        α_B_arr  = state.BiotCoefficient
        α_T_arr  = state.ThermalExpansionCoefficient

        Δp_i = p_arr[self_cell] - p0_arr[self_cell]
        ΔT_i = T_arr[self_cell] - T0_arr[self_cell]
    end

    # ---- Accumulate TPSA residual over neighbouring half-faces ---------
    residual = @SVector zeros(dim)

    (; faces, face_signs, neighbors, normals, areas, dists) = ldisc

    for k in eachindex(faces)
        j      = neighbors[k]       # neighbouring cell index
        f      = faces[k]
        sgn    = face_signs[k]
        n_k    = normals[k]         # outward unit normal *from self*
        A_k    = areas[k]
        d_k    = dists[k]

        E_j = E_arr[j]
        ν_j = ν_arr[j]
        λ_j, μ_j = lame_parameters(E_j, ν_j)

        # Harmonic mean of Lamé parameters across the face
        λ_f = 2*λ_i*λ_j / (λ_i + λ_j + eps(typeof(λ_i)))
        μ_f = 2*μ_i*μ_j / (μ_i + μ_j + eps(typeof(μ_i)))

        u_other = SVector(ntuple(i -> U[i, j], Val(dim)))

        t_elastic = tpsa_traction(u_self, u_other, n_k, A_k, d_k, λ_f, μ_f)
        residual = residual + t_elastic

        # Thermo-poroelastic coupling: subtract effective-pressure and
        # thermal-expansion terms (they enter with opposite signs relative
        # to elastic traction so they resist compression when p > 0).
        if coupled
            Δp_j = p_arr[j] - p0_arr[j]
            ΔT_j = T_arr[j] - T0_arr[j]
            α_B_f = (α_B_arr[self_cell] + α_B_arr[j]) / 2
            α_T_f = (α_T_arr[self_cell] + α_T_arr[j]) / 2
            K_f   = bulk_modulus((E_i + E_j)/2, (ν_i + ν_j)/2)

            # Face-averaged pressure and temperature
            Δp_f = (Δp_i + Δp_j) / 2
            ΔT_f = (ΔT_i + ΔT_j + ΔT_i + (T_arr[j] - T0_arr[j])) / 4

            # Biot effective stress correction: -α_B Δp n A
            # Thermal stress correction: +3 K α_T ΔT n A
            coupling_coeff = A_k * (-α_B_f * Δp_f + 3 * K_f * α_T_f * ΔT_f)
            residual = residual + coupling_coeff * n_k
        end
    end

    for i in 1:dim
        eq_buf[i] = residual[i]
    end
end

# ──────────────────────────────────────────────────────────────────────────
# Body-force driving force (optional)
# ──────────────────────────────────────────────────────────────────────────

"""
    BodyForce{T}

A uniform body-force vector applied to one or all cells.

Fields
------
- `cells::Vector{Int}` — cell indices to which the force is applied.
  Pass `Int[]` to apply to all cells.
- `value::SVector{N,T}` — force per unit volume [N/m³].
"""
struct BodyForce{N, T} <: JutulForce
    cells::Vector{Int}
    value::SVector{N, T}
end

function BodyForce(cells, value::AbstractVector{T}) where T
    N = length(value)
    return BodyForce{N, T}(collect(cells), SVector{N, T}(value))
end

export BodyForce

function Jutul.apply_forces_to_equation!(
        d,
        storage,
        model,
        eq    ::MechanicalEquilibriumEquation,
        eq_s,
        force ::BodyForce{N, T},
        time,
    ) where {N, T}
    dim    = model.system.dim
    domain = model.domain
    nc     = number_of_cells(domain)
    vols   = physical_representation(domain)

    target_cells = isempty(force.cells) ? (1:nc) : force.cells

    # Body force residual: -f_body * V_cell added to residual
    for c in target_cells
        vol = cell_volume(vols, c)
        for i in 1:dim
            d[i, c] -= force.value[i] * vol
        end
    end
end

"""
    cell_volume(mesh, cell_index)

Return the volume (or area in 2-D) of cell `cell_index`.
Falls back to 1 if the mesh does not provide volumes.
"""
function cell_volume(mesh, c)
    try
        return Jutul.cell_dims(mesh, c) |> prod
    catch
        return 1.0
    end
end

# ──────────────────────────────────────────────────────────────────────────
# Dirichlet boundary condition (penalty method)
# ──────────────────────────────────────────────────────────────────────────

"""
    DisplacementConstraint{N, T}

Dirichlet displacement boundary condition applied via the penalty method.

For each constrained cell `c` and each constrained component `α`, the equation
residual gains the extra term:

    R_c^α += penalty * (U_c^α - u_prescribed^α)

Setting `penalty ≫ 1` forces `U_c ≈ u_prescribed`.

Fields
------
- `cells`      — cell indices to constrain (default: first cell only).
- `value`      — prescribed displacement vector `SVector{N, T}`.
- `penalty`    — penalty stiffness (default `1e12`).
- `components` — which displacement components to constrain (default: all).
  Pass `[1]` to constrain only the x-component.
"""
struct DisplacementConstraint{N, T} <: JutulForce
    cells      ::Vector{Int}
    value      ::SVector{N, T}
    penalty    ::Float64
    components ::Vector{Int}
end

function DisplacementConstraint(
        cells,
        value   ::AbstractVector{T};
        penalty = 1e12,
        components = nothing,
    ) where T
    N = length(value)
    comp = isnothing(components) ? collect(1:N) : collect(components)
    return DisplacementConstraint{N, T}(
        collect(Int, cells),
        SVector{N, T}(value),
        Float64(penalty),
        comp,
    )
end

export DisplacementConstraint

function Jutul.apply_forces_to_equation!(
        d,
        storage,
        model,
        eq    ::MechanicalEquilibriumEquation,
        eq_s,
        force ::DisplacementConstraint{N, T},
        time,
    ) where {N, T}
    U = storage.state.Displacement
    for c in force.cells
        for α in force.components
            d[α, c] += force.penalty * (U[α, c] - force.value[α])
        end
    end
end
