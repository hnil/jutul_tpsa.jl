# ──────────────────────────────────────────────────────────────────────────
# High-level convenience interface
# ──────────────────────────────────────────────────────────────────────────

"""
    setup_tpsa_model(
        domain::DataDomain;
        dim        = 3,
        coupled    = false,
        E          = 1e9,
        ν          = 0.25,
        biot       = 1.0,
        alpha_T    = 1e-5,
        p_ref      = 0.0,
        T_ref      = 273.15,
    ) -> (model, state0, parameters)

Build a `SimulationModel` for quasi-static linear elasticity (or thermo-
poroelasticity when `coupled = true`) on `domain`.

The `domain` must be a `DataDomain` wrapping a `JutulMesh` that provides
`:cell_centroids`, `:face_centroids`, `:normals`, `:areas`, and the standard
connectivity arrays.

Arguments
---------
- `E`       — Young's modulus [Pa].  Scalar or length-`nc` vector.
- `ν`       — Poisson's ratio.  Scalar or length-`nc` vector.
- `biot`    — Biot coefficient α.  Scalar or length-`nc` vector.
- `alpha_T` — Linear thermal-expansion coefficient [1/K].  Scalar or
              length-`nc` vector.
- `p_ref`   — Reference pore pressure [Pa] (scalar or vector).
- `T_ref`   — Reference temperature [K] (scalar or vector).

Returns
-------
- `model`      — configured `SimulationModel`.
- `state0`     — zero-displacement initial state.
- `parameters` — parameter dictionary (modify `:PorePressure` /
                 `:Temperature` before each mechanical solve).
"""
function setup_tpsa_model(
        domain  ::DataDomain;
        dim     ::Int   = 3,
        coupled ::Bool  = false,
        E               = 1e9,
        ν               = 0.25,
        biot            = 1.0,
        alpha_T         = 1e-5,
        p_ref           = 0.0,
        T_ref           = 273.15,
    )
    sys   = LinearElasticitySystem(dim = dim, coupled = coupled)
    model = SimulationModel(domain, sys, output_level = :all)

    nc = number_of_cells(model.domain)
    _expand(v, n) = v isa AbstractVector ? v : fill(v, n)

    state0 = setup_state(model, Displacement = zeros(dim, nc))

    param_args = Dict{Symbol, Any}(
        :YoungModulus => _expand(E,    nc),
        :PoissonRatio => _expand(ν,    nc),
    )
    if coupled
        param_args[:BiotCoefficient]            = _expand(biot,    nc)
        param_args[:ThermalExpansionCoefficient] = _expand(alpha_T, nc)
        param_args[:PorePressure]               = _expand(0.0,     nc)
        param_args[:Temperature]                = _expand(T_ref,   nc)
        param_args[:ReferencePressure]          = _expand(p_ref,   nc)
        param_args[:ReferenceTemperature]       = _expand(T_ref,   nc)
    end
    parameters = setup_parameters(model; param_args...)

    return (model, state0, parameters)
end

"""
    tpsa_solve(
        model, state0, parameters;
        dt          = [1.0],
        forces      = nothing,
        info_level  = 0,
        kwarg...,
    ) -> (states, reports)

Run a quasi-static mechanical solve (a single "time-step" is sufficient for
the elliptic linear-elasticity system).

Returns the vector of states (one entry per time-step in `dt`) and the
simulation reports.

Notes
-----
- For a one-shot solve (no transient effects) pass `dt = [1.0]`.
- To include body forces pass a `BodyForce` (or vector thereof) as `forces`.
"""
function tpsa_solve(
        model,
        state0,
        parameters;
        dt          = [1.0],
        forces      = nothing,
        info_level  = 0,
        kwarg...,
    )
    if forces isa NamedTuple
        f = forces
    else
        f = setup_forces(model; body_force = forces)
    end
    case = JutulCase(model, dt, f; state0 = state0, parameters = parameters)
    return simulate(case; info_level = info_level, kwarg...)
end

# ── Forces setup for mechanical model ──────────────────────────────────────

"""
    setup_forces(model::SimulationModel{<:Any,<:LinearElasticitySystem}, …)

Extend Jutul's `setup_forces` for the mechanical model.  Accepts an optional
`body_force` keyword argument which may be a single `BodyForce` or a vector of
them.
"""
function Jutul.setup_forces(
        model   ::SimulationModel{<:Any, <:LinearElasticitySystem};
        body_force = nothing,
        dirichlet  = nothing,
    )
    return (body_force = body_force, dirichlet = dirichlet)
end

function Jutul.apply_forces_to_equation!(
        d,
        storage,
        model   ::SimulationModel{<:Any, <:LinearElasticitySystem},
        eq      ::MechanicalEquilibriumEquation,
        eq_s,
        force   ::Nothing,
        time,
    )
    return  # nothing to do
end

# ── Coupling convenience ────────────────────────────────────────────────────

export extract_darcy_fields, update_mechanical_parameters!
