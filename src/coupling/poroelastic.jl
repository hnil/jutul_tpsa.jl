# ──────────────────────────────────────────────────────────────────────────
# Thermo-poroelastic coupling helpers
# ──────────────────────────────────────────────────────────────────────────

"""
    extract_darcy_fields(darcy_state) -> (pressure, temperature)

Extract pore-pressure [Pa] and temperature [K] arrays from the state
dictionary returned by a JutulDarcy reservoir simulation.

Returns `(pressure, temperature)` where both are `AbstractVector{Float64}`
of length `nc` (number of cells).  If temperature is not present in the state
(i.e. the Darcy model is isothermal) `temperature` is `nothing`.
"""
function extract_darcy_fields(darcy_state::AbstractDict)
    pressure = darcy_state[:Pressure]
    temperature = get(darcy_state, :Temperature, nothing)
    return (pressure, temperature)
end

"""
    update_mechanical_parameters!(
        mech_parameters, darcy_state;
        p_ref = 0.0, T_ref = 273.15
    )

Copy pressure (and optionally temperature) from a JutulDarcy state dictionary
into the parameter dictionary `mech_parameters` of a mechanical model.

This is the typical coupling step: run a Darcy time-step, then call this
function before running the mechanical solve.

Arguments
---------
- `mech_parameters` — parameter `NamedTuple` / `Dict` from
  `Jutul.setup_parameters(mech_model)`.
- `darcy_state` — state from a JutulDarcy simulation step.
- `p_ref` — reference pressure [Pa] (default 0.0).
- `T_ref` — reference temperature [K] (default 273.15).
"""
function update_mechanical_parameters!(
        mech_parameters,
        darcy_state;
        p_ref = 0.0,
        T_ref = 273.15,
    )
    p, T = extract_darcy_fields(darcy_state)
    mech_parameters[:PorePressure] .= p
    if !isnothing(T)
        mech_parameters[:Temperature] .= T
    end
    return mech_parameters
end
