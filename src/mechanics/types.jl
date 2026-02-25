"""
    LinearElasticitySystem

Jutul system type for quasi-static linear elasticity (optionally with
thermo-poroelastic coupling).

Fields
------
- `dim::Int`   — spatial dimension (1, 2 or 3).  Default = 3.
- `coupled::Bool` — if `true`, expects `:PorePressure` and `:Temperature`
  parameters and includes Biot/thermal body-stress coupling. Default = `false`.
"""
Base.@kwdef struct LinearElasticitySystem <: JutulSystem
    dim::Int = 3
    coupled::Bool = false
end
