# ──────────────────────────────────────────────────────────────────────────
# Primary variables
# ──────────────────────────────────────────────────────────────────────────

"""
    Displacement <: VectorVariables

Cell-centred displacement vector.  The number of components equals the
spatial dimension stored in the system (`model.system.dim`).
"""
struct Displacement <: VectorVariables end

Jutul.values_per_entity(model, ::Displacement) = model.system.dim

function Jutul.select_primary_variables!(
        S,
        ::LinearElasticitySystem,
        model::SimulationModel,
    )
    S[:Displacement] = Displacement()
end

# ──────────────────────────────────────────────────────────────────────────
# Material parameters (cell-wise)
# ──────────────────────────────────────────────────────────────────────────

"""
    YoungModulus <: ScalarVariable

Young's modulus E [Pa] per cell.
"""
struct YoungModulus <: ScalarVariable end

"""
    PoissonRatio <: ScalarVariable

Poisson's ratio ν (dimensionless) per cell.
"""
struct PoissonRatio <: ScalarVariable end

"""
    BiotCoefficient <: ScalarVariable

Biot effective-stress coefficient α ∈ [0, 1] per cell.
Only used when `system.coupled == true`.
"""
struct BiotCoefficient <: ScalarVariable end

"""
    ThermalExpansionCoefficient <: ScalarVariable

Linear thermal-expansion coefficient α_T [1/K] per cell.
Only used when `system.coupled == true`.
"""
struct ThermalExpansionCoefficient <: ScalarVariable end

# ──────────────────────────────────────────────────────────────────────────
# Coupling parameters (pore-pressure and temperature as external inputs)
# ──────────────────────────────────────────────────────────────────────────

"""
    PorePressure <: ScalarVariable

Pore pressure [Pa] per cell.  Used as a parameter when running in coupled
mode (the values come from a JutulDarcy reservoir simulation).
"""
struct PorePressure <: ScalarVariable end

"""
    Temperature <: ScalarVariable

Temperature [K] per cell.  Used as a parameter when running in coupled mode.
"""
struct Temperature <: ScalarVariable end

"""
    ReferencePressure <: ScalarVariable

Reference pore pressure [Pa] subtracted from `PorePressure` before computing
the effective-stress coupling term.  Default 0 Pa.
"""
struct ReferencePressure <: ScalarVariable end

"""
    ReferenceTemperature <: ScalarVariable

Reference temperature [K] subtracted from `Temperature` before computing the
thermal-stress coupling term.  Default 273.15 K.
"""
struct ReferenceTemperature <: ScalarVariable end

# Default values for parameters
Jutul.default_value(model, ::YoungModulus) = 1e9          # 1 GPa
Jutul.default_value(model, ::PoissonRatio) = 0.25
Jutul.default_value(model, ::BiotCoefficient) = 1.0
Jutul.default_value(model, ::ThermalExpansionCoefficient) = 1e-5  # 1/K
Jutul.default_value(model, ::PorePressure) = 0.0
Jutul.default_value(model, ::Temperature) = 273.15
Jutul.default_value(model, ::ReferencePressure) = 0.0
Jutul.default_value(model, ::ReferenceTemperature) = 273.15

# Limits
Jutul.minimum_value(::YoungModulus) = 0.0
Jutul.minimum_value(::PoissonRatio) = -1.0 + 1e-8
Jutul.maximum_value(::PoissonRatio) = 0.5 - 1e-8
Jutul.minimum_value(::BiotCoefficient) = 0.0
Jutul.maximum_value(::BiotCoefficient) = 1.0

# Register parameters for all models using LinearElasticitySystem
function Jutul.select_parameters!(
        S,
        sys::LinearElasticitySystem,
        model::SimulationModel,
    )
    S[:YoungModulus] = YoungModulus()
    S[:PoissonRatio] = PoissonRatio()
    if sys.coupled
        S[:BiotCoefficient] = BiotCoefficient()
        S[:ThermalExpansionCoefficient] = ThermalExpansionCoefficient()
        S[:PorePressure] = PorePressure()
        S[:Temperature] = Temperature()
        S[:ReferencePressure] = ReferencePressure()
        S[:ReferenceTemperature] = ReferenceTemperature()
    end
end
