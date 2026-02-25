"""
    JutulTPSA

Two-Point Stress Approximation (TPSA) extension for Jutul and JutulDarcy.

This package implements the Two-Point Stress Approximation for linear
elasticity and thermo-poroelastic coupling. It provides:

- `LinearElasticitySystem`: a Jutul system for linear elasticity
- `MechanicalEquilibriumEquation`: the force-balance equation using TPSA
- `Displacement`: primary vector variable (3D displacement field)
- `ElasticStress`: secondary variable (symmetric stress tensor)
- `ElasticStrain`: secondary variable (symmetric strain tensor)
- Parameters: `YoungModulus`, `PoissonRatio`, `BiotCoefficient`,
  `ThermalExpansionCoefficient`, `ReferencePressure`, `ReferenceTemperature`
- `setup_tpsa_model`: convenience function to build a mechanical model
- `tpsa_solve`: run a mechanical solve with pressure/temperature coupling

The TPSA transmissibility for a face between cells i and j is:

    T_f^{αβ} = (A_f / h_f) * [μ δ_{αβ} + (λ + μ) n_f^α n_f^β]

where A_f is the face area, h_f the distance between cell centroids projected
onto the face normal, λ and μ are the Lamé parameters, and n_f is the outward
unit normal of the face.

References:
- Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization
  for Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942–968.
- Keilegavlen, E. & Nordbotten, J. M. (2017). Finite volume methods for
  elasticity with weak symmetry. Int. J. Numer. Meth. Engrg., 112, 939–962.
"""
module JutulTPSA

using Jutul
import Jutul: JutulDiscretization, JutulEquation, JutulForce, JutulSystem
using JutulDarcy
using LinearAlgebra
using StaticArrays

include("mechanics/types.jl")
include("mechanics/variables.jl")
include("mechanics/discretization.jl")
include("mechanics/equations.jl")
include("mechanics/secondary.jl")
include("coupling/poroelastic.jl")
include("setup.jl")

export LinearElasticitySystem
export Displacement
export ElasticStress, ElasticStrain
export YoungModulus, PoissonRatio
export BiotCoefficient, ThermalExpansionCoefficient
export ReferencePressure, ReferenceTemperature
export PorePressure, Temperature
export MechanicalEquilibriumEquation
export TPSADiscretization
export BodyForce, DisplacementConstraint
export setup_tpsa_model
export tpsa_solve

end # module JutulTPSA
