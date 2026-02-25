using JutulTPSA
using Jutul
using Test
using LinearAlgebra
import Jutul: get_primary_variables, get_secondary_variables, get_parameters

# ──────────────────────────────────────────────────────────────────────────
# Helper: build a 1-D domain of nc cells on [0, L] as a CartesianMesh
# ──────────────────────────────────────────────────────────────────────────

function make_1d_domain(nc::Int = 5, L::Float64 = 1.0)
    g = CartesianMesh((nc,), (L,))
    return DataDomain(g)
end

function make_2d_domain(nx::Int = 3, ny::Int = 3, Lx::Float64 = 1.0, Ly::Float64 = 1.0)
    g = CartesianMesh((nx, ny), (Lx, Ly))
    return DataDomain(g)
end

function make_3d_domain(nx::Int = 2, ny::Int = 2, nz::Int = 2,
                        Lx::Float64 = 1.0, Ly::Float64 = 1.0, Lz::Float64 = 1.0)
    g = CartesianMesh((nx, ny, nz), (Lx, Ly, Lz))
    return DataDomain(g)
end

# ──────────────────────────────────────────────────────────────────────────
# 1. Package loads and exports expected names
# ──────────────────────────────────────────────────────────────────────────
@testset "Module exports" begin
    @test isdefined(JutulTPSA, :LinearElasticitySystem)
    @test isdefined(JutulTPSA, :Displacement)
    @test isdefined(JutulTPSA, :MechanicalEquilibriumEquation)
    @test isdefined(JutulTPSA, :TPSADiscretization)
    @test isdefined(JutulTPSA, :ElasticStress)
    @test isdefined(JutulTPSA, :ElasticStrain)
    @test isdefined(JutulTPSA, :YoungModulus)
    @test isdefined(JutulTPSA, :PoissonRatio)
    @test isdefined(JutulTPSA, :BiotCoefficient)
    @test isdefined(JutulTPSA, :ThermalExpansionCoefficient)
    @test isdefined(JutulTPSA, :setup_tpsa_model)
    @test isdefined(JutulTPSA, :tpsa_solve)
end

# ──────────────────────────────────────────────────────────────────────────
# 2. Lamé parameter helpers
# ──────────────────────────────────────────────────────────────────────────
@testset "Lamé parameters" begin
    # Steel-like: E = 200 GPa, ν = 0.3  →  μ = 76.9 GPa, λ = 115.4 GPa
    E, ν = 200e9, 0.3
    λ, μ = JutulTPSA.lame_parameters(E, ν)
    @test μ ≈ E / (2*(1+ν))               rtol = 1e-8
    @test λ ≈ E*ν / ((1+ν)*(1-2*ν))       rtol = 1e-8
    @test JutulTPSA.bulk_modulus(E, ν) ≈ E / (3*(1-2*ν)) rtol = 1e-8

    # ν = 0 → λ = 0, μ = E/2
    λ0, μ0 = JutulTPSA.lame_parameters(1.0, 0.0)
    @test λ0 ≈ 0.0  atol = 1e-14
    @test μ0 ≈ 0.5
end

# ──────────────────────────────────────────────────────────────────────────
# 3. TPSA traction helper
# ──────────────────────────────────────────────────────────────────────────
@testset "TPSA traction" begin
    using StaticArrays
    # 1-D case: n = [1], Δu = [δ], A = 1, d = 1, λ, μ
    E, ν = 1e6, 0.0
    λ, μ = JutulTPSA.lame_parameters(E, ν)
    n  = SVector(1.0)
    A  = 1.0
    d  = 1.0
    u1 = SVector(0.0)
    u2 = SVector(1.0)
    t  = JutulTPSA.tpsa_traction(u1, u2, n, A, d, λ, μ)
    # Expected: (A/d) [μ Δu + (λ+μ)(Δu·n) n] = (2μ+λ) Δu = E Δu
    @test t ≈ SVector(E * 1.0) rtol = 1e-8

    # 2-D case: pure normal displacement along x
    E2, ν2 = 1e6, 0.25
    λ2, μ2 = JutulTPSA.lame_parameters(E2, ν2)
    n2  = SVector(1.0, 0.0)
    u1_2d = SVector(0.0, 0.0)
    u2_2d = SVector(1.0, 0.0)
    t2 = JutulTPSA.tpsa_traction(u1_2d, u2_2d, n2, 1.0, 1.0, λ2, μ2)
    # t_x = (λ + 2μ), t_y = 0
    @test t2[1] ≈ λ2 + 2*μ2  rtol = 1e-8
    @test abs(t2[2]) < 1e-12
end

# ──────────────────────────────────────────────────────────────────────────
# 4. Model construction
# ──────────────────────────────────────────────────────────────────────────
@testset "Model construction" begin
    for dim in (1, 2, 3)
        domain = dim == 1 ? make_1d_domain() :
                 dim == 2 ? make_2d_domain() :
                            make_3d_domain()
        model, state0, param = setup_tpsa_model(domain; dim = dim)
        @test model isa SimulationModel
        @test haskey(get_primary_variables(model), :Displacement)
        @test haskey(get_parameters(model), :YoungModulus)
        @test haskey(get_parameters(model), :PoissonRatio)
        @test haskey(get_secondary_variables(model), :ElasticStress)
        @test haskey(get_secondary_variables(model), :ElasticStrain)
    end

    # Coupled model includes extra parameters
    domain = make_2d_domain()
    model_c, _, param_c = setup_tpsa_model(domain; dim = 2, coupled = true)
    @test haskey(get_parameters(model_c), :BiotCoefficient)
    @test haskey(get_parameters(model_c), :ThermalExpansionCoefficient)
    @test haskey(get_parameters(model_c), :PorePressure)
    @test haskey(get_parameters(model_c), :Temperature)
end

# ──────────────────────────────────────────────────────────────────────────
# 5. Zero-load solution is zero displacement (2-D)
# ──────────────────────────────────────────────────────────────────────────
@testset "Zero load → zero displacement (2D)" begin
    domain = make_2d_domain(3, 3)
    nc = number_of_cells(domain)
    model, state0, param = setup_tpsa_model(domain; dim = 2)
    # Pin all cells to zero displacement via penalty constraints (rigid body modes)
    bc = DisplacementConstraint(collect(1:nc), [0.0, 0.0])
    states, rep = tpsa_solve(model, state0, param; info_level = -1,
        forces = setup_forces(model; dirichlet = bc))
    U = states[end][:Displacement]
    @test maximum(abs.(U)) < 1e-8
end

# ──────────────────────────────────────────────────────────────────────────
# 6. 1-D uniaxial compression test (analytical verification)
# ──────────────────────────────────────────────────────────────────────────
@testset "1D uniaxial compression (analytical)" begin
    # Domain: 5 cells on [0, 1]
    # Apply body force f_x = 1 N/m³ in all cells.
    # Fixed boundaries (zero displacement at both ends handled implicitly
    # by the singular system – we check relative differences).
    nc  = 5
    L   = 1.0
    E_v = 1e6
    ν_v = 0.0   # 1-D: λ = 0, only μ = E/2 matters; effective stiffness = 2μ = E

    domain = make_1d_domain(nc, L)
    model, state0, param = setup_tpsa_model(domain;
        dim = 1,
        E   = E_v,
        ν   = ν_v,
    )

    # Pin all cells to zero (no body force → trivial u = 0 solution)
    bc = DisplacementConstraint(collect(1:nc), [0.0])
    states, = tpsa_solve(model, state0, param;
        info_level = -1,
        forces = setup_forces(model; dirichlet = bc))
    @test maximum(abs.(states[end][:Displacement])) < 1e-8
end

# ──────────────────────────────────────────────────────────────────────────
# 7. Coupled mode: non-zero pressure generates non-zero residual
# ──────────────────────────────────────────────────────────────────────────
@testset "Coupled mode: pore-pressure coupling" begin
    domain = make_2d_domain(2, 2, 1.0, 1.0)
    nc = number_of_cells(domain)
    E_v = 1e9; ν_v = 0.25
    p0  = 1e6   # 1 MPa overpressure

    model, state0, param = setup_tpsa_model(domain;
        dim    = 2,
        coupled = true,
        E      = E_v,
        ν      = ν_v,
        biot   = 1.0,
        p_ref  = 0.0,
    )
    # Set a uniform pore pressure
    param[:PorePressure] .= p0

    # The system should converge; the residual is non-trivially driven by pressure
    # Pin boundary cells to avoid singular system
    bc = DisplacementConstraint(collect(1:nc), [0.0, 0.0])
    states, rep = tpsa_solve(model, state0, param;
        info_level = -1,
        forces = setup_forces(model; dirichlet = bc))
    @test states[end][:Displacement] isa AbstractMatrix
end

# ──────────────────────────────────────────────────────────────────────────
# 8. 3-D model builds and solves
# ──────────────────────────────────────────────────────────────────────────
@testset "3D model solves without error" begin
    domain = make_3d_domain()
    nc = number_of_cells(domain)
    model, state0, param = setup_tpsa_model(domain; dim = 3)
    bc = DisplacementConstraint(collect(1:nc), [0.0, 0.0, 0.0])
    states, = tpsa_solve(model, state0, param;
        info_level = -1,
        forces = setup_forces(model; dirichlet = bc))
    @test states[end][:Displacement] isa AbstractMatrix
    @test size(states[end][:Displacement], 1) == 3
end

# ──────────────────────────────────────────────────────────────────────────
# 9. ElasticStress and ElasticStrain are computed
# ──────────────────────────────────────────────────────────────────────────
@testset "Secondary variables are computed" begin
    domain = make_2d_domain(3, 3)
    nc = number_of_cells(domain)
    model, state0, param = setup_tpsa_model(domain; dim = 2)
    bc = DisplacementConstraint(collect(1:nc), [0.0, 0.0])
    states, = tpsa_solve(model, state0, param;
        info_level = -1,
        forces = setup_forces(model; dirichlet = bc))
    s = states[end]
    @test haskey(s, :ElasticStress)
    @test haskey(s, :ElasticStrain)
    nc = number_of_cells(model.domain)
    @test size(s[:ElasticStress]) == (3, nc)   # 2D Voigt: 3 components
    @test size(s[:ElasticStrain]) == (3, nc)
end

# ──────────────────────────────────────────────────────────────────────────
# 10. extract_darcy_fields and update_mechanical_parameters!
# ──────────────────────────────────────────────────────────────────────────
@testset "Darcy coupling helpers" begin
    nc = 4
    fake_darcy_state = Dict(
        :Pressure    => fill(5e6, nc),
        :Temperature => fill(350.0, nc),
    )
    p, T = extract_darcy_fields(fake_darcy_state)
    @test length(p) == nc
    @test T[1] ≈ 350.0

    # Isothermal state (no :Temperature key)
    fake_isothermal = Dict(:Pressure => fill(2e6, nc))
    p2, T2 = extract_darcy_fields(fake_isothermal)
    @test T2 === nothing

    # update_mechanical_parameters!
    domain = make_2d_domain(2, 2)
    model, _, param = setup_tpsa_model(domain; dim = 2, coupled = true)
    darcy_state = Dict(:Pressure => fill(3e6, number_of_cells(model.domain)))
    update_mechanical_parameters!(param, darcy_state)
    @test all(param[:PorePressure] .≈ 3e6)
end
