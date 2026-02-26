# ──────────────────────────────────────────────────────────────────────────
# FEM Reference Test for TPSA Linear Elasticity
# ──────────────────────────────────────────────────────────────────────────
#
# Validates the TPSA implementation against a standard Q4 finite element
# reference solver on identical Cartesian meshes.  The self-contained FEM
# implementation below uses bilinear (Q4) quadrilateral elements with 2×2
# Gauss quadrature under plane-strain conditions.
#
# For production FEM work, consider using Ferrite.jl or similar packages.
# ──────────────────────────────────────────────────────────────────────────

using JutulTPSA
using Jutul
using Test
using LinearAlgebra
using SparseArrays

# ══════════════════════════════════════════════════════════════════════════
# Q4 FEM Helper Functions
# ══════════════════════════════════════════════════════════════════════════

"""
    q4_stiffness(dx, dy, E, ν) -> Matrix{Float64}

8×8 element stiffness matrix for a rectangular Q4 plane-strain element
of dimensions `dx × dy` using 2×2 Gauss quadrature.
"""
function q4_stiffness(dx, dy, E, ν)
    c = E / ((1 + ν) * (1 - 2ν))
    D = c * [1-ν  ν    0
             ν    1-ν  0
             0    0    (1-2ν)/2]
    gp = 1 / sqrt(3)
    Ke = zeros(8, 8)
    for (ξ, η) in [(-gp,-gp), (gp,-gp), (gp,gp), (-gp,gp)]
        dNdξ = [-(1-η), (1-η), (1+η), -(1+η)] ./ 4
        dNdη = [-(1-ξ), -(1+ξ), (1+ξ), (1-ξ)] ./ 4
        dNdx = dNdξ .* (2 / dx)
        dNdy = dNdη .* (2 / dy)
        B = zeros(3, 8)
        for i in 1:4
            B[1, 2i-1] = dNdx[i]
            B[2, 2i]   = dNdy[i]
            B[3, 2i-1] = dNdy[i]
            B[3, 2i]   = dNdx[i]
        end
        Ke .+= B' * D * B * (dx * dy / 4)
    end
    return Ke
end

"""
    fem_global_stiffness(nx, ny, dx, dy, E, ν) -> SparseMatrixCSC

Assemble the global stiffness matrix for an `nx × ny` grid of Q4 elements.
Nodes are numbered row-major from bottom-left: node (ix,iy) → (iy-1)*(nx+1)+ix.
"""
function fem_global_stiffness(nx, ny, dx, dy, E, ν)
    Ke = q4_stiffness(dx, dy, E, ν)
    nn = (nx + 1) * (ny + 1)
    ndof = 2 * nn
    II = Int[]; JJ = Int[]; VV = Float64[]
    for ey in 1:ny, ex in 1:nx
        n1 = (ey - 1) * (nx + 1) + ex
        n2 = n1 + 1
        n3 = n2 + (nx + 1)
        n4 = n1 + (nx + 1)
        dofs = [2n1-1, 2n1, 2n2-1, 2n2, 2n3-1, 2n3, 2n4-1, 2n4]
        for i in 1:8, j in 1:8
            push!(II, dofs[i]); push!(JJ, dofs[j]); push!(VV, Ke[i, j])
        end
    end
    return sparse(II, JJ, VV, ndof, ndof)
end

"""
    fem_body_force_rhs(nx, ny, dx, dy, fx, fy) -> Vector{Float64}

Consistent nodal load vector for uniform body force `(fx, fy)` [N/m²].
Each node of a Q4 element receives `f_α * V_element / 4`.
"""
function fem_body_force_rhs(nx, ny, dx, dy, fx, fy)
    nn = (nx + 1) * (ny + 1)
    f = zeros(2 * nn)
    for ey in 1:ny, ex in 1:nx
        n1 = (ey - 1) * (nx + 1) + ex
        n2 = n1 + 1
        n3 = n2 + (nx + 1)
        n4 = n1 + (nx + 1)
        vol = dx * dy
        for n in [n1, n2, n3, n4]
            f[2n - 1] += fx * vol / 4
            f[2n]     += fy * vol / 4
        end
    end
    return f
end

"""
    fem_solve_penalty(K, f, bc_dofs, bc_vals; penalty=1e20) -> Vector{Float64}

Solve `Ku = f` with Dirichlet BCs via the penalty method.
"""
function fem_solve_penalty(K, f, bc_dofs, bc_vals; penalty = 1e20)
    Kp = copy(K); fp = copy(f)
    for (d, v) in zip(bc_dofs, bc_vals)
        Kp[d, d] += penalty
        fp[d]    += penalty * v
    end
    return Kp \ fp
end

"""
    fem_cell_center_values(u_nodal, nx, ny) -> Matrix{Float64}

Average nodal displacements to cell centres (2 × ncells).
"""
function fem_cell_center_values(u_nodal, nx, ny)
    uc = zeros(2, nx * ny)
    for ey in 1:ny, ex in 1:nx
        c = (ey - 1) * nx + ex
        n1 = (ey - 1) * (nx + 1) + ex
        n2 = n1 + 1
        n3 = n2 + (nx + 1)
        n4 = n1 + (nx + 1)
        for n in [n1, n2, n3, n4]
            uc[1, c] += u_nodal[2n - 1]
            uc[2, c] += u_nodal[2n]
        end
        uc[:, c] ./= 4
    end
    return uc
end

"""
    boundary_cells_2d(nx, ny) -> Vector{Int}

Indices of cells on the boundary of an `nx × ny` Cartesian grid.
Cell (i,j) has index `(j-1)*nx + i`.
"""
function boundary_cells_2d(nx, ny)
    cells = Int[]
    for j in 1:ny, i in 1:nx
        if i == 1 || i == nx || j == 1 || j == ny
            push!(cells, (j - 1) * nx + i)
        end
    end
    return cells
end

"""
    boundary_node_dofs_zero(nx, ny) -> (dofs, vals)

DOF indices and zero values for clamping all boundary nodes of an
`(nx+1) × (ny+1)` node grid.
"""
function boundary_node_dofs_zero(nx, ny)
    dofs = Int[]; vals = Float64[]
    for iy in 1:(ny + 1), ix in 1:(nx + 1)
        if ix == 1 || ix == nx + 1 || iy == 1 || iy == ny + 1
            n = (iy - 1) * (nx + 1) + ix
            push!(dofs, 2n - 1); push!(vals, 0.0)
            push!(dofs, 2n);     push!(vals, 0.0)
        end
    end
    return dofs, vals
end

# ══════════════════════════════════════════════════════════════════════════
# 1-D FEM Helper Functions (P1 bar elements)
# ══════════════════════════════════════════════════════════════════════════

"""
    fem_1d_solve(n, L, E; f_body=0.0, u_left=0.0, fix_right=false, u_right=0.0)

Solve a 1-D bar problem with `n` elements on `[0, L]`.
Returns nodal displacements (length `n+1`).
"""
function fem_1d_solve(n, L, E; f_body = 0.0, u_left = 0.0,
                      fix_right = false, u_right = 0.0)
    h = L / n
    # Assemble tridiagonal stiffness
    nn = n + 1
    II = Int[]; JJ = Int[]; VV = Float64[]
    f = zeros(nn)
    for e in 1:n
        ke = E / h
        n1, n2 = e, e + 1
        for (i, j, v) in [(n1,n1,ke),(n1,n2,-ke),(n2,n1,-ke),(n2,n2,ke)]
            push!(II, i); push!(JJ, j); push!(VV, v)
        end
        f[n1] += f_body * h / 2
        f[n2] += f_body * h / 2
    end
    K = sparse(II, JJ, VV, nn, nn)
    # BCs via penalty
    pen = 1e20
    K[1, 1] += pen; f[1] += pen * u_left
    if fix_right
        K[nn, nn] += pen; f[nn] += pen * u_right
    end
    return K \ f
end

# ══════════════════════════════════════════════════════════════════════════
# Test 11. FEM & TPSA vs analytical (1-D clamped bar under gravity)
# ══════════════════════════════════════════════════════════════════════════

@testset "FEM reference: 1D bar analytical comparison" begin
    # Problem: bar on [0,L], E constant, ν=0, body force f in +x.
    # Fixed at x=0 (u=0), free at x=L.
    # Analytical: u(x) = f/(2E) * x * (2L - x)
    #   (from E u'' + f = 0, u(0)=0, u'(L)=0)
    n   = 20
    L   = 1.0
    E_v = 1e6
    f_b = 1e4   # body force per unit length

    # ── Analytical solution at cell centres ──
    h = L / n
    x_cc = [(i - 0.5) * h for i in 1:n]
    u_analytical = [f_b / (2E_v) * x * (2L - x) for x in x_cc]

    # ── FEM solution (P1 bar elements) ──
    u_fem_nodes = fem_1d_solve(n, L, E_v; f_body = f_b)
    # Interpolate to cell centres
    u_fem_cc = [(u_fem_nodes[i] + u_fem_nodes[i+1]) / 2 for i in 1:n]

    # ── TPSA solution ──
    domain = DataDomain(CartesianMesh((n,), (L,)))
    model, state0, param = setup_tpsa_model(domain; dim = 1, E = E_v, ν = 0.0)
    # Fix left cell (closest to x=0)
    bc = DisplacementConstraint([1], [0.0])
    bf = BodyForce(Int[], [f_b])
    forces = setup_forces(model; body_force = bf, dirichlet = bc)
    states, = tpsa_solve(model, state0, param;
        info_level = -1, forces = forces)
    u_tpsa = vec(states[end][:Displacement])

    # ── Comparisons ──
    # FEM should be very close to analytical (second-order accurate)
    fem_err = maximum(abs.(u_fem_cc .- u_analytical))
    @test fem_err / maximum(u_analytical) < 0.01  # < 1% error

    # TPSA should also converge to the analytical solution
    # Cell 1 is pinned, so skip it; compare remaining cells
    tpsa_err = maximum(abs.(u_tpsa[2:end] .- u_analytical[2:end]))
    @test tpsa_err / maximum(u_analytical) < 0.1  # < 10% error

    # Both methods should give similar results at interior cells
    diff_fem_tpsa = maximum(abs.(u_tpsa[2:end] .- u_fem_cc[2:end]))
    @test diff_fem_tpsa / maximum(u_analytical) < 0.15
end

# ══════════════════════════════════════════════════════════════════════════
# Test 12. FEM & TPSA comparison (2-D gravity-loaded clamped square)
# ══════════════════════════════════════════════════════════════════════════

@testset "FEM reference: 2D gravity-loaded clamped square" begin
    E_val = 1e9
    ν_val = 0.25
    ρg    = 1e6   # body force density [N/m³] in -y direction
    nx, ny = 8, 8
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny

    # ── Q4 FEM reference ────────────────────────────────────────────────
    K = fem_global_stiffness(nx, ny, dx, dy, E_val, ν_val)
    f_rhs = fem_body_force_rhs(nx, ny, dx, dy, 0.0, -ρg)
    bc_dofs, bc_vals = boundary_node_dofs_zero(nx, ny)
    u_fem_nodal = fem_solve_penalty(K, f_rhs, bc_dofs, bc_vals)
    u_fem = fem_cell_center_values(u_fem_nodal, nx, ny)

    # ── TPSA solution ───────────────────────────────────────────────────
    g = CartesianMesh((nx, ny), (Lx, Ly))
    domain = DataDomain(g)
    nc = number_of_cells(domain)
    model, state0, param = setup_tpsa_model(domain; dim = 2, E = E_val, ν = ν_val)
    bnd = boundary_cells_2d(nx, ny)
    bc  = DisplacementConstraint(bnd, [0.0, 0.0])
    bf  = BodyForce(Int[], [0.0, -ρg])
    forces = setup_forces(model; body_force = bf, dirichlet = bc)
    states, = tpsa_solve(model, state0, param;
        info_level = -1, forces = forces)
    u_tpsa = states[end][:Displacement]

    # ── Comparison ──────────────────────────────────────────────────────
    interior = setdiff(1:nc, bnd)

    # Both should produce downward displacement under gravity
    @test all(u_fem[2, interior] .< 0)
    @test all(u_tpsa[2, interior] .< 0)

    # Displacement magnitudes should be physically consistent
    fem_max  = maximum(abs.(u_fem[2, interior]))
    tpsa_max = maximum(abs.(u_tpsa[2, interior]))
    @test fem_max > 0
    @test tpsa_max > 0

    # Same order of magnitude (different discretisation schemes)
    @test 0.2 < tpsa_max / fem_max < 5.0

    # Cell-by-cell comparison at interior cells
    max_diff = maximum(abs.(u_tpsa[2, interior] .- u_fem[2, interior]))
    @test max_diff / fem_max < 2.0  # within a factor of 2
end
