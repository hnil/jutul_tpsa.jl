# ══════════════════════════════════════════════════════════════════════════
# VEM vs TPSA Comparison for 2-D Linear Elasticity
# ══════════════════════════════════════════════════════════════════════════
#
# This example compares the Two-Point Stress Approximation (TPSA) from
# JutulTPSA with a first-order Virtual Element Method (VEM) reference
# implementation on the same Cartesian mesh.
#
# The VEM approach follows the framework used in Jutul by Olav Møyner for
# cell-centred discretisations on polyhedral grids.  On rectangular
# elements the first-order VEM coincides with classical bilinear FEM up
# to a stabilisation term that vanishes for constant-strain modes.
#
# References:
#   Beirão da Veiga, L. et al. (2013). Basic principles of virtual
#       element methods. Math. Models Methods Appl. Sci., 23(01).
#   Gain, A. L. et al. (2014). On the virtual element method for
#       three-dimensional linear elasticity problems on arbitrary
#       polyhedral meshes. Comput. Methods Appl. Mech. Engrg., 282.
#   Møyner, O. & Lie, K.-A. (2016). A multiscale two-point flux-
#       approximation method. J. Comput. Phys., 275.
# ══════════════════════════════════════════════════════════════════════════

using JutulTPSA
using Jutul
using LinearAlgebra
using SparseArrays
using Printf

# ──────────────────────────────────────────────────────────────────────────
# VEM Reference Implementation (first-order, 2-D plane strain, rectangles)
# ──────────────────────────────────────────────────────────────────────────

"""
    plane_strain_D(E, ν) -> Matrix{Float64}

3×3 plane-strain elasticity matrix.
"""
function plane_strain_D(E, ν)
    c = E / ((1 + ν) * (1 - 2ν))
    return c * [1-ν  ν    0
                ν    1-ν  0
                0    0    (1-2ν)/2]
end

"""
    vem_element_stiffness(vertices, D) -> Matrix{Float64}

First-order VEM element stiffness matrix for a convex polygon with
`vertices` (2×nv matrix, counter-clockwise) under plane-strain `D`.

For a first-order VEM on a polygon with nv vertices:
  K_VEM = K_consistency + K_stability
where
  K_consistency = A_e * Π^T D_bar Π
  K_stability   = α * trace(D) * (I - Π_proj)^T (I - Π_proj)

Π is the strain-displacement projection operator computed from the
edge normals (linear completeness), and α is a stabilisation parameter
(typically α ≈ 1).
"""
function vem_element_stiffness(vertices, D; α_stab = 1.0)
    nv = size(vertices, 2)
    ndof = 2 * nv
    # Element centroid and area
    xc = sum(vertices[1, :]) / nv
    yc = sum(vertices[2, :]) / nv
    area = polygon_area(vertices)

    # Build the B-bar matrix (3 × 2*nv) from edge normals
    # For first-order VEM: ε_proj = B_bar * u_h  where
    #   B_bar_α = (1/2A) Σ_e n_e^α |e|  (scaled edge contribution)
    Bbar = zeros(3, ndof)
    for i in 1:nv
        j = mod1(i + 1, nv)
        # Edge vector and outward normal
        ex = vertices[1, j] - vertices[1, i]
        ey = vertices[2, j] - vertices[2, i]
        le = sqrt(ex^2 + ey^2)
        nx =  ey / le   # outward normal (for CCW vertices)
        ny = -ex / le
        # Each edge contributes half to each endpoint
        for (k, w) in [(i, 0.5), (j, 0.5)]
            Bbar[1, 2k-1] += w * nx * le / area   # ε_xx
            Bbar[2, 2k]   += w * ny * le / area   # ε_yy
            Bbar[3, 2k-1] += w * ny * le / area   # γ_xy
            Bbar[3, 2k]   += w * nx * le / area
        end
    end

    # Consistency part: K_c = area * Bbar' * D * Bbar
    Kc = area * Bbar' * D * Bbar

    # Projection operator (maps DOFs to the rigid-body + linear modes)
    # For first-order VEM, the projector Π maps onto the 3-DOF space of
    # constant-strain displacement fields (plus 3 rigid-body modes = 6 total
    # in 2D, but we only need the strain projector for stiffness).
    # Stabilisation: K_s = α * tr(D) * (I - P)^T (I - P) where P projects
    # onto the range of K_c.
    # Simplified stabilisation: dofi-diagonal scaling
    stab_coeff = α_stab * tr(D) * area / ndof
    # Build P = Bbar^+ * Bbar (pseudo-inverse projection)
    # For rectangular elements this is a rank-3 projection
    if rank(Bbar) > 0
        P = Bbar' * pinv(Bbar' * Bbar) * Bbar  # (ndof × ndof) projection
        Ks = stab_coeff * (I(ndof) - P)' * (I(ndof) - P)
    else
        Ks = stab_coeff * I(ndof)
    end

    return Kc + Ks
end

"""
    polygon_area(vertices) -> Float64

Signed area of a polygon (positive for CCW winding).
"""
function polygon_area(v)
    nv = size(v, 2)
    A = 0.0
    for i in 1:nv
        j = mod1(i + 1, nv)
        A += v[1, i] * v[2, j] - v[1, j] * v[2, i]
    end
    return abs(A) / 2
end

"""
    vem_assemble(nx, ny, Lx, Ly, E, ν) -> SparseMatrixCSC

Assemble global VEM stiffness for an nx × ny Cartesian grid.
"""
function vem_assemble(nx, ny, Lx, Ly, E, ν)
    D = plane_strain_D(E, ν)
    dx, dy = Lx / nx, Ly / ny
    nn = (nx + 1) * (ny + 1)
    ndof = 2 * nn
    II = Int[]; JJ = Int[]; VV = Float64[]

    for ey in 1:ny, ex in 1:nx
        # Vertex coordinates (CCW: BL, BR, TR, TL)
        x0 = (ex - 1) * dx; y0 = (ey - 1) * dy
        verts = [x0    x0+dx  x0+dx  x0
                 y0    y0     y0+dy  y0+dy]
        Ke = vem_element_stiffness(verts, D)

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
    vem_body_force_rhs(nx, ny, Lx, Ly, fx, fy) -> Vector{Float64}

Consistent load vector for uniform body force.
"""
function vem_body_force_rhs(nx, ny, Lx, Ly, fx, fy)
    dx, dy = Lx / nx, Ly / ny
    nn = (nx + 1) * (ny + 1)
    f = zeros(2 * nn)
    for ey in 1:ny, ex in 1:nx
        n1 = (ey - 1) * (nx + 1) + ex; n2 = n1 + 1
        n3 = n2 + (nx + 1); n4 = n1 + (nx + 1)
        vol = dx * dy
        for n in [n1, n2, n3, n4]
            f[2n - 1] += fx * vol / 4
            f[2n]     += fy * vol / 4
        end
    end
    return f
end

"""
    solve_with_penalty(K, f, bc_dofs, bc_vals; penalty) -> Vector

Solve Ku=f with Dirichlet BCs via the penalty method.
"""
function solve_with_penalty(K, f, bc_dofs, bc_vals; penalty = 1e20)
    Kp = copy(K); fp = copy(f)
    for (d, v) in zip(bc_dofs, bc_vals)
        Kp[d, d] += penalty; fp[d] += penalty * v
    end
    return Kp \ fp
end

"""
    nodal_to_cell_centers(u, nx, ny) -> Matrix{Float64}

Average nodal displacements to cell centres (2 × ncells).
"""
function nodal_to_cell_centers(u, nx, ny)
    uc = zeros(2, nx * ny)
    for ey in 1:ny, ex in 1:nx
        c = (ey - 1) * nx + ex
        n1 = (ey - 1) * (nx + 1) + ex; n2 = n1 + 1
        n3 = n2 + (nx + 1); n4 = n1 + (nx + 1)
        for n in [n1, n2, n3, n4]
            uc[1, c] += u[2n - 1]; uc[2, c] += u[2n]
        end
        uc[:, c] ./= 4
    end
    return uc
end

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

function boundary_cells_2d(nx, ny)
    cells = Int[]
    for j in 1:ny, i in 1:nx
        if i == 1 || i == nx || j == 1 || j == ny
            push!(cells, (j - 1) * nx + i)
        end
    end
    return cells
end

# ──────────────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────────────

function run_comparison()
    println("=" ^ 72)
    println("  VEM vs TPSA Comparison — 2-D Gravity-Loaded Clamped Square")
    println("=" ^ 72)

    E_val = 1e9    # Young's modulus [Pa]
    ν_val = 0.25   # Poisson's ratio
    ρg    = 1e6    # body force in -y [N/m³]

    for (nx, ny) in [(4, 4), (8, 8), (16, 16)]
        Lx, Ly = 1.0, 1.0
        dx, dy = Lx / nx, Ly / ny
        nc = nx * ny

        # ── VEM solve ───────────────────────────────────────────────────
        K_vem  = vem_assemble(nx, ny, Lx, Ly, E_val, ν_val)
        f_vem  = vem_body_force_rhs(nx, ny, Lx, Ly, 0.0, -ρg)
        bc_d, bc_v = boundary_node_dofs_zero(nx, ny)
        u_vem  = solve_with_penalty(K_vem, f_vem, bc_d, bc_v)
        u_vem_cc = nodal_to_cell_centers(u_vem, nx, ny)

        # ── TPSA solve ──────────────────────────────────────────────────
        g = CartesianMesh((nx, ny), (Lx, Ly))
        domain = DataDomain(g)
        model, state0, param = setup_tpsa_model(domain; dim = 2,
            E = E_val, ν = ν_val)
        bnd = boundary_cells_2d(nx, ny)
        bc  = DisplacementConstraint(bnd, [0.0, 0.0])
        bf  = BodyForce(Int[], [0.0, -ρg])
        forces = setup_forces(model; body_force = bf, dirichlet = bc)
        states, = tpsa_solve(model, state0, param;
            info_level = -1, forces = forces)
        u_tpsa = states[end][:Displacement]

        # ── Comparison at interior cells ────────────────────────────────
        interior = setdiff(1:nc, bnd)
        vem_max  = maximum(abs.(u_vem_cc[2, interior]))
        tpsa_max = maximum(abs.(u_tpsa[2, interior]))
        diff_max = maximum(abs.(u_tpsa[2, interior] .- u_vem_cc[2, interior]))

        println()
        @printf("  Mesh %2d × %2d  (h = %.4f)\n", nx, ny, dx)
        println("  " * "-" ^ 50)
        @printf("    VEM  max |u_y| at interior cells: %12.6e\n", vem_max)
        @printf("    TPSA max |u_y| at interior cells: %12.6e\n", tpsa_max)
        @printf("    Max |u_y(TPSA) - u_y(VEM)|:      %12.6e\n", diff_max)
        if vem_max > 0
            @printf("    Relative difference:              %8.2f %%\n",
                100 * diff_max / vem_max)
        end
    end

    println()
    println("=" ^ 72)
    println("  Note: VEM and TPSA are fundamentally different discretisations.")
    println("  VEM uses nodal DOFs (averaged to cell centres for comparison),")
    println("  while TPSA uses cell-centred DOFs.  Both converge to the same")
    println("  continuous solution as the mesh is refined.")
    println("=" ^ 72)
end

run_comparison()
