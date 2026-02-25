# ══════════════════════════════════════════════════════════════════════════
# Thermal Stress Example — Temperature Change in a Line of Cells
# ══════════════════════════════════════════════════════════════════════════
#
# Demonstrates thermo-mechanical coupling in JutulTPSA by applying a
# strong temperature change along a horizontal line of cells in a 2-D
# vertical cross-section and plotting the resulting effective stress
# distribution vertically.
#
# Physical setup:
#   - 2-D rectangular domain (width × height) representing a vertical
#     cross-section of a subsurface formation.
#   - All boundaries are clamped (zero displacement).
#   - A horizontal band of cells at mid-height receives a large ΔT.
#   - The thermal expansion generates compressive stress in the heated
#     zone and tensile stress above/below.
#
# The effective stress σ_eff is defined as the von Mises equivalent:
#   σ_VM = √(σ_xx² + σ_yy² - σ_xx σ_yy + 3 σ_xy²)
# for plane-strain conditions.
# ══════════════════════════════════════════════════════════════════════════

using JutulTPSA
using Jutul
using LinearAlgebra
using Printf

# ──────────────────────────────────────────────────────────────────────────
# Problem parameters
# ──────────────────────────────────────────────────────────────────────────
const WIDTH  = 10.0    # domain width  [m]
const HEIGHT = 50.0    # domain height [m]
const NX     = 5       # cells in x
const NY     = 25      # cells in y

const E_rock   = 20e9      # Young's modulus [Pa]  (sandstone-like)
const ν_rock   = 0.25      # Poisson's ratio
const α_biot   = 0.8       # Biot coefficient
const α_T      = 1.2e-5    # thermal expansion coefficient [1/K]
const T_ref    = 350.0     # reference temperature [K]  (≈ 77 °C)
const ΔT_hot   = 150.0     # temperature increase in heated cells [K]

# ──────────────────────────────────────────────────────────────────────────
# Helper: identify cells in a horizontal band
# ──────────────────────────────────────────────────────────────────────────
function cells_in_band(ny, j_lo, j_hi, nx)
    cells = Int[]
    for j in j_lo:j_hi, i in 1:nx
        push!(cells, (j - 1) * nx + i)
    end
    return cells
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
# Von Mises effective stress (plane strain, 2-D Voigt: [σ_xx, σ_yy, σ_xy])
# ──────────────────────────────────────────────────────────────────────────
function von_mises_2d(σ_xx, σ_yy, σ_xy)
    return sqrt(σ_xx^2 + σ_yy^2 - σ_xx * σ_yy + 3 * σ_xy^2)
end

# ──────────────────────────────────────────────────────────────────────────
# Main example
# ──────────────────────────────────────────────────────────────────────────
function run_thermal_stress_example()
    println("=" ^ 72)
    println("  Thermal Stress: ΔT = $(ΔT_hot) K applied at mid-height")
    println("=" ^ 72)

    # ── Build mesh and model ────────────────────────────────────────────
    g = CartesianMesh((NX, NY), (WIDTH, HEIGHT))
    domain = DataDomain(g)
    nc = number_of_cells(domain)
    dx = WIDTH / NX
    dy = HEIGHT / NY

    model, state0, param = setup_tpsa_model(domain;
        dim     = 2,
        coupled = true,
        E       = E_rock,
        ν       = ν_rock,
        biot    = α_biot,
        alpha_T = α_T,
        T_ref   = T_ref,
        p_ref   = 0.0,
    )

    # ── Apply temperature perturbation along mid-height band ────────────
    # Heated band: rows j_lo to j_hi (1-indexed from bottom)
    j_mid = div(NY, 2)
    j_lo  = max(1, j_mid - 1)
    j_hi  = min(NY, j_mid + 1)
    hot_cells = cells_in_band(NY, j_lo, j_hi, NX)

    # Set temperature: T_ref everywhere except heated cells
    T_field = fill(T_ref, nc)
    for c in hot_cells
        T_field[c] = T_ref + ΔT_hot
    end
    param[:Temperature] .= T_field
    param[:PorePressure] .= 0.0   # no pressure contribution

    # ── Boundary conditions: clamp all boundary cells ───────────────────
    bnd = boundary_cells_2d(NX, NY)
    bc  = DisplacementConstraint(bnd, [0.0, 0.0])
    forces = setup_forces(model; dirichlet = bc)

    # ── Solve ───────────────────────────────────────────────────────────
    println("\n  Solving thermo-mechanical problem …")
    states, = tpsa_solve(model, state0, param;
        info_level = -1, forces = forces)
    sol = states[end]

    # ── Extract vertical profiles along the centre column ───────────────
    # Centre column: cells with i = ceil(NX/2)
    i_mid = cld(NX, 2)
    col_cells = [(j - 1) * NX + i_mid for j in 1:NY]

    cc = domain[:cell_centroids]      # 2 × nc
    σ  = sol[:ElasticStress]          # 3 × nc  (Voigt: σ_xx, σ_yy, σ_xy)
    U  = sol[:Displacement]           # 2 × nc

    y_profile     = [cc[2, c] for c in col_cells]
    T_profile     = [T_field[c] for c in col_cells]
    uy_profile    = [U[2, c] for c in col_cells]
    σ_yy_profile  = [σ[2, c] for c in col_cells]
    vm_profile    = [von_mises_2d(σ[1,c], σ[2,c], σ[3,c]) for c in col_cells]

    # ── Print results table ─────────────────────────────────────────────
    println("\n  Vertical profile along centre column (x = $(cc[1, col_cells[1]]) m):")
    println("  " * "-" ^ 68)
    @printf("  %8s  %8s  %12s  %12s  %12s\n",
        "y [m]", "ΔT [K]", "u_y [m]", "σ_yy [Pa]", "σ_VM [Pa]")
    println("  " * "-" ^ 68)
    for k in eachindex(col_cells)
        ΔT_k = T_profile[k] - T_ref
        @printf("  %8.2f  %8.1f  %12.4e  %12.4e  %12.4e\n",
            y_profile[k], ΔT_k, uy_profile[k], σ_yy_profile[k], vm_profile[k])
    end

    # ── ASCII plot of von Mises stress vs depth ─────────────────────────
    println("\n  Von Mises Effective Stress vs Depth (centre column)")
    println("  " * "=" ^ 62)

    vm_max = maximum(vm_profile)
    plot_width = 50
    for k in eachindex(col_cells)
        bar_len = vm_max > 0 ? round(Int, vm_profile[k] / vm_max * plot_width) : 0
        bar_len = clamp(bar_len, 0, plot_width)
        marker = T_profile[k] > T_ref + 1 ? "█" : "▓"
        bar = marker ^ bar_len
        @printf("  %6.1f m |%s\n", y_profile[k], bar)
    end
    println("  " * "-" ^ 62)
    @printf("  Scale: |%s| = %.2e Pa (von Mises)\n", "█" ^ plot_width, vm_max)
    println("  █ = heated cells,  ▓ = unheated cells")

    # ── ASCII plot of σ_yy vs depth ─────────────────────────────────────
    println("\n  Vertical Stress σ_yy vs Depth (centre column)")
    println("  " * "=" ^ 62)

    σ_abs_max = maximum(abs.(σ_yy_profile))
    half_width = 25
    for k in eachindex(col_cells)
        if σ_abs_max > 0
            pos = round(Int, σ_yy_profile[k] / σ_abs_max * half_width)
        else
            pos = 0
        end
        pos = clamp(pos, -half_width, half_width)
        # Build bar centred at half_width
        line = fill(' ', 2 * half_width + 1)
        line[half_width + 1] = '|'
        if pos > 0
            for p in (half_width + 2):(half_width + 1 + pos)
                line[p] = '+'
            end
        elseif pos < 0
            for p in (half_width + 1 + pos):(half_width)
                line[p] = '-'
            end
        end
        @printf("  %6.1f m  %s\n", y_profile[k], String(line))
    end
    println("  " * "-" ^ 62)
    @printf("  Scale: each character ≈ %.2e Pa\n", σ_abs_max / half_width)
    println("  '+' = tensile σ_yy,  '-' = compressive σ_yy")

    println("\n" * "=" ^ 72)
    println("  Summary:")
    @printf("    Max von Mises stress:   %12.4e Pa\n", vm_max)
    @printf("    Max |σ_yy|:             %12.4e Pa\n", σ_abs_max)
    @printf("    Max |u_y|:              %12.4e m\n", maximum(abs.(uy_profile)))
    @printf("    Heated band:            rows %d–%d  (y = %.1f–%.1f m)\n",
        j_lo, j_hi, (j_lo - 0.5) * dy, (j_hi + 0.5) * dy)
    println("=" ^ 72)
end

run_thermal_stress_example()
