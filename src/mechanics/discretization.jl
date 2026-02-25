# ──────────────────────────────────────────────────────────────────────────
# TPSA Discretization
# ──────────────────────────────────────────────────────────────────────────

"""
    TPSADiscretization

Stores the geometric information required for the Two-Point Stress
Approximation on a given mesh.

For each half-face the following are precomputed and stored:
- `half_face_map` — the standard Jutul half-face connectivity (cells, faces,
  face_pos, face_sign).
- `T_hf` — half-face TPSA transmissibility matrix (dim × dim SMatrix) for
  unit elastic moduli; scaled at assembly time by cell properties.
- `normals_hf` — outward unit face-normal at the half-face (SVector{dim}).
- `areas_hf` — face area at the half-face.
- `dist_hf` — distance from the cell centroid to the face centroid projected
  onto the face normal (always > 0).

The per-half-face TPSA transmissibility for isotropic material with Lamé
constants λ and μ is:

    T_hf^{αβ} = (A_f / h_hf) * [μ δ_{αβ} + (λ + μ) n^α n^β]

where h_hf is the distance from the cell centroid to the face centroid
projected onto n.
"""
struct TPSADiscretization{N, T, HFM} <: JutulDiscretization
    half_face_map::HFM   # Jutul half_face_map named tuple
    normals_hf::Vector{SVector{N, T}}
    areas_hf::Vector{T}
    dist_hf::Vector{T}
end

# ---- Constructor from DataDomain ------------------------------------------

function TPSADiscretization(domain::DataDomain)
    g = physical_representation(domain)
    N_nb = get_neighborship(g)            # 2 × nf Int array
    nc = number_of_cells(g)
    nf = number_of_faces(g)
    dim = Jutul.dim(g)

    # Retrieve geometry arrays from domain data
    cc = domain[:cell_centroids]   # dim × nc
    fc = domain[:face_centroids]   # dim × nf
    fn = domain[:normals]          # dim × nf  (outward from cell N[1,:])
    fa = domain[:areas]            # nf

    # Normalise face normals to unit vectors
    fn_unit = similar(fn)
    for f in 1:nf
        n = fn[:, f]
        fn_unit[:, f] = n / norm(n)
    end

    # Build the Jutul half-face map
    hfm = Jutul.half_face_map(N_nb, nc)

    # Number of half-faces
    nhf = length(hfm.faces)

    T_type = eltype(fa)
    normals_hf = Vector{SVector{dim, T_type}}(undef, nhf)
    areas_hf   = Vector{T_type}(undef, nhf)
    dist_hf    = Vector{T_type}(undef, nhf)

    for cell in 1:nc
        for idx in hfm.face_pos[cell]:(hfm.face_pos[cell+1]-1)
            f    = hfm.faces[idx]
            sgn  = hfm.face_sign[idx]
            # Unit outward normal from *this* cell's perspective
            n_f  = sgn > 0 ? fn_unit[:, f] : -fn_unit[:, f]
            A_f  = fa[f]
            # Distance from cell centroid to face centroid projected on normal
            diff = fc[:, f] - cc[:, cell]
            d    = abs(dot(diff, n_f))
            if d < eps(T_type)
                d = norm(diff)   # fall-back for degenerate geometry
            end
            normals_hf[idx] = SVector{dim, T_type}(n_f)
            areas_hf[idx]   = A_f
            dist_hf[idx]    = d
        end
    end

    return TPSADiscretization{dim, T_type, typeof(hfm)}(
        hfm, normals_hf, areas_hf, dist_hf
    )
end

# ---- Callable interface required by Jutul ----------------------------------

function (D::TPSADiscretization)(cell::Int, ::Cells)
    hfm = D.half_face_map
    loc  = hfm.face_pos[cell]:(hfm.face_pos[cell+1]-1)
    return (
        faces      = @views(hfm.faces[loc]),
        face_signs = @views(hfm.face_sign[loc]),
        neighbors  = @views(hfm.cells[loc]),
        normals    = @views(D.normals_hf[loc]),
        areas      = @views(D.areas_hf[loc]),
        dists      = @views(D.dist_hf[loc]),
    )
end

# ---- Domain discretization hook --------------------------------------------

function Jutul.discretize_domain(
        d::DataDomain,
        system::LinearElasticitySystem,
        ::Val{:default};
        kwarg...,
    )
    disc = TPSADiscretization(d)
    discretizations = (tpsa = disc,)
    return DiscretizedDomain(physical_representation(d), discretizations; kwarg...)
end

# ──────────────────────────────────────────────────────────────────────────
# Helper: compute Lamé constants from E and ν
# ──────────────────────────────────────────────────────────────────────────

"""
    lame_parameters(E, ν) -> (λ, μ)

Return the first Lamé constant λ and the shear modulus μ for an isotropic
linear elastic material with Young's modulus `E` [Pa] and Poisson's ratio `ν`.
"""
@inline function lame_parameters(E, ν)
    μ = E / (2*(1 + ν))
    λ = E*ν / ((1 + ν)*(1 - 2*ν))
    return (λ, μ)
end

"""
    bulk_modulus(E, ν) -> K

Drained bulk modulus K = E / (3(1-2ν)).
"""
@inline function bulk_modulus(E, ν)
    return E / (3*(1 - 2*ν))
end

# ──────────────────────────────────────────────────────────────────────────
# TPSA traction on a single half-face
# ──────────────────────────────────────────────────────────────────────────

"""
    tpsa_traction(u_self, u_other, n, A, d, λ, μ) -> SVector

Compute the TPSA traction vector on a half-face for cell pair (self, other).

    t^α = (A/d) [μ (u_other - u_self)^α + (λ+μ) ((u_other-u_self)·n) n^α]

This is the contribution of the face to the force balance of cell *self*
(positive = force entering *self*).
"""
@inline function tpsa_traction(u_self, u_other, n, A, d, λ, μ)
    Δu    = u_other - u_self
    coeff = A / d
    normal_proj = dot(Δu, n)
    return coeff * (μ * Δu + (λ + μ) * normal_proj * n)
end
