"""Shared 3D single-band effective-mass QD solver machinery.

Extracted from inSb_dot_in_inAs.ipynb so it can be reused across material systems, dot shapes,
and (eventually) a multiband extension without copy-pasting. See that notebook for the full
derivation/citations; this module only holds the tested numerical building blocks.

Citation policy (applies to all docstrings in this package): references are given at the
level of *named results* plus paper/chapter locations. Equation numbers from the paywalled
originals (Eshelby 1957, Davies 1998, BenDaniel-Duke 1966, ...) are deliberately NOT quoted,
because they could not be verified against the actual sources when these citations were
added -- a plausible-looking but wrong equation number is worse than none. Page references
are only given where there is a verifiable provenance chain (e.g., aestimo's own
aeslibs/VBHM.py header cites specific pages of Chuang and Harrison).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HBAR2_OVER_2M0 = 0.0380998  # eV * nm^2 -- hbar^2/(2*m0), converted from the standard 3.80998 eV*Angstrom^2

# Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001), except Band_offset which
# matches this repo's aestimo_database.py (calibrated for the GaAs-substrate family; see the
# InGaSb/InAsSb entries there for how the InAs/Sb-family type-II offsets were re-derived).
MATERIALS = {
    'GaAs': dict(
        m_e=0.067, m_h=0.51,
        a0=5.6533,
        Eg=1.4223, Band_offset=0.65,
        Ac=-7.17, Av=1.16,
        C11=118.79, C12=53.76, C44=59.4,
        gamma1=6.8, gamma2=1.9, gamma3=2.73, delta_so=0.28,  # Luttinger params + split-off (Vurgaftman2001)
        epsilonStatic=12.9,
    ),
    'InAs': dict(
        m_e=0.026, m_h=0.41,
        a0=6.0583,
        Eg=0.4, Band_offset=0.63,
        Ac=-5.08, Av=1.00,
        C11=83.29, C12=45.26, C44=39.59,
        gamma1=20.4, gamma2=8.3, gamma3=9.1, delta_so=0.38,
        epsilonStatic=15.15,
    ),
    'InSb': dict(
        m_e=0.0135, m_h=0.405,
        a0=6.4794,
        Eg=0.174, Band_offset=2.22,  # derived value, see database.py's InSb entry -- type-II family only
        Ac=-6.04, Av=0.31,
        epsilonStatic=17.5,
        C11=68.47, C12=37.35, C44=31.11,
        gamma1=34.8, gamma2=15.5, gamma3=16.5, delta_so=0.81,
    ),
}

for _name, _m in MATERIALS.items():
    _m['fi_e'] = _m['Band_offset'] * _m['Eg']
    _m['fi_h'] = -(1 - _m['Band_offset']) * _m['Eg']


def eigenstrain(a_dot, a_matrix):
    """Lattice-mismatch eigenstrain (negative = dot is compressed to fit the matrix).

    References: the misfit/eigenstrain of a buried dot, defined from the lattice constants of
    inclusion and matrix, as used by J. H. Davies, J. Appl. Phys. 84, 1358 (1998). Note that
    sign/normalization conventions (which lattice constant sits in the denominator) vary
    between authors; this one measures the strain needed to squeeze the free-standing dot
    material onto the matrix lattice.
    """
    return (a_matrix - a_dot) / a_dot


def voigt_poisson_ratio(C11, C12, C44):
    """Isotropic Poisson ratio approximation for a cubic crystal (Voigt average).

    References: Voigt averaging of cubic elastic constants -- R. Hill, Proc. Phys. Soc. A 65,
    349 (1952) (the classic Voigt/Reuss/Hill discussion). K = (C11+2*C12)/3 is exact for cubic
    symmetry; the shear average G_V = (C11-C12+3*C44)/5 and the resulting nu are the isotropic
    approximation, needed because the Eshelby/Davies strain results used downstream assume an
    isotropic medium.
    """
    K = (C11 + 2 * C12) / 3.0        # bulk modulus -- exact for cubic symmetry
    G = (C11 - C12 + 3 * C44) / 5.0  # shear modulus -- Voigt average (approximation)
    return (3 * K - 2 * G) / (2 * (3 * K + G))


def trace_strain_from_mask(inside_mask, eps_star, nu):
    """Tr(strain) field for an inclusion of ARBITRARY shape, under the same approximations
    used throughout this project (homogeneous isotropic elasticity, purely dilatational
    eigenstrain eps_star).

    This is NOT sphere-specific: Davies, J. Appl. Phys. 84, 1358 (1998) ("simple picture")
    shows that in this approximation the elastic field of any inclusion is a superposition of
    point centers of dilatation, each of whose dilatation is a delta function -- so Tr(strain)
    is exactly proportional to the inclusion's characteristic function: uniform inside (with
    the same value as the classic Eshelby sphere), exactly zero outside, for ANY shape. That
    is what makes lens/pyramid/ellipsoid shapes usable with the single-band hydrostatic-only
    model without a numerical elasticity solve. What is *not* shape-independent is the full
    strain tensor (shear/biaxial components) -- a multiband/Bir-Pikus treatment would need a
    real shape-dependent strain calculation.
    """
    eps_inside_diag = eps_star * (1 + nu) / (3 * (1 - nu))
    trace_inside = 3 * eps_inside_diag
    return np.where(inside_mask, trace_inside, 0.0), trace_inside


# References for trace_strain_from_mask:
#  - J. D. Eshelby, Proc. R. Soc. London A 241, 376 (1957): the ellipsoidal-inclusion
#    solution; the uniform interior strain of a sphere with dilatational eigenstrain, from
#    which the (1+nu)/(3(1-nu)) factor above comes.
#  - J. H. Davies, J. Appl. Phys. 84, 1358 (1998): the "simple picture" -- for arbitrary
#    inclusion shapes (isotropic, homogeneous elasticity, dilatational eigenstrain) the
#    displacement field is a superposition of point centers of dilatation, so Tr(strain) is
#    proportional to the inclusion's characteristic function.


def eshelby_sphere_trace_strain(inside_mask, eps_star, nu):
    """Backwards-compatible alias kept for the earlier sphere notebooks: the sphere is just
    the special case of trace_strain_from_mask (see that docstring for why the result is
    shape-independent)."""
    return trace_strain_from_mask(inside_mask, eps_star, nu)


# --- Dot shape masks (all take coordinate arrays from np.meshgrid(..., indexing='ij')) ---

def sphere_mask(X, Y, Z, radius, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center
    return (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 < radius ** 2


def ellipsoid_mask(X, Y, Z, rx, ry, rz, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center
    return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2 < 1.0


def lens_mask(X, Y, Z, base_radius, height, z_base=0.0):
    """Flat-bottomed circular lens (half-ellipsoid dome): the base is the disk
    x^2 + y^2 <= base_radius^2 in the plane z = z_base, and the dome rises to
    z_base + height on the z axis. base_radius sets the lateral (x,y) size and
    height sets the vertical (z) size independently."""
    zz = Z - z_base
    return (zz >= 0.0) & ((X ** 2 + Y ** 2) / base_radius ** 2 + (zz / height) ** 2 < 1.0)


def pyramid_mask(X, Y, Z, base, z_base=0.0):
    """Square-based pyramid with {101}-type side facets: base of side `base` centered on the
    z axis in the plane z = z_base, apex at z_base + base/2. The {101} facets fix the aspect
    ratio (height = base/2), so the single parameter `base` sets the size -- this is the
    island geometry of C. Pryor, Phys. Rev. B 57, 7190 (1998) (Fig. 1 there; no wetting
    layer, matching that paper's dot-only calculations), chosen to allow direct comparison
    against its one-band/eight-band benchmark energies. Analytic volume for sanity checks:
    base^2 * (base/2) / 3."""
    zz = Z - z_base
    half = base / 2.0
    return (zz >= 0.0) & (np.abs(X) < half - zz) & (np.abs(Y) < half - zz)


def build_hamiltonian(mass_field, potential_field, h):
    """Assemble the sparse 3D Ben Daniel-Duke Hamiltonian (eV) on a cubic grid of spacing h (nm).

    Position-dependent mass is discretized with the standard harmonic-mean prescription for
    probability-current continuity across a mass discontinuity. Hard-wall (Dirichlet) boundary
    conditions are enforced explicitly via a "ghost neighbor" diagonal contribution at each of
    the six box faces -- omitting this silently turns the box into a reflecting (Neumann)
    boundary instead, which was caught here by testing against the analytic infinite-cubic-well
    ground state (3*hbar^2*pi^2/(2*m*L^2)) before trusting any QD result.

    References: the (1/m(r)) operator ordering of the kinetic term is the BenDaniel-Duke
    Hamiltonian -- D. J. BenDaniel and C. B. Duke, Phys. Rev. 152, 683 (1966). The
    finite-difference/shooting treatment of position-dependent mass in heterostructures is
    covered in P. Harrison, "Quantum Wells, Wires and Dots" (Wiley) -- the same reference
    aestimo's aeslibs/VBHM.py cites (pp. 357-362 in its header, for the related k.p+FDM
    machinery); aestimo's own 1D `Schro` function uses the same prescription.
    """
    Nx, Ny, Nz = mass_field.shape
    idx = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)

    diag = potential_field.copy().astype(float)
    rows, cols, data = [], [], []

    for axis in range(3):
        m_next = np.take(mass_field, np.arange(1, mass_field.shape[axis]), axis=axis)
        m_here = np.take(mass_field, np.arange(0, mass_field.shape[axis] - 1), axis=axis)
        m_half = 2 * m_here * m_next / (m_here + m_next)  # harmonic mean
        t = HBAR2_OVER_2M0 / (m_half * h**2)

        idx_here = np.take(idx, np.arange(0, idx.shape[axis] - 1), axis=axis)
        idx_next = np.take(idx, np.arange(1, idx.shape[axis]), axis=axis)

        rows.append(idx_here.ravel()); cols.append(idx_next.ravel()); data.append(-t.ravel())
        rows.append(idx_next.ravel()); cols.append(idx_here.ravel()); data.append(-t.ravel())

        diag_contrib = np.zeros_like(diag)
        slicer_here = [slice(None)] * 3; slicer_here[axis] = slice(0, -1)
        slicer_next = [slice(None)] * 3; slicer_next[axis] = slice(1, None)
        diag_contrib[tuple(slicer_here)] += t
        diag_contrib[tuple(slicer_next)] += t
        diag += diag_contrib

        lo = [slice(None)] * 3; lo[axis] = 0
        hi = [slice(None)] * 3; hi[axis] = -1
        diag[tuple(lo)] += HBAR2_OVER_2M0 / (mass_field[tuple(lo)] * h**2)
        diag[tuple(hi)] += HBAR2_OVER_2M0 / (mass_field[tuple(hi)] * h**2)

    rows.append(idx.ravel()); cols.append(idx.ravel()); data.append(diag.ravel())

    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    n = Nx * Ny * Nz
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def solve_states(mass_field, potential_field, h, n_states=4, hole=False):
    """Solve for the lowest n_states confined states. Set hole=True to use the sign-flip
    trick (solve for -potential_field, negate eigenvalues back) appropriate for a carrier
    confined where potential_field is a local *maximum* rather than a minimum.

    Uses a custom shift-invert factorization (permc_spec='MMD_AT_PLUS_A') rather than
    eigsh's internal default (COLAMD), which was measured to be 4-8x slower on this 3D
    grid's sparsity pattern (e.g. ~38s vs ~10s to factorize a 41^3-point Hamiltonian).

    References: shift-invert Lanczos via ARPACK -- R. B. Lehoucq, D. C. Sorensen, and
    C. Yang, "ARPACK Users' Guide" (SIAM, 1998), which is what scipy's eigsh wraps.
    """
    sign = -1.0 if hole else 1.0
    H = build_hamiltonian(mass_field, sign * potential_field, h)
    sigma = (sign * potential_field).min() - 1e-3
    n = H.shape[0]
    shifted = (H - sigma * sp.identity(n)).tocsc()
    lu = spla.splu(shifted, permc_spec='MMD_AT_PLUS_A')
    OPinv = spla.LinearOperator((n, n), matvec=lu.solve)
    E, psi = spla.eigsh(H, k=n_states, sigma=sigma, which='LM', OPinv=OPinv)
    order = np.argsort(E)
    E, psi = E[order], psi[:, order]
    if hole:
        E = -E
    cell_vol = h**3
    psi = psi / np.sqrt((psi**2).sum(axis=0) * cell_vol)
    return E, psi


def particle_in_box_ground_state(m, L):
    """Analytic ground state (eV, above the potential floor) of an infinite cubic box of
    side L (nm) for a particle of mass m (m0). Useful as a sanity check / convergence target.

    References: textbook result (E_111 = 3*hbar^2*pi^2/(2*m*L^2) for the 3D infinite cubic
    well) -- see, e.g., D. J. Griffiths, "Introduction to Quantum Mechanics", the separable
    3D infinite-square-well problem.
    """
    return 3 * HBAR2_OVER_2M0 * np.pi**2 / (m * L**2)
