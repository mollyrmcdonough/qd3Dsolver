"""Full strain tensor for a buried inclusion of arbitrary shape, by Fourier continuum elasticity.

This replaces the hydrostatic-only model in qdsolver_core.trace_strain_from_mask, which returns
Tr(eps) and nothing else. Every shear component was discarded there, so the Bir-Pikus b and d
deformation potentials -- which Pryor states dominate hole confinement in these dots -- could not
enter the valence-band Hamiltonian at all.

Method
------
Linear continuum elasticity with a dilatational eigenstrain in the dot, solved exactly in Fourier
space. Writing the stress as sigma_ij = C_ijkl (eps_kl - eps*_kl) with the transformation strain
eps*_kl = eps_T delta_kl chi(r) confined to the dot (chi = the shape's characteristic function),
mechanical equilibrium div sigma = 0 becomes, for the displacement field u,

    C_ijkl d_j d_l u_k = C_ijkl d_j eps*_kl

which in Fourier space is a 3x3 linear system per wavevector,

    K_ik(k) u_k(k) = -i eps_T (C11 + 2 C12) chi(k) k_i ,

with K_ik = C_ijkl k_j k_l the acoustic (Christoffel) tensor. Solving it and forming
eps_ij = (i/2)(k_j u_i + k_i u_j) gives the complete strain tensor in one FFT round trip.

This is the same construction as A. D. Andreev, J. R. Downes, D. A. Faux and E. P. O'Reilly,
"Strain distributions in quantum dots of arbitrary shape", J. Appl. Phys. 86, 297 (1999): the
shape enters only through chi(k), so any mask from qdsolver_core works unchanged, and the full
anisotropic cubic C11/C12/C44 are kept rather than Voigt-averaged to an isotropic nu.

Assumptions, and how they differ from the model this replaces
-------------------------------------------------------------
KEPT, and this is the point: all six strain components, with cubic elastic anisotropy.

STILL APPROXIMATE:
  * Homogeneous elastic constants -- SUPERSEDED for quantitative work by elasticity_fd.
    Constant C_ijkl is what makes the Fourier solve exact and O(N log N), but the real dot and
    matrix differ (InAs is ~25% softer than GaAs in bulk modulus), and the caller must pass one
    set for both. This is not a small effect and it is not bracketed by the two choices: for a
    misfitting inclusion the dilatation depends on the INCLUSION's bulk modulus and the MATRIX's
    shear modulus, so the true value lies outside the interval any single choice can reach
    (measured: homogeneous [-0.0841, -0.0927] vs true -0.1068, worth ~15% in the band-edge
    shift; see cb_depth_diagnostic.py). Use elasticity_fd.solve_strain_fd, which minimizes the
    strain energy in real space and carries position-dependent constants -- that is Pryor's own
    method in Phys. Rev. B 57, 7190 (1998). This module remains the right choice when the
    constants really are uniform, as an independent check on the FD solver (they agree in that
    limit), and for speed.
  * Linear elasticity. At ~7% mismatch this is being pushed; nonlinear corrections exist in the
    literature but linear is still what most production k.p work uses.
  * Continuum, so the result carries the C4v symmetry of the pyramid rather than the true C2v
    symmetry of the zincblende lattice. Only an atomistic (valence force field) relaxation gets
    that right; see C. Pryor, J. Kim, L. W. Wang, A. J. Williamson and A. Zunger, "Comparison of
    two methods for describing the strain profiles in quantum dots", J. Appl. Phys. 83, 2548
    (1998) for a direct comparison of the two approaches.
  * Periodic boundary conditions with the k = 0 mode dropped, i.e. zero mean TOTAL strain. That
    is the clamped-cell condition, which is the correct infinite-matrix limit only if the
    padding is large enough that the strain has decayed at the box faces. `padding_report`
    measures that directly; it is not assumed.

UNBLOCKED BY THE SHEAR COMPONENTS, and now implemented in piezoelectric.py: the shear drives a
polarization P_i = 2 e14 eps_jk (i,j,k cyclic) in zincblende, whose bound charge -div P sources
an electrostatic potential. This was simply not computable under the hydrostatic-only model --
there was no shear to feed it. Note that it is NOT the only term that lowers the symmetry of a
C4v pyramid: the multiband p-doublet splitting was measured here to be nonzero (3.0 meV at four
bands, 9.4 meV at six) with the piezoelectric potential switched off, because the Bir-Pikus
shear terms already couple the bands anisotropically. See piezoelectric.py for what first-order
piezoelectricity does and does not capture.

Sign convention
---------------
`eps_star` is exactly what qdsolver_core.eigenstrain(a_dot, a_matrix) returns: negative when the
dot material must be compressed to fit the matrix. The transformation strain used internally is
-eps_star (referred to the matrix lattice), and the returned components are the ELASTIC strain
that the deformation potentials act on -- total strain minus eigenstrain -- so that they are
measured relative to each material's own unstrained lattice.

The clean check of that convention is `pseudomorphic_layer_strain`: for a slab spanning the whole
cell this solver must return exactly eps_xx = eps_yy = eps_star and eps_zz = -2 (C12/C11) eps_star
inside the layer, and zero outside. That is the textbook pseudomorphic result, it is exact rather
than asymptotic, and it is anisotropic, so it pins down both the sign convention and the elastic
tensor at once.
"""
import numpy as np


class StrainTensor:
    """The six independent components of a symmetric strain tensor, each a grid-shaped array.

    Tensor (not engineering) shear convention throughout: exy = (du_x/dy + du_y/dx)/2, so the
    engineering shear is twice these. Bir-Pikus matrix elements are written for the tensor
    components, which is why this matters.
    """

    __slots__ = ('exx', 'eyy', 'ezz', 'exy', 'eyz', 'ezx')

    # Iterate over THIS, never over __slots__: a subclass that adds slots of its own (as
    # elasticity_fd.FDStrainTensor does, to carry solver convergence data) shadows __slots__
    # with only its own names, and every component-wise method would then silently operate on
    # the wrong attributes.
    COMPONENTS = ('exx', 'eyy', 'ezz', 'exy', 'eyz', 'ezx')

    def __init__(self, exx, eyy, ezz, exy, eyz, ezx):
        self.exx, self.eyy, self.ezz = exx, eyy, ezz
        self.exy, self.eyz, self.ezx = exy, eyz, ezx

    @property
    def trace(self):
        return self.exx + self.eyy + self.ezz

    @property
    def biaxial(self):
        """exx + eyy - 2 ezz, the combination the Bir-Pikus b deformation potential couples to."""
        return self.exx + self.eyy - 2 * self.ezz

    @property
    def shape(self):
        return self.exx.shape

    def as_dict(self):
        return {k: getattr(self, k) for k in self.COMPONENTS}

    def at(self, mask):
        """Mean of each component over a boolean mask -- the usual way to quote 'the strain in
        the dot' for a field that is not actually uniform.

        Use `at_rms` for the shear components. Every dot shape here is symmetric under x -> -x
        and y -> -y, so <exy>, <eyz> and <ezx> vanish identically over the dot no matter how
        large the local shear is; a mean-based shear diagnostic reports zero and looks like a
        solver bug.
        """
        return {k: float(getattr(self, k)[mask].mean()) for k in self.COMPONENTS}

    def at_rms(self, mask):
        """Root-mean-square of each component over a boolean mask. The meaningful magnitude for
        the shear components, which average to zero by symmetry."""
        return {k: float(np.sqrt((getattr(self, k)[mask] ** 2).mean()))
                for k in self.COMPONENTS}

    def __repr__(self):
        return (f"StrainTensor(shape={self.shape}, "
                f"trace in [{self.trace.min():.4f}, {self.trace.max():.4f}])")


def _christoffel_inverse(kx, ky, kz, C11, C12, C44):
    """Inverse of K_ik = C_ijkl k_j k_l for cubic symmetry, as six real arrays (K is symmetric).

    Computed by the explicit cofactor formula rather than np.linalg.solve on a stacked (...,3,3)
    array: at 128^3 the stacked form allocates a few hundred MB of temporaries for what is a
    closed-form 3x3 inverse. The k = 0 entry is singular (rigid translation is undetermined) and
    is masked to zero here; the caller relies on that to drop the DC mode.
    """
    k2x, k2y, k2z = kx * kx, ky * ky, kz * kz
    cs = C12 + C44

    K11 = C11 * k2x + C44 * (k2y + k2z)
    K22 = C11 * k2y + C44 * (k2x + k2z)
    K33 = C11 * k2z + C44 * (k2x + k2y)
    K12 = cs * kx * ky
    K13 = cs * kx * kz
    K23 = cs * ky * kz

    A11 = K22 * K33 - K23 * K23
    A12 = K13 * K23 - K12 * K33
    A13 = K12 * K23 - K13 * K22
    A22 = K11 * K33 - K13 * K13
    A23 = K12 * K13 - K11 * K23
    A33 = K11 * K22 - K12 * K12

    det = K11 * A11 + K12 * A12 + K13 * A13
    nonzero = det != 0.0
    inv_det = np.zeros_like(det)
    np.divide(1.0, det, out=inv_det, where=nonzero)

    return (A11 * inv_det, A12 * inv_det, A13 * inv_det,
            A22 * inv_det, A23 * inv_det, A33 * inv_det)


def solve_strain(inside_mask, eps_star, C11, C12, C44, h, return_total=False):
    """Elastic strain tensor for `inside_mask` embedded in a matrix, on a uniform grid.

    Parameters
    ----------
    inside_mask : bool array (Nx, Ny, Nz)
        The dot's characteristic function, from any qdsolver_core `*_mask` helper.
    eps_star : float
        qdsolver_core.eigenstrain(a_dot, a_matrix); negative for a compressed dot.
    C11, C12, C44 : float
        Cubic elastic constants (any consistent units -- only ratios matter here).
    h : float
        Grid spacing (nm). Only sets the units of the intermediate displacement field; the
        strain is homogeneous of degree zero in k and so is independent of h.
    return_total : bool
        If True, also return the TOTAL strain (the compatible strain derived from u, without
        the eigenstrain subtracted). Useful for checking div sigma = 0 and for piezoelectric
        polarization, which is driven by the total strain's shear components.

    Returns
    -------
    StrainTensor (elastic), or (elastic, total) if return_total.
    """
    chi = np.ascontiguousarray(inside_mask, dtype=float)
    shape = chi.shape
    eps_T = -float(eps_star)  # transformation strain, referred to the matrix lattice

    chi_k = np.fft.fftn(chi)
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=h)
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=h)
    kz = 2 * np.pi * np.fft.fftfreq(shape[2], d=h)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    I11, I12, I13, I22, I23, I33 = _christoffel_inverse(KX, KY, KZ, C11, C12, C44)

    # u_k = -i eps_T (C11 + 2 C12) chi_k * [K^-1 k]_k ; the k = 0 mode is already zeroed by the
    # masked inverse, which is exactly the "zero mean total strain" (clamped cell) condition.
    pref = -1j * eps_T * (C11 + 2 * C12) * chi_k
    ux = pref * (I11 * KX + I12 * KY + I13 * KZ)
    uy = pref * (I12 * KX + I22 * KY + I23 * KZ)
    uz = pref * (I13 * KX + I23 * KY + I33 * KZ)

    def ifft_real(arr):
        return np.fft.ifftn(arr).real

    exx = ifft_real(1j * KX * ux)
    eyy = ifft_real(1j * KY * uy)
    ezz = ifft_real(1j * KZ * uz)
    exy = ifft_real(0.5j * (KY * ux + KX * uy))
    eyz = ifft_real(0.5j * (KZ * uy + KY * uz))
    ezx = ifft_real(0.5j * (KX * uz + KZ * ux))

    total = StrainTensor(exx, eyy, ezz, exy, eyz, ezx)

    # Elastic strain = total - eigenstrain. The eigenstrain is purely dilatational, so only the
    # diagonal is affected and the shear components are already elastic.
    off = eps_T * chi
    elastic = StrainTensor(exx - off, eyy - off, ezz - off, exy, eyz, ezx)

    return (elastic, total) if return_total else elastic


# ---------------------------------------------------------------------------------------
# Analytic references used to validate the solver
# ---------------------------------------------------------------------------------------

def pseudomorphic_layer_strain(eps_star, C11, C12):
    """Exact elastic strain in a [001] pseudomorphic layer: (eps_parallel, eps_zz).

    A layer thin enough to strain coherently onto a thick substrate takes the substrate's
    in-plane lattice constant, so eps_xx = eps_yy = eps_star by definition, and relaxes freely
    along z under the zero-stress condition sigma_zz = 0, giving eps_zz = -2 (C12/C11) eps_par.

    This is the standard result behind every strained-quantum-well calculation (it is the same
    formula aestimo's Strain_and_Masses uses), it is exact within linear elasticity, and it is
    anisotropic -- which makes it the sharpest available test of solve_strain's sign convention
    and elastic tensor.

    It is the fill-fraction -> 0 limit of `clamped_slab_strain`, NOT what solve_strain returns
    for a slab occupying an appreciable part of a periodic cell -- see that function.
    """
    return eps_star, -2.0 * (C12 / C11) * eps_star


def clamped_slab_strain(eps_star, C11, C12, fill_fraction):
    """Exact elastic strain in a [001] slab filling `fill_fraction` of a CLAMPED periodic cell.

    This, not `pseudomorphic_layer_strain`, is what solve_strain must reproduce for a slab,
    because a periodic cell of zero mean total strain is a strained superlattice held at a fixed
    average lattice constant -- the barrier deforms too, and shares the mismatch with the slab in
    proportion to the fill fraction f.

    Only k = (0, 0, kz) survives for a slab, so the Fourier solution is a one-liner: the total
    eps_zz is eps_T (C11 + 2 C12)/C11 inside, minus its cell average (the dropped DC mode), and
    the elastic strain subtracts eps_T. In-plane there is no displacement at all, so the total
    eps_xx is exactly zero and the elastic eps_xx is exactly eps_star for any f.

        eps_par = eps_star
        eps_zz  = -eps_star [ (C11 + 2 C12)/C11 (1 - f) - 1 ]

    At f -> 0 this collapses to the pseudomorphic -2(C12/C11) eps_star. The f-dependence is the
    finite-cell clamping error in closed form, which is why this is the useful test: it validates
    the solver AND calibrates how much padding a real dot needs.
    """
    eps_T = -eps_star
    return eps_star, eps_T * ((C11 + 2.0 * C12) / C11 * (1.0 - fill_fraction) - 1.0)


def isotropic_sphere_trace(eps_star, nu):
    """Exact Tr(eps_elastic) inside an isotropic Eshelby sphere: 2(1-2nu)/(1-nu) * eps_star.

    Derivation: for a dilatational eigenstrain eps_T the constrained (total) interior strain is
    uniform, eps^c = alpha eps_T delta with alpha = (1+nu)/(3(1-nu)) -- J. D. Eshelby, Proc. R.
    Soc. London A 241, 376 (1957). The ELASTIC strain, which is what the deformation potentials
    see, is eps^c - eps* = (alpha - 1) eps_T delta, and

        3 (alpha - 1) = -2 (1 - 2 nu) / (1 - nu)

    so with eps_T = -eps_star the trace is +2(1-2nu)/(1-nu) eps_star.

    NOTE the factor relative to qdsolver_core.trace_strain_from_mask, which returns
    (1+nu)/(1-nu) * eps_star -- the CONSTRAINED strain, not the elastic strain. The two differ by
    (1+nu)/(2(1-2nu)), which is 1.165x at the Voigt nu = 0.235 of GaAs, so the older hydrostatic
    model overstates the band-edge shift by that factor.

    solve_strain reproduces this only up to the O(fill fraction) clamping correction of the
    finite periodic cell, exactly as `clamped_slab_strain` quantifies for the slab geometry.
    """
    return 2.0 * (1.0 - 2.0 * nu) / (1.0 - nu) * eps_star


def isotropic_constants(nu, C11=100.0):
    """Cubic constants (C11, C12, C44) satisfying the isotropy condition C44 = (C11-C12)/2 for a
    given Poisson ratio. Lets solve_strain be driven into the regime where the Eshelby sphere
    and Davies results apply exactly, so they can be used as ground truth."""
    C12 = C11 * nu / (1.0 - nu)
    return C11, C12, 0.5 * (C11 - C12)


def padding_report(strain, inside_mask):
    """How well the clamped periodic cell approximates an infinite matrix.

    Returns (max |Tr eps| on the box faces) / (mean |Tr eps| in the dot). The periodic solve
    forces zero mean total strain over the cell, so the elastic strain carries a spurious
    uniform offset of -3 eps_T * (dot volume fraction) that does not decay with distance. This
    ratio tracks that offset, and it is the number to drive down with padding: it is O(f), not
    O(exp(-distance)), so doubling the box helps only as fast as the volume fraction falls.

    A uniform strain cannot simply be subtracted to fix this -- a homogeneous strain added to
    the far field would have to be stress-free, and C eps = 0 forces it to vanish. Enlarging the
    cell is the only remedy within this method.
    """
    tr = strain.trace
    faces = np.concatenate([np.abs(tr[0]).ravel(), np.abs(tr[-1]).ravel(),
                            np.abs(tr[:, 0]).ravel(), np.abs(tr[:, -1]).ravel(),
                            np.abs(tr[:, :, 0]).ravel(), np.abs(tr[:, :, -1]).ravel()])
    inner = np.abs(tr[inside_mask]).mean()
    return float(faces.max() / inner) if inner > 0 else float('nan')
