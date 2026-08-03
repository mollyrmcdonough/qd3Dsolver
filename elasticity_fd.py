"""Inhomogeneous continuum elasticity: strain energy minimized by conjugate gradient.

This is Pryor's method rather than ours. C. Pryor, Phys. Rev. B 57, 7190 (1998), Sec. II:

    "First, the strain is calculated using linear continuum elastic theory. The strain energy
     for the system is computed using a finite differencing approximation, and then minimized
     using the conjugate gradient algorithm."

(the strain-energy functional there cites L. D. Landau and E. M. Lifshitz, "Theory of
Elasticity", Pergamon, London, 1959), and the calculation is "done on a cubic grid with
periodic boundary conditions".

Why this exists alongside strain_fourier.solve_strain
-----------------------------------------------------
Both solve the SAME physics -- linear continuum elasticity with a dilatational lattice-mismatch
eigenstrain, periodic cell -- and both find the same stationary point of the same energy
functional. They differ only in method, and in exactly one consequence of it:

    strain_fourier : diagonalize the Christoffel tensor per wavevector. One FFT round trip,
                     no iteration, no discretization error in the elasticity operator. But the
                     Fourier diagonalization REQUIRES spatially constant C_ijkl.
    this module    : discretize the energy in real space and minimize it. Costs an iteration
                     and a discretization error, but carries position-dependent C_ijkl for
                     free -- each element simply uses its own material's constants.

That difference is not academic. InAs is ~25% softer than GaAs in bulk modulus, and for a
misfitting inclusion the elastic dilatation is

    Tr(eps) = -3 eps_T * 4 mu_m / (3 K_i + 4 mu_m)

-- the INCLUSION's bulk modulus and the MATRIX's shear modulus, and nothing else. A homogeneous
solve must take both from one material, so it cannot reach the true value for any choice of
constants: the homogeneous InAs/GaAs bracket is [-0.0841, -0.0927] while the true inhomogeneous
value is -0.1068, outside it. Measured against Pryor's quoted conduction-well depths this is
worth ~15%, and it is the entire residual discrepancy in pryor_fig2 (see cb_depth_diagnostic.py).

Discretization
--------------
Trilinear (Q1) hexahedral finite elements with full 2x2x2 Gauss quadrature. Displacements live
on grid NODES; strain and material constants live on ELEMENTS. Because the elements are
identical cubes, the shape-function derivatives are the same for every element and the whole
operator reduces to a fixed 8x8 (corner-pair) x 3x3 (component) coefficient array, applied
matrix-free by shifted multiply-adds.

Q1 with full quadrature is used rather than the plainer "average the corner differences"
stencil, which is the same element with one-point quadrature and admits zero-energy hourglass
modes in 3D -- those would be excited by the sharp dot boundary and appear as checkerboard
noise in the displacement field.

Element-centred strain is a feature, not a compromise: the element centres form a uniform cubic
grid of the same spacing, so `solve_strain_fd` takes and returns fields on exactly the grid the
caller already uses, with no interpolation anywhere and no averaging across the material
interface (where the strain is genuinely discontinuous).

Preconditioning
---------------
Plain CG on this operator converges slowly for the same reason any 3D elliptic grid operator
does. The preconditioner used here is the EXACT inverse of the corresponding HOMOGENEOUS
operator, applied in Fourier space: the uniform-material stencil is constant-coefficient, so its
symbol S(k) = sum_ab K[a,b] exp(i k.(b-a) h) is an explicit 3x3 Hermitian matrix that can be
inverted per wavevector. What is left for CG to resolve is only the material CONTRAST, which is
mild (0.75), so convergence takes tens of iterations rather than thousands. This is the
preconditioning idea behind FFT-based homogenization schemes -- H. Moulinec and P. Suquet,
Comput. Methods Appl. Mech. Engrg. 157, 69 (1998) -- used here purely as a preconditioner, so
it changes the cost of the solve and not its answer.

The k = 0 mode is singular (rigid translation is undetermined) and is projected out, which is
the same zero-mean-displacement condition strain_fourier imposes by dropping its DC mode.
"""
import itertools

import numpy as np

from strain_fourier import StrainTensor

# Corner offsets of the reference hexahedron, and the corresponding +/-1 signs.
CORNERS = np.array(list(itertools.product([0, 1], repeat=3)))     # (8, 3), values 0/1
SIGNS = 2.0 * CORNERS - 1.0                                       # (8, 3), values -1/+1
_GAUSS = np.array(list(itertools.product([-1.0, 1.0], repeat=3))) / np.sqrt(3.0)   # (8, 3)


class FDStrainTensor(StrainTensor):
    """A StrainTensor that also carries how the iterative solve went.

    StrainTensor uses __slots__, so the convergence data cannot simply be attached to an
    instance -- and it should not be silently dropped either: unlike the Fourier solve, this one
    can fail to converge, and the caller needs to be able to see that it did not.

    `cg_converged` is the single field a caller should branch on. `cg_modes_dropped` is how many
    wavevectors the preconditioner had to leave uninverted; anything other than 1 (the k = 0
    rigid translation) means the preconditioner is filtering the solution and the field is
    suspect however small the residual looks -- see `_nonsingular`.
    """

    __slots__ = ('cg_iterations', 'cg_residual', 'cg_converged', 'cg_modes_dropped')


def _shape_derivatives(h):
    """dN_n/dx_d at each Gauss point, shape (8 gauss, 8 nodes, 3 dirs), plus the same at the
    element centre.

    On the reference cube [-1,1]^3, N_n = prod_d (1 + s_nd xi_d)/2, so
    dN_n/dxi_d = (s_nd/2) prod_{d' != d} (1 + s_nd' xi_d')/2, and dx_d = (h/2) dxi_d.
    """
    def derivs(xi):
        # xi: (npts, 3). Returns (npts, 8, 3).
        fac = (1.0 + SIGNS[None, :, :] * xi[:, None, :]) / 2.0      # (npts, 8, 3)
        out = np.empty((xi.shape[0], 8, 3))
        for d in range(3):
            other = [dd for dd in range(3) if dd != d]
            out[:, :, d] = (SIGNS[None, :, d] / 2.0) * fac[:, :, other[0]] * fac[:, :, other[1]]
        return out * (2.0 / h)

    return derivs(_GAUSS), derivs(np.zeros((1, 3)))[0]


def _cubic_tensor(C11, C12, C44):
    """C_ijkl for cubic symmetry, as a (3,3,3,3) array.

    C_ijkl = C12 d_ij d_kl + C44 (d_ik d_jl + d_il d_jk) + (C11 - C12 - 2 C44) [i=j=k=l]
    which gives C_1111 = C11, C_1122 = C12, C_1212 = C44 as required.
    """
    d = np.eye(3)
    C = (C12 * np.einsum('ij,kl->ijkl', d, d)
         + C44 * (np.einsum('ik,jl->ijkl', d, d) + np.einsum('il,jk->ijkl', d, d)))
    aniso = C11 - C12 - 2.0 * C44
    for i in range(3):
        C[i, i, i, i] += aniso
    return C


def _element_arrays(C11, C12, C44, h):
    """(K, W) for one element: the 8x8x3x3 stiffness coefficients and the 8x3 eigenstrain
    load shape.

    With grad_ij = d_i u_j = sum_n G[n,i] U[n,j], the energy (1/2) C_ijkl grad_ij grad_kl gives

        K[a,b,j,l] = sum_gauss w detJ  C_ijkl G[g,a,i] G[g,b,k]
        W[a,j]     = sum_gauss w detJ  G[g,a,j]           (so the eigenstrain force is
                                                           eps_T (C11 + 2 C12) W)

    using C_ijkk = (C11 + 2 C12) d_ij for the dilatational eigenstrain.
    """
    G, _ = _shape_derivatives(h)
    C = _cubic_tensor(C11, C12, C44)
    wdet = (h / 2.0) ** 3                       # 2x2x2 Gauss weights are all 1
    K = wdet * np.einsum('ijkl,gai,gbk->abjl', C, G, G)
    W = wdet * G.sum(axis=0)
    return K, W


def _roll(a, shift, axes=(-3, -2, -1)):
    return np.roll(a, shift, axis=axes)


def _symbol(K, n):
    """The homogeneous operator's Fourier symbol S(k), shape (3, 3, Nx, Ny, Nz).

    S(k)_jl = sum_{a,b} K[a,b,j,l] exp(i k.(b - a)), the exact Fourier transform of the
    constant-coefficient stencil. Hermitian and positive semi-definite for a stable material.
    """
    kx = 2 * np.pi * np.fft.fftfreq(n[0])
    ky = 2 * np.pi * np.fft.fftfreq(n[1])
    kz = 2 * np.pi * np.fft.fftfreq(n[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    S = np.zeros((3, 3) + tuple(n), dtype=complex)
    for a in range(8):
        for b in range(8):
            d = CORNERS[b] - CORNERS[a]
            ph = np.exp(1j * (d[0] * KX + d[1] * KY + d[2] * KZ))
            for j in range(3):
                for l in range(3):
                    if K[a, b, j, l] != 0.0:
                        S[j, l] += K[a, b, j, l] * ph
    return S


def _nonsingular(M, cond_tol=1e-9, scale_tol=1e-12):
    """Which wavevectors of a Hermitian PSD symbol may be inverted. M is (..., 3, 3).

    Only k = 0 is genuinely singular -- a rigid translation costs no energy, which is the
    statement sum_{a,b} K[a,b] = 0, so S(0) = 0 exactly. The test for that has to be SCALE FREE,
    and this is where an earlier version of this function was wrong in a way that only showed up
    on large grids.

    Because S vanishes as |k|^2 at small k, det S ~ |k|^6. On a grid of N points per side the
    smallest nonzero wavevector is 2 pi / N, so

        det_min / det_max  ~  (2 / N)^6

    which is 5e-10 at N = 70 but 4e-12 at N = 158. Comparing det against a fixed fraction of its
    own global maximum -- the old `|det| > 1e-10 * |det|.max()` -- therefore starts discarding
    long-wavelength modes once the grid passes N ~ 100, and discards a wider shell of them the
    finer the grid gets. Those modes are perfectly well conditioned; they are merely small, as
    the physics says they must be. Zeroing them turns the preconditioner into a low-pass filter,
    and CG cannot restore what the preconditioner never lets into the Krylov space. The result is
    a WRONG STRAIN FIELD whose error grows with resolution, reported with a converged residual.

    The scale-free measure is det against (tr/3)^3: both scale as the cube of the block's own
    magnitude, so their ratio is O(1) for any well-conditioned block however small it is, and the
    test becomes a statement about conditioning rather than about magnitude. k = 0 is excluded
    separately, on magnitude, which is the property it actually lacks.
    """
    tr = np.trace(M, axis1=-2, axis2=-1).real / 3.0
    det = np.abs(np.linalg.det(M))
    scale = np.maximum(tr, 0.0)
    return (tr > scale_tol * tr.max()) & (det > cond_tol * scale ** 3)


class _Operator:
    """Matrix-free action of the assembled stiffness, and its homogeneous-symbol preconditioner.

    Node p receives contributions from every element that contains it. Element i contributes to
    node i + a for each corner a, so

        F[p]_j = sum_{a,b} K_{mat(p-a)}[a,b,j,l] u[p + b - a]_l .

    Two materials means K_{mat(i)} = K_matrix + (K_dot - K_matrix) chi(i), so each corner pair
    needs one multiply against the shifted displacement and one against chi times it.
    """

    def __init__(self, chi, K_dot, K_mat, h):
        self.chi = chi
        self.K_m = K_mat
        self.dK = K_dot - K_mat
        self.shape = chi.shape
        self.h = h
        # Average material for the preconditioner symbol: the contrast is what CG must resolve,
        # so preconditioning with the volume-averaged operator leaves it the smallest job.
        f = float(chi.mean())
        self._symbol_inverse(K_mat + f * self.dK)

    def _symbol_inverse(self, K):
        S = _symbol(K, self.shape)
        M = np.moveaxis(S, [0, 1], [-2, -1])                  # (Nx, Ny, Nz, 3, 3), Hermitian PSD
        good = _nonsingular(M)
        inv = np.zeros_like(M)
        inv[good] = np.linalg.inv(M[good])
        self.Sinv = np.moveaxis(inv, [-2, -1], [0, 1])
        self.modes_dropped = int(good.size - good.sum())

    def apply(self, U):
        """U, and the result, have shape (3, Nx, Ny, Nz) on nodes.

        Looped with the source corner `b` outermost so each shifted copy of U (and its
        chi-weighted partner) is formed once and consumed by all eight destination corners.
        The alternative order needs all eight shifted copies resident at once, which doubles
        the peak footprint on the large grids this is used at.
        """
        Fa = np.zeros((8,) + U.shape, dtype=U.dtype)
        for b in range(8):
            Ub = _roll(U, tuple(-CORNERS[b]))
            Ubc = self.chi[None] * Ub
            for a in range(8):
                for j in range(3):
                    for l in range(3):
                        km, dk = self.K_m[a, b, j, l], self.dK[a, b, j, l]
                        if km:
                            Fa[a, j] += km * Ub[l]
                        if dk:
                            Fa[a, j] += dk * Ubc[l]

        out = np.zeros_like(U)
        for a in range(8):
            out += _roll(Fa[a], tuple(CORNERS[a]))
        return out

    def precondition(self, R):
        Rk = np.fft.fftn(R, axes=(-3, -2, -1))
        Uk = np.einsum('jl...,l...->j...', self.Sinv, Rk)
        return np.fft.ifftn(Uk, axes=(-3, -2, -1)).real


def solve_strain_fd(inside_mask, eps_star, C_dot, C_matrix, h, tol=1e-10, maxiter=2000,
                    verbose=False, return_total=False, require_converged=True):
    """Elastic strain tensor for an inclusion with its OWN elastic constants.

    Drop-in counterpart to strain_fourier.solve_strain, with the same sign conventions and the
    same return type, but taking two sets of elastic constants instead of one.

    Parameters
    ----------
    inside_mask : bool array (Nx, Ny, Nz)
        The dot's characteristic function, interpreted on ELEMENT centres -- i.e. on exactly
        the grid the caller is already using for everything else.
    eps_star : float
        qdsolver_core.eigenstrain(a_dot, a_matrix); negative for a compressed dot.
    C_dot, C_matrix : (C11, C12, C44)
        Cubic elastic constants of inclusion and matrix, in any consistent unit.
    h : float
        Grid spacing (nm).
    require_converged : bool
        Raise if the CG did not reach `tol`, or if the preconditioner had to drop any wavevector
        beyond k = 0. Default True, because an unconverged or filtered strain field is not a
        slightly worse answer -- it is a wrong one, and it propagates silently into every band
        edge computed from it. Pass False only to inspect a failure.

    Returns
    -------
    StrainTensor (elastic), or (elastic, total) if return_total. Same shape as inside_mask.
    Carries cg_iterations, cg_residual, cg_converged, cg_modes_dropped.

    Raises
    ------
    RuntimeError
        If require_converged and the solve did not converge cleanly.
    """
    chi = np.ascontiguousarray(inside_mask, dtype=float)
    eps_T = -float(eps_star)

    K_d, W_d = _element_arrays(*C_dot, h)
    K_m, _ = _element_arrays(*C_matrix, h)
    op = _Operator(chi, K_d, K_m, h)

    # Load vector: only dot elements carry an eigenstrain, and they carry it with their own
    # bulk modulus. Element i pushes on node i + a.
    bulk = eps_T * (C_dot[0] + 2 * C_dot[1]) * chi
    F = np.zeros((3,) + chi.shape)
    for a in range(8):
        for j in range(3):
            if W_d[a, j]:
                F[j] += W_d[a, j] * _roll(bulk, tuple(CORNERS[a]))

    U, iters, rel, breakdown = _pcg(op, F, tol=tol, maxiter=maxiter, verbose=verbose)
    converged = (rel < tol) and not breakdown and op.modes_dropped <= 1

    if require_converged and not converged:
        why = []
        if breakdown:
            why.append(f"CG broke down at iteration {iters} ({breakdown})")
        if rel >= tol:
            why.append(f"residual {rel:.2e} did not reach tol {tol:.0e} in {iters} iterations")
        if op.modes_dropped > 1:
            why.append(f"preconditioner dropped {op.modes_dropped} wavevectors (only k = 0, "
                       f"i.e. 1, is legitimate) -- it is filtering the solution")
        raise RuntimeError(
            f"strain solve did not converge on grid {chi.shape} ({chi.size:,} sites): "
            + "; ".join(why)
            + ". The strain field is wrong, not merely imprecise. Pass require_converged=False "
              "to inspect it.")

    # Strain at element centres from the trilinear gradient at xi = 0.
    _, G0 = _shape_derivatives(h)
    grad = np.zeros((3, 3) + chi.shape)          # grad[i, j] = d_i u_j
    for n in range(8):
        Un = _roll(U, tuple(-CORNERS[n]))
        for i in range(3):
            if G0[n, i]:
                grad[i] += G0[n, i] * Un

    exx, eyy, ezz = grad[0, 0], grad[1, 1], grad[2, 2]
    exy = 0.5 * (grad[0, 1] + grad[1, 0])
    eyz = 0.5 * (grad[1, 2] + grad[2, 1])
    ezx = 0.5 * (grad[2, 0] + grad[0, 2])
    total = StrainTensor(exx, eyy, ezz, exy, eyz, ezx)

    off = eps_T * chi
    elastic = FDStrainTensor(exx - off, eyy - off, ezz - off, exy, eyz, ezx)
    elastic.cg_iterations, elastic.cg_residual = iters, rel
    elastic.cg_converged, elastic.cg_modes_dropped = converged, op.modes_dropped

    return (elastic, total) if return_total else elastic


def _pcg(op, F, tol, maxiter, verbose=False):
    """Preconditioned conjugate gradients on the (singular, consistent) stiffness system.

    The operator is positive semi-definite with a three-dimensional null space of rigid
    translations. The load has zero net force, so the system is consistent and CG converges to
    the solution orthogonal to that null space provided the iterates are kept orthogonal to it
    -- which is what removing the mean of each component does, and what the preconditioner's
    zeroed k = 0 block does automatically.

    Returns (U, iterations, relative residual, breakdown reason or None). CG on a semi-definite
    system with an imperfect preconditioner can break down rather than converge slowly, and the
    two need telling apart: a breakdown leaves U holding whatever the last step put there, which
    can be arbitrarily large. Both curvature quantities are checked, since either going
    non-positive means the iteration has lost a property it relies on.
    """
    def project(V):
        return V - V.mean(axis=(-3, -2, -1), keepdims=True)

    F = project(F)
    U = np.zeros_like(F)
    R = F.copy()
    Z = op.precondition(R)
    P = Z.copy()
    rz = float(np.vdot(R, Z).real)
    f_norm = float(np.linalg.norm(F))
    if f_norm == 0.0:
        return U, 0, 0.0, None
    if rz <= 0.0:
        return project(U), 0, 1.0, f"preconditioner is not positive definite (r.Mr = {rz:.3e})"

    rel, breakdown = 1.0, None
    for it in range(1, maxiter + 1):
        AP = op.apply(P)
        pap = float(np.vdot(P, AP).real)
        if pap <= 0:
            breakdown = f"non-positive curvature p.Ap = {pap:.3e}"
            break
        alpha = rz / pap
        U += alpha * P
        R -= alpha * AP
        rel = float(np.linalg.norm(R)) / f_norm
        if verbose and (it % 10 == 0 or rel < tol):
            print(f"    CG {it:4d}  ||r||/||f|| = {rel:.3e}")
        if rel < tol:
            break
        Z = op.precondition(R)
        rz_new = float(np.vdot(R, Z).real)
        if rz_new <= 0.0:
            breakdown = f"preconditioner lost positive definiteness (r.Mr = {rz_new:.3e})"
            break
        P = Z + (rz_new / rz) * P
        rz = rz_new

    return project(U), it, rel, breakdown


def strain_energy(strain, inside_mask, C_dot, C_matrix, h):
    """Total elastic strain energy of a strain field, for checking that the minimizer really
    lowers it relative to any competitor (the variational statement the method rests on)."""
    chi = np.asarray(inside_mask, dtype=float)
    e = strain
    out = 0.0
    for C, w in ((C_dot, chi), (C_matrix, 1.0 - chi)):
        C11, C12, C44 = C
        out += np.sum(w * (0.5 * C11 * (e.exx**2 + e.eyy**2 + e.ezz**2)
                           + C12 * (e.exx * e.eyy + e.eyy * e.ezz + e.ezz * e.exx)
                           + 2.0 * C44 * (e.exy**2 + e.eyz**2 + e.ezx**2)))
    return float(out * h ** 3)


def inhomogeneous_sphere_trace(eps_star, K_i, mu_m):
    """Exact elastic dilatation inside a misfitting ISOTROPIC sphere whose moduli differ from
    the matrix: Tr(eps) = -3 eps_T * 4 mu_m / (3 K_i + 4 mu_m), with eps_T = -eps_star.

    Only the inclusion's bulk modulus and the matrix's shear modulus appear -- which is exactly
    why a homogeneous solver cannot reproduce it for any single choice of constants, and why
    this is the sharpest available test of the inhomogeneous machinery.

    Reduces to strain_fourier.isotropic_sphere_trace when K_i is the matrix's own bulk modulus.
    Reference: J. D. Eshelby, Proc. R. Soc. London A 241, 376 (1957) for the inclusion and
    inhomogeneity problems; this dilatational special case is the standard misfitting-sphere
    result.
    """
    eps_T = -eps_star
    return -3.0 * eps_T * 4.0 * mu_m / (3.0 * K_i + 4.0 * mu_m)


def voigt_moduli(C11, C12, C44):
    """(K, mu) Voigt averages, matching qdsolver_core.voigt_poisson_ratio's convention."""
    return (C11 + 2 * C12) / 3.0, (C11 - C12 + 3 * C44) / 5.0
