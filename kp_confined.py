"""Discretized (confined) 6-band Luttinger-Kohn Hamiltonian on a 3D grid.

Builds on kp_luttinger.py (validated against bulk dispersion/effective masses/split-off with
k as a number) by replacing k -> operator. Two different discretizations are used deliberately:

- Diagonal terms (k_x^2, k_y^2, k_z^2, feeding P and Q): the direct 3-point Ben Daniel-Duke
  stencil, the same one already validated in qdsolver_core.build_hamiltonian. NOT built by
  squaring a first-derivative matrix -- that construction only couples grid points two apart
  (i to i+/-2, skipping i+/-1 entirely) and has a well-known spurious "checkerboard" null space
  (the alternating +1,-1,+1,-1,... mode is invisible to a central-difference first derivative).
- Cross terms (k_x k_y etc., feeding R and S): built from Hermitian central-difference
  first-derivative operators D_x, D_y, D_z, symmetrized as (D_i diag(alpha) D_j + D_j diag(alpha)
  D_i)/2 -- this is Hermitian by construction whenever D_i, D_j, and diag(alpha) are each
  Hermitian, regardless of whether the alpha(r) formulas themselves are correct (that's checked
  separately, the same "Hermiticity from construction, physics from validation" split used
  throughout this project).

Known caveat -- operator ordering: with position-dependent Luttinger parameters there is a
genuine physical ambiguity in how gamma(r) is ordered against the derivatives (gamma k_i k_j vs
k_i gamma k_j vs the symmetrized average). This module uses plain symmetrization, which
guarantees Hermiticity but is NOT the rigorously derived envelope-function ordering:
 - M. G. Burt, J. Phys.: Condens. Matter 4, 6651 (1992): exact envelope-function theory showing
   the correct ordering is generally asymmetric.
 - B. A. Foreman, Phys. Rev. B 48, 4964 (1993): the resulting ("Burt-Foreman") ordering for the
   valence-band Hamiltonian, and why naive symmetrization can produce spurious solutions.
Finite-difference multiband k.p with symmetrized ordering is documented to be prone to spurious
high-k solutions creeping into the low end of the spectrum in some circumstances. The validation
notebook checks for this (via the periodic bulk-dispersion comparison and a Hermiticity check)
but it's a known, general limitation of this class of method, not fully eliminated here.

(Per the citation policy in qdsolver_core.py, equation numbers are not quoted; correctness here
rests on the machine-precision periodic bulk-dispersion cross-check rather than transcription.)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qdsolver_core import HBAR2_OVER_2M0


def first_derivative_operator(shape, axis, h, periodic=False):
    """Hermitian matrix for k_axis = -i d/d(axis), central difference (1/nm).

    periodic=True wraps the last plane back to the first -- only used for the bulk-dispersion
    validation (a finite QD box is never periodic; that case uses periodic=False, i.e. hard
    walls, matching the diagonal-term boundary treatment).
    """
    Nx, Ny, Nz = shape
    idx = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)
    n_ax = shape[axis]

    idx_here = np.take(idx, np.arange(0, n_ax - 1), axis=axis)
    idx_next = np.take(idx, np.arange(1, n_ax), axis=axis)
    coeff = -1j / (2 * h)
    rows = [idx_here.ravel(), idx_next.ravel()]
    cols = [idx_next.ravel(), idx_here.ravel()]
    data = [np.full(idx_here.size, coeff), np.full(idx_here.size, -coeff)]

    if periodic:
        idx_first = np.take(idx, [0], axis=axis)
        idx_last = np.take(idx, [n_ax - 1], axis=axis)
        rows += [idx_last.ravel(), idx_first.ravel()]
        cols += [idx_first.ravel(), idx_last.ravel()]
        data += [np.full(idx_first.size, coeff), np.full(idx_first.size, -coeff)]

    n = Nx * Ny * Nz
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex).tocsr()


def diagonal_k2_operator(alpha_field, h, periodic=False):
    """d/daxis(alpha(r) d/daxis), the direct 3-point stencil, for a single Cartesian axis
    (unlike qdsolver_core.build_hamiltonian, which sums all three axes with one shared
    coefficient -- P and Q need different linear combinations of the three separately).

    periodic=True wraps the last plane back to the first, matching first_derivative_operator's
    periodic flag -- used only for the bulk-dispersion validation. Getting this flag wrong
    silently injects a spurious energy offset even for a uniform (k=0) state: with periodic=True
    intended but this left hard-wall, a "ghost wall" diagonal term still gets added at the two
    boundary faces, which is exactly the bug this flag was added to fix.
    """
    def _single_axis(axis):
        shape = alpha_field.shape
        Nx, Ny, Nz = shape
        idx = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)
        diag = np.zeros(shape)
        rows, cols, data = [], [], []

        a_next = np.take(alpha_field, np.arange(1, shape[axis]), axis=axis)
        a_here = np.take(alpha_field, np.arange(0, shape[axis] - 1), axis=axis)
        a_half = 0.5 * (a_here + a_next)
        t = a_half / h**2

        idx_here = np.take(idx, np.arange(0, shape[axis] - 1), axis=axis)
        idx_next = np.take(idx, np.arange(1, shape[axis]), axis=axis)
        rows.append(idx_here.ravel()); cols.append(idx_next.ravel()); data.append(-t.ravel())
        rows.append(idx_next.ravel()); cols.append(idx_here.ravel()); data.append(-t.ravel())

        slicer_here = [slice(None)] * 3; slicer_here[axis] = slice(0, -1)
        slicer_next = [slice(None)] * 3; slicer_next[axis] = slice(1, None)
        diag[tuple(slicer_here)] += t
        diag[tuple(slicer_next)] += t

        if periodic:
            a_first = np.take(alpha_field, 0, axis=axis)
            a_last = np.take(alpha_field, shape[axis] - 1, axis=axis)
            t_wrap = 0.5 * (a_first + a_last) / h**2
            idx_first = np.take(idx, [0], axis=axis)
            idx_last = np.take(idx, [shape[axis] - 1], axis=axis)
            rows.append(idx_first.ravel()); cols.append(idx_last.ravel()); data.append(-t_wrap.ravel())
            rows.append(idx_last.ravel()); cols.append(idx_first.ravel()); data.append(-t_wrap.ravel())
            lo = [slice(None)] * 3; lo[axis] = 0
            hi = [slice(None)] * 3; hi[axis] = -1
            diag[tuple(lo)] += t_wrap
            diag[tuple(hi)] += t_wrap
        else:
            lo = [slice(None)] * 3; lo[axis] = 0
            hi = [slice(None)] * 3; hi[axis] = -1
            diag[tuple(lo)] += alpha_field[tuple(lo)] / h**2
            diag[tuple(hi)] += alpha_field[tuple(hi)] / h**2

        rows.append(idx.ravel()); cols.append(idx.ravel()); data.append(diag.ravel())
        rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
        n = Nx * Ny * Nz
        return sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex).tocsr()
    return _single_axis


def cross_k2_operator(alpha_field, Di, Dj):
    """Symmetrized Hermitian representation of alpha(r)*k_i*k_j for i != j:
    (D_i @ diag(alpha) @ D_j + D_j @ diag(alpha) @ D_i) / 2.

    Note this is the *symmetrized* ordering choice, not the Burt-Foreman envelope-function
    ordering (see the module docstring's caveat and the Burt 1992 / Foreman 1993 references):
    the two differ only where alpha(r) varies, i.e. at the dot/matrix interface."""
    A = sp.diags(alpha_field.ravel())
    return (Di @ A @ Dj + Dj @ A @ Di) / 2.0


class GridOperators:
    """Bundles the Hermitian first-derivative and diagonal-k2 operators for a grid shape, so
    they're built once and reused across P/Q/R/S for a given (shape, h)."""

    def __init__(self, shape, h, periodic=False):
        self.shape = shape
        self.h = h
        self.periodic = periodic
        self.Dx = first_derivative_operator(shape, 0, h, periodic)
        self.Dy = first_derivative_operator(shape, 1, h, periodic)
        self.Dz = first_derivative_operator(shape, 2, h, periodic)

    def k2(self, alpha_field, axis):
        return diagonal_k2_operator(alpha_field, self.h, self.periodic)(axis)

    def cross(self, alpha_field, axis_i, axis_j):
        D = {0: self.Dx, 1: self.Dy, 2: self.Dz}
        return cross_k2_operator(alpha_field, D[axis_i], D[axis_j])


def build_confined_luttinger_kohn(ops, gamma1_field, gamma2_field, gamma3_field,
                                   delta_so_field, V_field):
    """Assemble the confined 6x6-block sparse Hamiltonian (eV) from position-dependent
    Luttinger parameters, split-off energy, and a common (band-edge + strain) potential V_field
    added identically to every band -- valid as-is only when strain is purely hydrostatic (true
    for the spherical dot; a non-spherical shape's shear strain would need genuine Bir-Pikus
    Q/R/S-like terms added here too, not yet implemented).
    """
    g = HBAR2_OVER_2M0
    kxx = ops.k2(gamma1_field, 0); kyy = ops.k2(gamma1_field, 1); kzz = ops.k2(gamma1_field, 2)
    P = g * (kxx + kyy + kzz)

    kxx2 = ops.k2(gamma2_field, 0); kyy2 = ops.k2(gamma2_field, 1); kzz2 = ops.k2(gamma2_field, 2)
    Q = g * (kxx2 + kyy2 - 2 * kzz2)

    kxx_g2 = ops.k2(gamma2_field, 0); kyy_g2 = ops.k2(gamma2_field, 1)
    kxky_g3 = ops.cross(gamma3_field, 0, 1)
    R = g * (-np.sqrt(3) * (kxx_g2 - kyy_g2) + 1j * 2 * np.sqrt(3) * kxky_g3)

    kxkz_g3 = ops.cross(gamma3_field, 0, 2)
    kykz_g3 = ops.cross(gamma3_field, 1, 2)
    S = g * 2 * np.sqrt(3) * (kxkz_g3 - 1j * kykz_g3)

    n = P.shape[0]
    Vdiag = sp.diags(V_field.ravel())
    Dso = sp.diags(delta_so_field.ravel())

    Sc = S.conj().T  # S* as an operator (conjugate transpose, since S itself isn't Hermitian)
    zero = sp.csr_matrix((n, n), dtype=complex)

    blocks = [[None]*6 for _ in range(6)]
    blocks[0][0] = P + Q + Vdiag
    blocks[1][1] = P - Q + Vdiag
    blocks[2][2] = P - Q + Vdiag
    blocks[3][3] = P + Q + Vdiag
    blocks[4][4] = P + Vdiag - Dso
    blocks[5][5] = P + Vdiag - Dso

    blocks[0][1] = -S
    blocks[0][2] = R
    blocks[0][3] = zero
    blocks[0][4] = -Sc / np.sqrt(2)
    blocks[0][5] = np.sqrt(2) * R

    blocks[1][2] = zero
    blocks[1][3] = R
    blocks[1][4] = -np.sqrt(2) * Q
    blocks[1][5] = np.sqrt(1.5) * S

    blocks[2][3] = S
    blocks[2][4] = np.sqrt(1.5) * Sc
    blocks[2][5] = np.sqrt(2) * Q

    blocks[3][4] = -np.sqrt(2) * R
    blocks[3][5] = S / np.sqrt(2)

    blocks[4][5] = zero

    for i in range(6):
        for j in range(i):
            blocks[i][j] = blocks[j][i].conj().T

    return sp.bmat(blocks, format='csr')


def safe_sigma_hole(V_field, delta_so_field, buffer=0.1):
    """A conservative (safely-below-the-true-ground-state) shift-invert target for the hole
    convention (V_field already negated, i.e. this is what was passed to
    build_confined_luttinger_kohn). Using sigma=-V_h.max() (a fine guess for a *single* band
    with a light kinetic correction) silently converged eigsh to the wrong -- not lowest --
    eigenvalues once the split-off offset (up to ~0.4 eV) and multiband coupling were in play;
    caught by comparing a decoupled-limit test against a known single-band answer and finding
    the assembled matrix was correct but the eigsh target wasn't."""
    return V_field.min() - delta_so_field.max() - buffer


def solve_confined_states(H, n_states=4, sigma=None):
    n = H.shape[0]
    if sigma is None:
        sigma = 0.0
    shifted = (H - sigma * sp.identity(n, dtype=complex)).tocsc()
    lu = spla.splu(shifted, permc_spec='MMD_AT_PLUS_A')
    OPinv = spla.LinearOperator((n, n), matvec=lu.solve, dtype=complex)
    E, psi = spla.eigsh(H, k=n_states, sigma=sigma, which='LM', OPinv=OPinv)
    order = np.argsort(E)
    return E[order], psi[:, order]
