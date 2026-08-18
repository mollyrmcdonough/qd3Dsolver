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

THE OPERATOR THIS MODULE DISCRETIZES IS UNBOUNDED AT AN ABRUPT INTERFACE
------------------------------------------------------------------------
Not a caveat, and not a property of the discretization. With position-dependent Luttinger
parameters there is a genuine ambiguity in how gamma(r) is ordered against the derivatives
(gamma k_i k_j vs k_i gamma k_j vs the symmetrized average). This module uses plain
symmetrization, which guarantees Hermiticity -- and which, at a step in gamma, gives an operator
with no maximum at all. The interface modes everything here has struggled with are that
unboundedness, faithfully reproduced.

The reason is that the six-band kinetic tensor is Legendre-Hadamard elliptic (every bulk band
curves downward, gamma1 - 2 gamma2 > 0 and gamma1 - 2 gamma3 > 0) but NOT strongly elliptic: as
an 18x18 matrix over (direction, band) it has eigenvalues up to +1.74 for InSb. Constant or
continuous coefficients only ever expose the rank-one directions, where it is negative; a
discontinuity exposes the rest. See `kp_planewave.strong_ellipticity` for the derivation and
`scripts/planewave_validation.py --only 3` for the controlled measurement, in which restoring
strong ellipticity -- and nothing else -- turns a spectrum that quadruples with basis size into
one flat to 0.1 meV.

So no choice of stencil in this module can help; three were tried and none did. The fix is
Burt-Foreman ORDERING, a different operator:
 - M. G. Burt, J. Phys.: Condens. Matter 4, 6651 (1992): exact envelope-function theory showing
   the correct ordering is generally asymmetric.
 - B. A. Foreman, Phys. Rev. B 48, 4964 (1993): the resulting ("Burt-Foreman") ordering for the
   valence-band Hamiltonian, and why naive symmetrization can produce spurious solutions.

Until that lands, results from this module are trustworthy only where the runaway is small
against the well depth -- large islands and coarse grids, in that order. `heterostructure.
six_band_holes` filters eigenvalues above the island valence edge, which removes the worst of it
but cannot remove the contamination of what lies below.

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
    the two differ only where alpha(r) varies, i.e. at the dot/matrix interface.

    THE OPERATOR IS BROKEN AT INTERFACES, NOT THIS DISCRETIZATION OF IT. That correction was
    established by `kp_planewave`, and it reverses what earlier versions of this docstring said.

    Measured on a 2.5 nm InSb island in InAs with no strain: the largest eigenvalue climbs ABOVE
    the dot valence edge as the grid is refined -- +0.295 eV at h = 0.50 nm, +0.519 at 0.312,
    +0.691 at 0.208, +0.860 at 0.156, against a dot edge of +0.590 -- while the identical test
    with NO interface stays clean at every spacing (-0.279 to -0.237, always below the edge). The
    artifact becomes MORE island-localized as h shrinks (42% -> 70%), so no localization threshold
    separates it from a real bound state, and it survives grading the interface over a monolayer.

    The natural reading of that was a stencil inconsistency: D_i and D_j here are CENTRAL
    differences, reaching two grid points per direction and sampling alpha at NODES, whereas
    `diagonal_k2_operator` uses a compact 3-point stencil and samples alpha averaged at cell
    FACES, so where alpha jumps the two families see the interface in different places. That
    reading is WRONG, and two rewrites built on it changed nothing (`cross_k2_operator_staggered`,
    `cross_k2_operator_fe`).

    What is actually happening: the symmetrized ordering k_i alpha k_j gives an operator that is
    Legendre-Hadamard elliptic but NOT strongly elliptic, and at a discontinuous interface such an
    operator has no maximum at all. A plane-wave basis, which has no stencil and no cell, produces
    the same runaway -- 0.457 -> 1.951 eV as the basis grows at fixed grid -- and scaling gamma2,
    gamma3 down until strong ellipticity is restored makes it vanish, flat to 0.1 meV, with
    nothing else changed. See `kp_planewave.strong_ellipticity` and
    `scripts/planewave_validation.py --only 3`.

    So this stencil is not the problem and replacing it cannot be the fix; the fix is Burt-Foreman
    ORDERING, which is a different operator. Kept as the default because it is a correct
    discretization of the operator the rest of this package assumes."""
    A = sp.diags(alpha_field.ravel())
    return (Di @ A @ Dj + Dj @ A @ Di) / 2.0


#: Q1 element factors on a uniform cell, per axis, for nodes at local offsets 0 and 1.
#: A[p,q] = integral of phi'_p phi_q  (dimensionless);  M[p,q] = integral of phi_p phi_q  (units h)
_Q1_A = np.array([[-0.5, -0.5], [0.5, 0.5]])
_Q1_M = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0          # times h
_Q1_M_LUMPED = np.array([[1.0, 0.0], [0.0, 1.0]]) / 2.0   # times h


def cross_k2_operator_fe(alpha_field, h, axis_i, axis_j, periodic=False, lump=False):
    """k_i alpha k_j for i != j, as a Q1 trilinear finite element. A REJECTED FIX, kept as a record.

    Why it was written
    ------------------
    `cross_k2_operator` samples alpha at NODES over a two-cell-wide central stencil while
    `diagonal_k2_operator` samples it averaged at cell FACES over a compact one. Where alpha jumps
    the two disagree about where the material changes, and that WAS BELIEVED to be what makes
    small dots unreachable. It is not -- see the end of this docstring. Measured on Yeap's 2.5 nm
    InSb/InAs geometry with
    `scripts/gamma3_diagnostic.py`, confinement over h = 0.50 / 0.40 / 0.32 nm:

        all fields stepped          429.4 -> 311.3 -> 225.8 meV   (-118, -85: never settles)
        gamma3 held uniform         601.0 -> 562.8 -> 564.0 meV   (-38, +1: converged)
        gamma1,2,3 uniform (InSb)   616.4 -> 565.5 -> 558.5 meV   (-51, -7: converged)
        gamma1,2,3 uniform (InAs)   557.2 -> 509.6 -> 505.5 meV   (-48, -4: converged)

    `ops.cross` is called with gamma3 and nothing else, so the one diverging case is exactly the
    one with a step in the cross-term coefficient. The third row is the control that matters: it
    carries InSb's LARGE gamma3 = 16.5 everywhere, bigger than the stepped case has anywhere, and
    converges -- so it is the discontinuity, not the magnitude.

    Here alpha is averaged over each cell's eight corner nodes and the element is the same Q1
    trilinear hexahedron `elasticity_fd` already validates, so both operator families see the
    interface in the same place.

    The symbol, which is the check the previous attempt failed
    ---------------------------------------------------------
    For uniform alpha, with (a, b, c) = k*h along (i, j, the third axis):

        consistent:  alpha * sin(a) sin(b) * (2 + cos c) / (3 h^2)
        lumped:      alpha * sin(a) sin(b) / h^2          -- identical to central differences

    Both carry the sin(a)sin(b) factor, so both change sign with k_i k_j as a mixed derivative
    must, and both reduce to alpha*k_i*k_j as h -> 0. Contrast `cross_k2_operator_staggered`,
    whose symbol [cos(a-b) - cos(a) - cos(b) + 1]/h^2 is NON-NEGATIVE everywhere and therefore
    cannot represent a mixed derivative at all -- it reaches +4/h^2 at (pi, -pi) where the true
    k_i k_j is negative. That is why it failed validation, and why the symbol is checked on paper
    before the operator is trusted.

    `lump=True` replaces the transverse consistent mass (1,4,1)/6 with (1,0,1)->diag(1/2), making
    the bulk symbol EXACTLY central's. That isolates the alpha-sampling change from the mass
    smearing, so a difference between the two is attributable to sampling alone.

    **Bulk agreement is therefore automatic and proves nothing.** It is worth running only to
    catch assembly bugs. The validation that counts is interface h-convergence.

    IT DOES NOT FIX THE ARTIFACT -- measured, and kept as a record
    --------------------------------------------------------------
    Case A rerun with `scheme='fe'`, same geometry and spacings:

        central     429.4 -> 311.3 -> 225.8 meV   (-118, -85)
        fe          507.2 -> 364.0 -> 270.8 meV   (-143, -93)   slightly WORSE
        fe-lumped   461.6 -> 320.5 -> 233.8 meV   (-141, -87)   the same

    The operator itself is sound: Hermitian to machine zero, symbol matching the formula above to
    7e-15 across the whole Brillouin zone, and correctly sign-changing with k_i k_j. The
    *reasoning* was wrong, and the lumped row is what proves it. 'fe-lumped' has EXACTLY central's
    bulk symbol and differs from it only in where alpha is sampled -- cell average versus node.
    It drifts by the same amount. So alpha's sampling location across the step is not the lever,
    and neither is the mass smearing that separates 'fe' from 'fe-lumped'.

    **What is left is the operator ORDERING, which is a different thing from the sampling.** All
    three schemes here discretize the same symmetrized form (k_i a k_j + k_j a k_i)/2. Burt's
    exact envelope-function theory says that form is not the correct one when the parameters
    vary: the R and S blocks require specific asymmetric orderings of the gamma3 contributions
    (Burt, J. Phys.: Condens. Matter 4, 6651 (1992); Foreman, Phys. Rev. B 48, 4964 (1993)).
    Every scheme in this module symmetrizes, so none of them can be right at an abrupt interface
    no matter how alpha is sampled -- which is exactly what these three rows measure.

    CONFIRMED INDEPENDENTLY, and more sharply than these rows could
    ---------------------------------------------------------------
    `kp_planewave` solves the same symmetrized operator in a basis with no stencil, no cell and no
    sampling choice at all, and reproduces the runaway: at fixed grid, growing only the basis, the
    top of the spectrum goes 0.457 -> 1.951 eV with steps that do not shrink. Scaling gamma2 and
    gamma3 by 0.25 -- which takes lambda_max of the 18x18 kinetic tensor from +1.741 to -0.559,
    i.e. restores STRONG ellipticity, and changes nothing else -- makes the same ladder flat to
    0.1 meV. The symmetrized operator is Legendre-Hadamard elliptic but not strongly elliptic, and
    at a discontinuous interface such an operator simply has no maximum.

    So this scheme did not fail. It was aimed at a cause that was not there, and so was the
    staggered one, and so was the original sampling diagnosis. No cross-term stencil can fix this.
    """
    alpha = np.asarray(alpha_field, dtype=float)
    shape = alpha.shape
    Nx, Ny, Nz = shape
    axis_k = 3 - axis_i - axis_j            # the axis that is neither i nor j
    M = _Q1_M_LUMPED if lump else _Q1_M

    idx = np.arange(Nx * Ny * Nz).reshape(shape)
    ncell = [n if periodic else n - 1 for n in shape]
    if min(ncell) < 1:
        raise ValueError(f"grid {shape} too small for a Q1 cell")

    def gather(offset):
        """Node index / field value at local corner `offset` of every cell."""
        sl = tuple(np.arange(ncell[d]) + offset[d] for d in range(3))
        take = lambda arr: arr[np.ix_(sl[0] % Nx, sl[1] % Ny, sl[2] % Nz)]
        return take(idx), take(alpha)

    corners = [(p, q, r) for p in (0, 1) for q in (0, 1) for r in (0, 1)]
    node_idx = {o: gather(o)[0] for o in corners}
    # alpha at the cell centre: the mean over the eight corners. This is the whole point --
    # one value per cell, shared by every term the cell contributes, so the cross and diagonal
    # families cannot disagree about where the material changes.
    a_cell = sum(gather(o)[1] for o in corners) / 8.0

    rows, cols, data = [], [], []
    for oa in corners:
        for ob in corners:
            w = (_Q1_A[oa[axis_i], ob[axis_i]] * _Q1_A[ob[axis_j], oa[axis_j]]
                 * M[oa[axis_k], ob[axis_k]])
            if w == 0.0:
                continue
            # element integral carries one factor of h from the mass term; the lumped nodal
            # volume h^3 turns the bilinear form into an operator. Net 1/h^2.
            rows.append(node_idx[oa].ravel())
            cols.append(node_idx[ob].ravel())
            data.append((w / h ** 2) * a_cell.ravel())

    n = Nx * Ny * Nz
    K = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n), dtype=complex).tocsr()
    # (K + K^T)/2 IS the symmetrization (k_i a k_j + k_j a k_i)/2, because the (j,i) element
    # matrix is the transpose of the (i,j) one. Real symmetric, hence Hermitian.
    return (K + K.T) / 2.0


def forward_difference_operator(shape, axis, h, periodic=False):
    """Forward difference B_i: (f[i+1] - f[i]) / h. Real, and deliberately NOT Hermitian.

    B_i^dagger is the backward difference, so B_i^dagger @ diag(alpha) @ B_i is exactly the
    compact 3-point stencil with alpha averaged at cell faces -- i.e. precisely what
    `diagonal_k2_operator` builds. That is the point of having this: using the same B operators
    for the cross terms makes both families sample alpha at the same places on the same stencil,
    which is what a finite-element weak form does automatically and what the central-difference
    construction fails to do.

    The quadratic form is the discrete version of integrating by parts once,

        <psi| -d_i alpha d_j |psi>  =  integral (d_i psi)* alpha (d_j psi),

    so the matrix for k_i alpha k_j is B_i^dagger @ diag(alpha) @ B_j, carrying no factors of i
    -- matching `diagonal_k2_operator`, which likewise carries none. (k = -i d, and the two
    factors of -i multiply to -1, cancelling the minus sign in -d_i alpha d_j.)
    """
    Nx, Ny, Nz = shape
    n = Nx * Ny * Nz
    idx = np.arange(n).reshape(Nx, Ny, Nz)
    n_ax = shape[axis]

    idx_here = np.take(idx, np.arange(0, n_ax - 1), axis=axis)
    idx_next = np.take(idx, np.arange(1, n_ax), axis=axis)
    rows = [idx_here.ravel(), idx_here.ravel()]
    cols = [idx_next.ravel(), idx_here.ravel()]
    data = [np.full(idx_here.size, 1.0 / h), np.full(idx_here.size, -1.0 / h)]

    if periodic:
        idx_first = np.take(idx, [0], axis=axis)
        idx_last = np.take(idx, [n_ax - 1], axis=axis)
        rows += [idx_last.ravel(), idx_last.ravel()]
        cols += [idx_first.ravel(), idx_last.ravel()]
        data += [np.full(idx_first.size, 1.0 / h), np.full(idx_first.size, -1.0 / h)]

    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex).tocsr()


def cross_k2_operator_staggered(alpha_field, Bi, Bj):
    """alpha(r)*k_i*k_j for i != j, from forward differences: the Hermitian part of Bi^H alpha Bj.

    FAILED VALIDATION -- DO NOT USE. Kept, with this record, so the idea is not retried blindly.

    The intent was to remove the stencil/sampling inconsistency documented in
    `cross_k2_operator` by building the cross terms the same way as the diagonal ones. It is
    Hermitian to machine precision (0.00e+00) and correct to leading order in k, but it is wrong
    at large k, and two checks caught it:

      * BULK. On a UNIFORM material the two schemes must agree exactly, since they can only
        differ where alpha(r) varies. They do not: the top eigenvalue differs by 4.5e-2 eV at
        h = 0.50 nm, 2.3e-2 at 0.312, 9.5e-3 at 0.208. Disagreeing in the bulk means the
        operator does not reproduce the bulk dispersion, which is disqualifying on its own.
      * INTERFACE. Far worse than what it replaced -- top eigenvalue +4.03 eV at h = 0.50 nm,
        +11.47 at 0.312, +18.75 at 0.250, against a dot valence edge of +0.590.

    The reason is the symbol. With B^dagger having symbol (exp(-i a) - 1)/h and B symbol
    (exp(i b) - 1)/h, the Hermitian part of Bi^H alpha Bj has symbol

        [cos(a - b) - cos(a) - cos(b) + 1] / h^2,        a = k_i h,  b = k_j h

    which tends to k_i k_j as h -> 0 but is NON-NEGATIVE everywhere -- at a = pi, b = -pi it
    gives +4/h^2 where the true k_i k_j is negative. A mixed derivative must be able to change
    sign with the relative sign of k_i and k_j, and this cannot. The central-difference symbol
    sin(a) sin(b) / h^2 does change sign correctly, which is why the original scheme, whatever
    its interface trouble, is better behaved than this.

    So the cross-term problem is NOT fixed by naively reusing the diagonal construction. A
    correct staggered mixed derivative needs forward and backward differences paired across the
    two axes (and the corresponding alpha averaging), or the Burt-Foreman ordering proper.
    """
    A = sp.diags(alpha_field.ravel())
    M = Bi.getH() @ A @ Bj
    return (M + M.getH()) / 2.0


class GridOperators:
    """Bundles the first-derivative and diagonal-k2 operators for a grid shape, so they're built
    once and reused across P/Q/R/S for a given (shape, h).

    `scheme` selects how the cross terms k_i alpha k_j are discretized. **The choice does not
    matter for the interface problem** -- all four discretize the same symmetrized operator, and
    that operator is unbounded at an abrupt interface whatever stencil is used (see the module
    docstring). The alternatives are kept as a record of what was tried, not as options.

      'central' (default) -- the original `cross_k2_operator`. Samples alpha at NODES over a
          two-cell stencil while the diagonal terms sample it at cell FACES over a compact one.
          That mismatch was long suspected as the cause and is not.
      'fe' -- `cross_k2_operator_fe`. Q1 trilinear element, alpha averaged over each cell's eight
          corners, so both operator families see the interface in the same place. A correct
          operator; REJECTED, because it changes nothing (-143/-93 against central's -118/-85).
      'fe-lumped' -- as 'fe' but with the transverse mass lumped, which makes the bulk symbol
          exactly equal to 'central'. Isolates alpha-sampling from mass smearing, and shows
          neither is the lever (-141/-87).
      'staggered' -- `cross_k2_operator_staggered`. FAILED VALIDATION, kept only as a record of
          a rejected attempt; its symbol is non-negative everywhere and so cannot represent a
          mixed derivative. Do not use.

    Default is 'central' and should stay there: it is a correct discretization of the operator the
    rest of this package assumes, and no other choice here improves on it. Anything relying on a
    fine grid across a heterointerface is unreliable until the ORDERING is fixed.
    """

    SCHEMES = ('central', 'fe', 'fe-lumped', 'staggered')

    def __init__(self, shape, h, periodic=False, scheme='central'):
        if scheme not in self.SCHEMES:
            raise ValueError(f"scheme must be one of {self.SCHEMES}, got {scheme!r}")
        self.shape = shape
        self.h = h
        self.periodic = periodic
        self.scheme = scheme
        self.Dx = first_derivative_operator(shape, 0, h, periodic)
        self.Dy = first_derivative_operator(shape, 1, h, periodic)
        self.Dz = first_derivative_operator(shape, 2, h, periodic)
        if scheme == 'staggered':
            self.Bx = forward_difference_operator(shape, 0, h, periodic)
            self.By = forward_difference_operator(shape, 1, h, periodic)
            self.Bz = forward_difference_operator(shape, 2, h, periodic)

    def k2(self, alpha_field, axis):
        return diagonal_k2_operator(alpha_field, self.h, self.periodic)(axis)

    def cross(self, alpha_field, axis_i, axis_j):
        if self.scheme in ('fe', 'fe-lumped'):
            return cross_k2_operator_fe(alpha_field, self.h, axis_i, axis_j,
                                        periodic=self.periodic,
                                        lump=(self.scheme == 'fe-lumped'))
        if self.scheme == 'staggered':
            B = {0: self.Bx, 1: self.By, 2: self.Bz}
            return cross_k2_operator_staggered(alpha_field, B[axis_i], B[axis_j])
        D = {0: self.Dx, 1: self.Dy, 2: self.Dz}
        return cross_k2_operator(alpha_field, D[axis_i], D[axis_j])


def build_confined_luttinger_kohn(ops, gamma1_field, gamma2_field, gamma3_field,
                                   delta_so_field, V_field):
    """SUPERSEDED AND KNOWN WRONG -- use kp_pryor.confined_hamiltonian instead.

    This matrix fails two checks that the original bulk validation was not sensitive to:
      1. It places the split-off band at -delta on the hole-convention diagonal instead of
         +delta, which makes spurious split-off states the lowest eigenvalues (observed:
         "bound" states ~300 meV below the potential floor).
      2. Its off-diagonal R/S elements are in the wrong positions. In the spherical
         approximation (gamma2 = gamma3) the valence bands must be exactly isotropic; this
         matrix gives 21 meV of anisotropy at |k| = 0.3/nm, kp_pryor gives 1e-16 eV.
    The earlier validation compared only [001] effective masses, where R and S vanish by
    symmetry, and cross-checked the discretization against kp_luttinger.py -- which shares
    both errors, so the agreement proved the discretization correct but not the matrix.

    The DISCRETIZATION machinery in this module (GridOperators, diagonal_k2_operator,
    cross_k2_operator, first_derivative_operator) is unaffected and is what kp_pryor builds on.

    Assemble the confined 6x6-block sparse Hamiltonian (eV) from position-dependent
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
