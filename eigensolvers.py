"""Factorization-free iterative eigensolvers for the 3D Hamiltonians in this package.

Why this module exists
----------------------
Everything up to now solved the eigenproblem by shift-invert Lanczos: factorize (H - sigma*I)
with `scipy.sparse.linalg.splu` and hand the resulting solve to ARPACK as OPinv. That works
well for the single-band Hamiltonian (one unknown per grid point) but was measured here to
collapse for the confined six-band Luttinger-Kohn Hamiltonian, where the matrix is six times
larger in each dimension and far denser per row: on the multiband test grids the factorization
time went ~24 s at N = 9 -> >115 s at N = 11 -> ~116 minutes at N = 31, and alternative fill
reducing orderings (COLAMD, MMD_AT_PLUS_A, NATURAL) did not change the picture. That is
classic sparse-LU fill-in: a 3D grid operator has no good elimination ordering, and the
factor is far denser than the matrix.

Iterative methods never form a factorization -- they only need matrix-vector products, which
for these Hamiltonians cost milliseconds because the matrix has a few tens of nonzeros per
row. This is not a novel idea for this problem: Pryor's own eight-band quantum-dot
calculations, Phys. Rev. B 57, 7190 (1998) (Sec. II), state that the sparse Hamiltonian "is
easily diagonalized using the Lanczos algorithm" -- on 1990s hardware, at grid sizes
comparable to ours.

What is provided
----------------
- `solve_lowest`   : LOBPCG for the algebraically lowest eigenpairs (no factorization).
- `solve_shift_invert` : the existing splu + ARPACK path, kept for cross-checking and because
                     it is still the better choice for the small single-band problems.
- `residual_norms` / `certify` : the verification layer (see below).
- preconditioner builders: `jacobi_preconditioner`, `ilu_preconditioner`.

Verification, and why residuals are the right check
---------------------------------------------------
For a Hermitian matrix A, if ||A v - theta v|| = r for a unit vector v, then A has an
eigenvalue within r of theta -- a rigorous, computable, two-sided error bar obtained from a
single matrix-vector product, independent of which algorithm produced (theta, v). This is the
standard residual bound for the Hermitian eigenproblem; see B. N. Parlett, "The Symmetric
Eigenvalue Problem" (SIAM Classics in Applied Mathematics, 1998), the chapter on
Rayleigh-Ritz approximation and residual bounds. `certify` below applies it.

That bound is exactly what makes an iterative solver safe to trust here. It does NOT, however,
certify that the returned eigenvalues are the *lowest* ones -- only that each is close to
some eigenvalue. Missing a state is the characteristic failure mode of iterative eigensolvers
(and it bit this project before: shift-invert with a badly chosen sigma silently converged to
real-but-not-lowest eigenvalues of the multiband Hamiltonian). `solve_lowest` therefore also
reports whether the block converged with a gap to the next Ritz value, and the recommended
practice is to ask for a few more states than needed and check that the extras are well
separated.

References
----------
- LOBPCG: A. V. Knyazev, "Toward the Optimal Preconditioned Eigensolver: Locally Optimal
  Block Preconditioned Conjugate Gradient Method", SIAM J. Sci. Comput. 23, 517 (2001).
  This is the method `scipy.sparse.linalg.lobpcg` implements.
- Implicitly restarted Lanczos / ARPACK (what `eigsh` wraps): R. B. Lehoucq, D. C. Sorensen
  and C. Yang, "ARPACK Users' Guide" (SIAM, 1998).
- General background on large sparse eigenproblems, Rayleigh-Ritz and shift-invert:
  Y. Saad, "Numerical Methods for Large Eigenvalue Problems", revised ed. (SIAM, 2011), and
  Z. Bai, J. Demmel, J. Dongarra, A. Ruhe and H. van der Vorst (eds.), "Templates for the
  Solution of Algebraic Eigenvalue Problems" (SIAM, 2000).
- Incomplete-LU preconditioning: Y. Saad, "Iterative Methods for Sparse Linear Systems",
  2nd ed. (SIAM, 2003), the ILU chapter.

Per the citation policy in qdsolver_core.py, equation numbers are not quoted.
"""
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------------------
# Verification layer
# --------------------------------------------------------------------------------------

def residual_norms(A, E, V):
    """||A v_i - E_i v_i|| for each column of V (V need not be normalized; it is normalized
    internally so the returned numbers are directly usable as error bars in eV).

    See the module docstring: for Hermitian A this is a rigorous bound on the distance from
    E_i to the nearest true eigenvalue (Parlett, "The Symmetric Eigenvalue Problem").
    """
    V = np.asarray(V)
    norms = np.linalg.norm(V, axis=0)
    Vn = V / norms
    R = A @ Vn - Vn * np.asarray(E)[np.newaxis, :]
    return np.linalg.norm(R, axis=0)


def rayleigh_quotients(A, V):
    """v^H A v / v^H v for each column -- the variationally best energy for a given vector.

    Useful as an independent recomputation of the eigenvalues: a Ritz value returned by the
    solver and the Rayleigh quotient recomputed from its vector must agree, and disagreement
    means the solver's bookkeeping (not the physics) is wrong.
    """
    V = np.asarray(V)
    AV = A @ V
    num = np.einsum('ij,ij->j', V.conj(), AV).real
    den = np.einsum('ij,ij->j', V.conj(), V).real
    return num / den


def certify(A, E, V, tol=1e-6, label='', verbose=True):
    """Run both checks and report. Returns (residuals, rayleigh, ok)."""
    res = residual_norms(A, E, V)
    rq = rayleigh_quotients(A, V)
    drift = np.abs(rq - np.asarray(E))
    ok = bool(np.all(res < tol))
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}max residual {res.max():.3e} eV, max |Ritz - Rayleigh| {drift.max():.3e} eV"
              f"  -> {'OK' if ok else 'NOT CONVERGED at tol=%.0e' % tol}")
    return res, rq, ok


def hermiticity_error(A, n_probe=8, seed=0):
    """Estimate ||A - A^H|| cheaply by probing with random vectors, so Hermiticity can be
    checked on operators that are too large to transpose comfortably. Guards the assumption
    every method here relies on."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    worst = 0.0
    for _ in range(n_probe):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        if np.iscomplexobj(A):
            x = x + 1j * rng.standard_normal(n)
            y = y + 1j * rng.standard_normal(n)
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        # |<y, Ax> - <Ay, x>| over unit vectors is a lower bound on ||A - A^H||, and probing
        # with random directions makes it a usable estimate.
        worst = max(worst, abs(np.vdot(y, A @ x) - np.vdot(A @ y, x)))
    return worst


# --------------------------------------------------------------------------------------
# Preconditioners
# --------------------------------------------------------------------------------------

def jacobi_preconditioner(A, sigma=None):
    """M ~ (diag(A) - sigma)^-1. The cheapest useful preconditioner: no fill-in, no setup
    cost. For a 3D grid operator it is a weak preconditioner (it does nothing about the
    long-wavelength modes that make the operator ill-conditioned), but for these Hamiltonians
    the diagonal carries the large, strongly varying band-offset/split-off potential, so it is
    considerably better than nothing.

    `sigma` should be at or slightly below the wanted part of the spectrum, so the
    preconditioner approximates the inverse of the shifted operator whose small eigenvalues we
    are chasing (Saad, "Numerical Methods for Large Eigenvalue Problems", preconditioning
    discussion).
    """
    d = A.diagonal().real.astype(float)
    if sigma is not None:
        d = d - sigma
    # Avoid dividing by ~0 where the shift lands on a diagonal entry.
    safe = np.where(np.abs(d) < 1e-12, 1.0, d)
    inv = 1.0 / safe

    def apply(x):
        # LOBPCG hands this whole blocks (n, m) as well as single columns (n,) / (n, 1);
        # broadcasting a bare (n,) `inv` against an (n, 1) column would silently build an
        # (n, n) outer product instead of scaling it, so index the axis explicitly.
        return inv[:, np.newaxis] * x if x.ndim == 2 else inv * x

    return spla.LinearOperator(A.shape, matvec=apply, matmat=apply, dtype=A.dtype)


def ilu_preconditioner(A, sigma=0.0, drop_tol=1e-4, fill_factor=10, verbose=False):
    """M ~ (A - sigma*I)^-1 from an INCOMPLETE LU factorization.

    The middle ground between "no factorization" and the complete `splu` that stalled: entries
    below `drop_tol` are discarded during elimination and `fill_factor` caps the memory, so
    the fill-in that made the exact factorization intractable is bounded by construction. The
    result is far too inaccurate to use as a direct solver, which is fine -- as a
    preconditioner it only has to make LOBPCG's job easier.

    References: Y. Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed. (SIAM, 2003),
    the incomplete-LU chapter (ILUT: dual dropping by threshold and by fill count, which is
    what SuperLU's ILU and hence `scipy.sparse.linalg.spilu` implement).
    """
    n = A.shape[0]
    shifted = (A - sigma * sp.identity(n, dtype=A.dtype, format='csr')).tocsc()
    t0 = time.time()
    ilu = spla.spilu(shifted, drop_tol=drop_tol, fill_factor=fill_factor)
    if verbose:
        nnz_ilu = ilu.L.nnz + ilu.U.nnz
        print(f"  ILU: {time.time()-t0:.1f}s, nnz(L)+nnz(U) = {nnz_ilu:,} "
              f"({nnz_ilu/max(A.nnz,1):.1f}x the matrix)")
    # Supply matmat as well as matvec: SuperLU solves a whole block in one call, whereas
    # LinearOperator's default matmat loops over columns.
    return spla.LinearOperator(A.shape, matvec=ilu.solve, matmat=ilu.solve, dtype=A.dtype)


# --------------------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------------------

def solve_lowest(A, k=4, M=None, X0=None, tol=1e-8, maxiter=2000, extra=None,
                 seed=0, verbose=True, certify_tol=1e-6):
    """Algebraically lowest `k` eigenpairs of Hermitian `A` by LOBPCG -- no factorization.

    LOBPCG (Knyazev, SIAM J. Sci. Comput. 23, 517 (2001)) minimizes the Rayleigh quotient over
    a block of vectors, searching at each step the subspace spanned by the current block, the
    preconditioned residuals, and the previous step's directions -- a block preconditioned
    conjugate-gradient method for eigenvalues. Only matrix-vector products with A (and
    applications of M) are needed.

    Practical notes, learned the hard way on this problem:
    - Ask for more vectors than you need. `extra` (default max(2, k//2)) pads the block; LOBPCG
      converges the interior of a block much faster than its edge, so the top vectors of the
      block act as a buffer and the reported `k` come out cleaner. The extras also let you see
      the gap to the next state.
    - `A` must be Hermitian to machine precision, not just in principle. The multiband
      Hamiltonian is Hermitian by construction, but check with `hermiticity_error` if in doubt.
    - LOBPCG is an *extremal* method: it finds the bottom of the spectrum. For hole states, use
      the sign-flip convention already used throughout this package (solve for -H, negate the
      eigenvalues back) rather than trying to target an interior shift.

    Returns (E, V, info) with info carrying the residual norms, wall time, and iteration count.
    """
    n = A.shape[0]
    if extra is None:
        extra = max(2, k // 2)
    block = k + extra
    if block > n:
        block = n

    if X0 is None:
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, block))
        if np.iscomplexobj(A):
            X0 = X0 + 1j * rng.standard_normal((n, block))
    elif X0.shape[1] < block:
        rng = np.random.default_rng(seed)
        pad = rng.standard_normal((n, block - X0.shape[1]))
        if np.iscomplexobj(A):
            pad = pad + 1j * rng.standard_normal((n, block - X0.shape[1]))
        X0 = np.hstack([X0, pad])

    t0 = time.time()
    with warnings.catch_warnings():
        # scipy warns when the requested tolerance is not reached within maxiter; we would
        # rather report the achieved residuals ourselves than have it raise.
        warnings.simplefilter('ignore')
        E, V, hist = spla.lobpcg(A, X0, M=M, largest=False, tol=tol, maxiter=maxiter,
                                 retResidualNormsHistory=True, verbosityLevel=0)
    dt = time.time() - t0
    n_iter = len(hist)

    order = np.argsort(E)
    E, V = E[order], V[:, order]
    res_all = residual_norms(A, E, V)

    E_k, V_k, res_k = E[:k], V[:, :k], res_all[:k]
    gap = (E[k] - E[k - 1]) if len(E) > k else np.nan

    if verbose:
        print(f"  LOBPCG: {dt:.1f}s, {n_iter} iterations, block {block}, max residual on the "
              f"wanted {k}: {res_k.max():.2e} eV; gap to next Ritz value {gap*1e3:.2f} meV")
        if res_k.max() > certify_tol:
            print(f"  WARNING: residual exceeds {certify_tol:.0e} eV -- eigenvalues are only "
                  f"guaranteed to within that residual; raise maxiter or precondition better.")

    info = dict(time=dt, iterations=n_iter, residuals=res_k, residuals_all=res_all,
                block=block, gap_to_next=gap, converged=bool(res_k.max() < certify_tol))
    return E_k, V_k, info


def solve_shift_invert(A, k=4, sigma=0.0, permc_spec='MMD_AT_PLUS_A', verbose=True):
    """Lowest-nearest-to-sigma eigenpairs via a COMPLETE sparse LU factorization + ARPACK.

    Kept as the reference implementation to cross-check `solve_lowest` against, and because it
    remains the faster choice for the small single-band problems. Its weakness is fill-in: see
    the module docstring for the measured scaling that motivated the iterative path.

    References: R. B. Lehoucq, D. C. Sorensen and C. Yang, "ARPACK Users' Guide" (SIAM, 1998)
    for the implicitly restarted Lanczos method `eigsh` wraps; the shift-invert spectral
    transformation is covered there and in Saad's "Numerical Methods for Large Eigenvalue
    Problems".
    """
    n = A.shape[0]
    t0 = time.time()
    shifted = (A - sigma * sp.identity(n, dtype=A.dtype)).tocsc()
    lu = spla.splu(shifted, permc_spec=permc_spec)
    t_fact = time.time() - t0
    OPinv = spla.LinearOperator((n, n), matvec=lu.solve, dtype=A.dtype)
    E, V = spla.eigsh(A, k=k, sigma=sigma, which='LM', OPinv=OPinv)
    dt = time.time() - t0
    order = np.argsort(E)
    E, V = E[order], V[:, order]
    if verbose:
        nnz_lu = lu.L.nnz + lu.U.nnz
        print(f"  shift-invert: {dt:.1f}s total ({t_fact:.1f}s factorization), "
              f"nnz(L)+nnz(U) = {nnz_lu:,} ({nnz_lu/max(A.nnz,1):.1f}x the matrix)")
    info = dict(time=dt, factorization_time=t_fact, residuals=residual_norms(A, E, V))
    return E, V, info


def folded_operator(A, sigma):
    """(A - sigma*I)^2 as a LinearOperator -- never formed explicitly (squaring the matrix
    would destroy its sparsity); each application is two matrix-vector products."""
    n = A.shape[0]

    def apply(x):
        y = A @ x - sigma * x
        return A @ y - sigma * y

    return spla.LinearOperator((n, n), matvec=apply, matmat=apply, dtype=A.dtype)


def solve_interior(A, k=4, sigma=0.0, M=None, tol=1e-8, maxiter=3000, extra=None,
                   seed=0, verbose=True, certify_tol=1e-6):
    """Eigenpairs of Hermitian `A` closest to `sigma`, WITHOUT a factorization -- the folded
    spectrum method.

    A and (A - sigma*I)^2 have the same eigenvectors, and the eigenvalues of A nearest sigma
    become the SMALLEST eigenvalues of the squared operator. So an extremal method (LOBPCG
    here) can reach interior states, which is what the eight-band Hamiltonian needs: its bound
    electron states sit in the gap with the entire valence continuum below them, and its bound
    hole states have the conduction continuum above, so neither is at an end of the spectrum.

    References: L.-W. Wang and A. Zunger, "Solving Schroedinger's equation around a desired
    energy: application to silicon quantum dots", J. Chem. Phys. 100, 2394 (1994),
    doi:10.1063/1.466486 -- the folded spectrum method.

    The price is conditioning: squaring the operator squares the eigenvalue gaps, so relative
    separations shrink and convergence is markedly slower than for an extremal problem. That
    is the trade against shift-invert, which converges in few iterations but needs the
    factorization this package cannot afford at multiband sizes. Choose `sigma` inside the gap
    and close to the states you want; the closer it is, the better the folded gaps.

    Eigenvalues are recovered as Rayleigh quotients of the ORIGINAL A (not as
    sigma +/- sqrt(mu), which loses the sign), so the returned energies are directly
    comparable with the other solvers here.
    """
    F = folded_operator(A, sigma)
    n = A.shape[0]
    if extra is None:
        extra = max(2, k // 2)
    block = min(k + extra, n)

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, block))
    if np.iscomplexobj(A):
        X0 = X0 + 1j * rng.standard_normal((n, block))

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu, V, hist = spla.lobpcg(F, X0, M=M, largest=False, tol=tol, maxiter=maxiter,
                                  retResidualNormsHistory=True, verbosityLevel=0)
    dt = time.time() - t0

    E = rayleigh_quotients(A, V)
    order = np.argsort(np.abs(E - sigma))       # closest to sigma first
    E, V = E[order][:k], V[:, order][:, :k]
    order2 = np.argsort(E)                      # then report in ascending energy
    E, V = E[order2], V[:, order2]
    res = residual_norms(A, E, V)

    if verbose:
        print(f"  folded LOBPCG: {dt:.1f}s, {len(hist)} iterations, block {block}, "
              f"sigma = {sigma:.4f} eV, max residual {res.max():.2e} eV")
        if res.max() > certify_tol:
            print(f"  WARNING: residual exceeds {certify_tol:.0e} eV -- the folded problem is "
                  f"badly conditioned; move sigma closer to the wanted states or raise maxiter.")

    info = dict(time=dt, iterations=len(hist), residuals=res,
                converged=bool(res.max() < certify_tol))
    return E, V, info


def solve_lowest_hole(A, k=4, **kwargs):
    """Hole states: the physically wanted valence states are at the TOP of the (electron-
    convention) spectrum, which is not where an extremal method looks. Negate, solve for the
    bottom, negate back -- the same sign-flip convention `qdsolver_core.solve_states` uses.

    This is what makes LOBPCG applicable to the valence problem at all, and it is strictly
    better than the interior shift-invert it replaces: `kp_confined.safe_sigma_hole` exists
    only because choosing an interior target sigma was error-prone, and there is no sigma to
    choose here.
    """
    E, V, info = solve_lowest(-A, k=k, **kwargs)
    return -E[::-1], V[:, ::-1], info
