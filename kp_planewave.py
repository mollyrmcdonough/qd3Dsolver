"""Six-band k.p in a PLANE-WAVE basis, with Burt-Foreman ordering. The small-dot solver.

In one paragraph: the interface modes that made small dots unreachable are not a discretization
defect -- the SYMMETRIZED six-band operator has no maximum at an abrupt interface, and every
correct discretization must reproduce that. Burt-Foreman ordering is a different operator and
fixes it: lambda_max of the kinetic tensor drops from +1.741 to machine zero, and a cutoff ladder
that ran away from 0.457 to 1.951 eV converges instead to 0.1813 eV with a final step of 0.3 meV.
That is Yeap et al.'s 2.5 nm island, which the finite-difference solver cannot compute at any
spacing. `ordering='burt-foreman'` is the default here.

What this module was built to do, and what it found instead
-----------------------------------------------------------
It was built as an independent referee. `kp_confined`'s finite-difference solver produces
interface-pinned modes whose energy diverges as ~0.046/h^2 eV nm^2 -- Hermitian, Kramers-paired,
converged to 1e-8, and 40-70% localized in the island, so no localization threshold separates
them from real states. Three fixes had been tried and three had failed: correcting the
compact-vs-central sampling mismatch, a staggered cross term (a WRONG operator -- its symbol is
non-negative everywhere and cannot represent a mixed derivative), and a Q1 finite-element cross
term (a CORRECT operator, symbol exact to 7e-15, that changed nothing). A plane-wave basis has no
stencil at all, so it could settle whether any grid was to blame.

It settled it. **The modes are not a discretization defect. The symmetrized six-band operator
with an abrupt heterointerface has no maximum, and every correct discretization must reproduce
that.** The three fixes did not fail; they were aimed at a cause that was not there.

The argument
------------
Write the six-band valence Hamiltonian in divergence form,

    H = sum_ij  k_i A_ij(r) k_j  +  W(r),        A_ij = A_ji^dagger,   W = W^dagger

with the 6x6 matrix fields A_ij built from gamma1, gamma2, gamma3 and W from the band edges and
the Bir-Pikus strain terms. Collect the direction index i and the band index b into one index and
A becomes a single 18x18 Hermitian matrix (`kinetic_tensor`). Whether the spectrum is bounded
above is then a question about its eigenvalues, and there are two different conditions:

* **Legendre-Hadamard ellipticity** -- <phi|A|phi> <= 0 for RANK-ONE phi = k (x) v only. This is
  exactly the statement that every bulk band curves downward, i.e. gamma1 - 2 gamma2 > 0 and
  gamma1 - 2 gamma3 > 0. It holds comfortably for every material here.
* **Strong ellipticity** -- the same for EVERY 18-vector. It fails for every material here:

      material   gamma1-2gamma2  gamma1-2gamma3   lambda_max(A)   worst rank-one
      GaAs            +2.86           +1.12          +0.226          -0.043
      GaSb            +4.00           +1.40          +0.533          -0.063
      InAs            +3.00           +1.60          +0.937          -0.082
      InSb            +3.80           +1.80          +1.741          -0.102

With CONSTANT coefficients Fourier diagonalizes the form and only rank-one directions ever
appear, so Legendre-Hadamard suffices; the same holds for coefficients that vary CONTINUOUSLY, by
freezing them locally. At a DISCONTINUOUS interface that argument fails and the positive
directions of A become accessible. The form then has no upper bound, and a Galerkin method
enlarging its basis climbs forever.

The measurement (`scripts/planewave_validation.py --only 3`)
------------------------------------------------------------
On Yeap et al.'s 2.5 nm InSb/InAs island, holding the grid at h = 0.20 nm -- so the coefficient
field is FIXED and only the basis grows -- and scaling gamma2, gamma3 by s, which moves
lambda_max(A) through zero and changes nothing else:

    gmax     s = 1.00 (lambda_max = +1.741)    s = 0.25 (lambda_max = -0.559)
    3.00              +0.4571                            +0.0260
    4.00              +0.5773                            +0.0261
    5.00              +1.0115                            +0.0262
    6.00              +1.2030                            +0.0262
    7.00              +1.5860                            +0.0263
    7.85              +1.9508                            +0.0263

    island valence edge 0.6773 eV -- s = 1 passes it near gmax 4.3 and never stops

Same geometry, same abrupt interface, same strain, same band edges, same solver. The only thing
that changed is the sign of lambda_max(A), and it is the difference between a sequence converged
to 0.1 meV and one that has quadrupled and is still climbing.

Refining the grid instead, at a fixed basis fraction gmax = nyquist/2, separates the two cases
just as sharply. At s = 1 the ceiling scales as 1/h^2 -- E*h^2 = 0.094, 0.094, 0.073, 0.078
eV nm^2 at h = 0.40, 0.30, 0.25, 0.20 -- while at s = 0.25 the energy itself is flat at 0.0269,
0.0232, 0.0256, 0.0263 eV. The finite-difference solver's artifact was measured at 0.046/h^2
eV nm^2: the same law, the same order, and for the same reason. h is not acting as a stencil
spacing in either method. It is setting how sharp the interface is allowed to be, which is what
cuts off an otherwise unbounded operator.

THE FIX: Burt-Foreman ordering, implemented here and it works
--------------------------------------------------------------
Ordering is a statement about which coefficient sits BETWEEN k_i and k_j, and it is naturally
stated in the Dresselhaus-Kip-Kittel orbital basis, where the cross term is one scalar N. Writing
the XY element of the valence Hamiltonian as

    H_XY = k_x N+ k_y  +  k_y N- k_x ,        N+ + N- = N = -6 gamma3 (hbar^2/2m0)

any split reproduces the bulk, because k_x and k_y commute. Symmetrized ordering takes
N+ = N- = N/2. Burt's exact envelope-function theory and Foreman's valence-band result fix it
instead at **N- = M**, the same coefficient that appears on the diagonal (Burt, J. Phys.: Condens.
Matter 4, 6651 (1992); Foreman, Phys. Rev. B 48, 4964 (1993)).

That single change takes lambda_max(A) from +1.741 to **5e-16** for InSb, and to machine zero for
every other material here -- negative semidefinite, so the operator is bounded above by the local
valence edge and `spectrum_bound` becomes a real bound. Note what it implies physically:
Burt-Foreman puts gamma1 and gamma2 into the CROSS terms, where symmetrized ordering had only
gamma3. That is why holding gamma3 uniform appeared to cure the artifact.

The acceptance test, same island and grid as above, real parameters throughout, only the ordering
different (`scripts/planewave_validation.py --only 5`):

    gmax        symmetrized                  burt-foreman
    3.00          +0.4571                       +0.1727
    4.00          +0.5773   (+120 meV)          +0.1787   (+6.0 meV)
    5.00          +1.0115   (+434 meV)          +0.1799   (+1.1 meV)
    6.00          +1.2030   (+192 meV)          +0.1808   (+1.0 meV)
    7.00          +1.5860   (+383 meV)          +0.1811   (+0.2 meV)
    7.85          +1.9507   (+365 meV)          +0.1813   (+0.3 meV)

    variational bound 0.7630 eV: burt-foreman respects it, symmetrized exceeds it by 1.19 eV

Converged to a few tenths of a meV, at a dot size the finite-difference solver could not touch at
any spacing. `ordering='burt-foreman'` is therefore the DEFAULT for every solver entry point here;
`'symmetrized'` remains available because the diagnostics above are about it.

It holds InAs/GaAs too, and that case is more informative than it looks
(`scripts/planewave_validation.py --only 6`). On a 12 nm InAs/GaAs ellipsoid at h = 0.40,
confinement over a rising cutoff:

    symmetrized     13.3 -> -68.5 -> -264.8 meV   (last step -196.3: running away)
    burt-foreman    39.1 ->  38.5 ->   38.3 meV   (last step   -0.2: converged)

and at 6 nm, 12.6 -> -246.2 against 97.5 -> 95.8. So the SYMMETRIZED operator is unbounded in
InAs/GaAs as well -- the eightfold smaller lambda_max delays the runaway, it does not prevent it.
The Pryor benchmark passed because it was run on a large island at a coarse spacing, where the
0.046/h^2 ceiling is small against the well; that is a statement about where the benchmark sits,
not about the operator being sound there.

The basis change this needs is derived, not transcribed. `dkk_to_pryor_unitary` solves the
intertwining condition for the 6x6 unitary between the DKK basis and Pryor's, and checks it is
unique, unitary, and exact at fresh k -- rather than hand-copying Clebsch-Gordan coefficients
whose phase conventions differ between sources.

What is still true, and what to watch
-------------------------------------
0. **With the ordering fixed, the box becomes the binding constraint.** A six-band hole carries a
   LIGHT-hole admixture, and its tail is long even when the state is nominally deep: at m ~ 0.015
   and E ~ 0.18 eV the decay length is 3.7 nm against 0.73 nm for the heavy component. Measured on
   the 2.5 nm island at h = 0.25, widening the crop 1.5 -> 2 -> 3 -> 4 nm moves the level by
   -7.3, -5.8, -1.9 meV. Converging, but not at the 2 nm crop that is convenient. `crop_env`
   exists so the strain can keep a generous box while the k.p box is chosen on this criterion --
   and the criterion has to be measured per system, not assumed from the confinement energy.
1. **Smoothing is a regularization, never a fix.** A continuous coefficient also restores
   boundedness, but the bound grows as the width shrinks, so the answer depends on the width.
   `smooth_fields` takes a width in nanometres precisely so a grid-refinement test means
   something. With the ordering fixed there is no reason to reach for it.
2. **The finite-difference solver is still symmetrized.** `kp_confined` and everything built on it
   (`heterostructure.six_band_holes`, the production sweeps) discretize the unbounded operator and
   keep their resolution floor. Porting the ordering there is a separate job.
3. **The link between k.p ellipticity and spurious solutions is treated in the literature** --
   Foreman, Phys. Rev. B 56, R12748 (1997), and Veprek, Steiger and Witzigmann, Phys. Rev. B 76,
   165320 (2007) -- neither of which was read here. The derivation and measurements above stand on
   their own; check them against those papers rather than assume they agree.
4. **The InAs/GaAs benchmark was never evidence that the antimonides would behave.** lambda_max is
   +0.226 for GaAs against +1.741 for InSb, an eightfold difference, and that is the difference
   between a benchmark that passed and a system that could not be computed at all.

What is validated here
----------------------
The assembly is exact, and that is worth separating from the physics above. `validate_channels`
reproduces `kp_pryor.bulk_hamiltonian` to 0.0; a uniform material gives blocks equal to the bulk
matrix at every G to 1.8e-15 eV, with no coupling between different G at all; Hermiticity is
3.0e-16; the exact diagonal matches the built matrix to 3.6e-15 eV; and the strongly elliptic
control obeys the variational bound. Section 1 of `scripts/planewave_validation.py` runs all of
it in seconds.

It also agrees with the finite-difference solver on a real 3D heterostructure -- but only where
that question is answerable. Section 4 runs both on a 12 nm InSb/InAs ellipsoid with gamma2,
gamma3 scaled by 0.25, so the operator is strongly elliptic and a right answer exists, and both
periodic so the boundary-value problem is literally the same:

    h = 0.50   FD (186,624 unknowns) +0.36835 eV   PW +0.36436 -> +0.36567 -> +0.36653  (-1.8 meV)
    h = 0.40   FD (380,880 unknowns) +0.36652 eV   PW +0.36359 -> +0.36481 -> +0.36551  (-1.0 meV)

The plane-wave value climbs toward the finite-difference one from below, as a Galerkin method
must, and the finite-difference value itself moves 1.8 meV between the two grids -- so the two
completely independent discretizations are converging on the same number to about a millielectron
volt. Running the same comparison at s = 1 would be meaningless: neither method converges there,
so agreement or disagreement would say nothing about either.

Method
------
Periodic supercell on the grid the strain solver already uses (`crop_env` narrows it by
subsetting, never by interpolating). The matrix element of a kinetic term is

    <G| k_i alpha k_j |G'>  =  G_i  alpha(G - G')  G'_j

exactly -- no stencil, no cell, no choice about where the material changes -- provided the cutoff
stays within half the Nyquist wavenumber, so the coefficient convolution never wraps. The apply is
matrix-free: two batched FFTs and twelve small gemms.

Conventions
-----------
Electron convention throughout, matching `kp_pryor` -- valence bands curve downward, so the hole
ground state is the LARGEST algebraic eigenvalue and no shift-invert is needed.

Plane waves are normalized as psi(r) = sum_G c_G exp(i G.r), i.e. `norm='forward'` in numpy's FFT,
under which sum_G |c_G|^2 = <|psi|^2> over the cell. The coefficient-vector inner product is then
the plain Euclidean one and the discretized operator is Hermitian in it.

Periodic boundary conditions. For a hole bound by hundreds of meV, which decays within a
nanometre or two, this is equivalent to the finite-difference solver's Dirichlet box; for a
weakly bound or unbound carrier it is not, and neither is the box.
"""
import time

import numpy as np

import kp_pryor as kp
from qdsolver_core import HBAR2_OVER_2M0 as G0

try:                       # scipy.fft threads the transform; numpy.fft does not.
    import scipy.fft as _fftmod
    _FFT_WORKERS = -1
except ImportError:        # pragma: no cover
    import numpy.fft as _fftmod
    _FFT_WORKERS = None

SQRT3 = np.sqrt(3.0)

#: The eight real parameter channels of Pryor's matrix. `A` (the conduction diagonal) is carried
#: for completeness and is identically zero in the six-band slice.
CHANNELS = ('A', 'P', 'Q', 'R_re', 'R_im', 'S_re', 'S_im', 'delta')


def term_matrices(n_bands=6):
    """The constant band matrices M_c such that H = sum_c value_c * M_c.

    Pryor's matrix is REAL-LINEAR in the eight channels (P, Q, Re R, Im R, Re S, Im S, delta, A):
    `kp_pryor._blocks` builds it from those scalars using only complex conjugation, multiplication
    by fixed constants, and addition. So setting one channel to 1 (or to i, for the real and
    imaginary halves of the complex ones) and the rest to zero reads off the matrix that channel
    multiplies -- and the whole Hamiltonian is recovered by superposition.

    This is deliberately DERIVED from `kp_pryor` rather than transcribed again. A second
    transcription of the same matrix would make any disagreement between the two solvers
    ambiguous: the entire value of this module is that it is a different DISCRETIZATION of the
    same Hamiltonian, so the Hamiltonian must come from one place. `validate_channels` checks the
    superposition reproduces `kp_pryor.bulk_hamiltonian` exactly.

    Each M is Hermitian, which `_assemble` guarantees by construction (it fills the upper triangle
    and completes by adjoint), and which is asserted here anyway.
    """
    out = {}
    for key in CHANNELS:
        vals = dict(A=0j, P=0j, Q=0j, R=0j, S=0j, delta=0j)
        if key == 'R_re':
            vals['R'] = 1.0 + 0j
        elif key == 'R_im':
            vals['R'] = 1j
        elif key == 'S_re':
            vals['S'] = 1.0 + 0j
        elif key == 'S_im':
            vals['S'] = 1j
        else:
            vals[key] = 1.0 + 0j
        blocks = kp._blocks(vals['A'], vals['P'], vals['Q'], vals['R'], vals['S'],
                            0j, 0j, vals['delta'], np.conj)
        rows = kp._assemble(blocks, n_bands, np.conj, 0j)
        M = np.array(rows, dtype=complex)
        if not np.allclose(M, M.conj().T, atol=0, rtol=0):
            raise AssertionError(f"channel matrix {key} is not Hermitian")
        out[key] = M
    return out


def kinetic_terms():
    """SYMMETRIZED ordering only, kept as the independent check on `kinetic_blocks`.

    Not what the solver uses -- `kinetic_blocks` is, because it can express Burt-Foreman ordering
    and this cannot. This construction reads the symmetrized tensor straight off `kp_pryor`'s
    channel decomposition, by an entirely different route from the DKK basis change, and
    `validate_ordering('symmetrized')` asserts the two agree. That agreement is what ties the
    derived basis change back to the validated Hamiltonian, so do not delete this.

    The kinetic part as (alpha field, channel, C) triples. Each triple means a contribution
    sum_ij k_i [ C_ij * alpha(r) * M_channel ] k_j to the divergence-form Hamiltonian, with C a
    real SYMMETRIC 3x3 array carrying the hbar^2/2m0 factor. Symmetry of C is what makes
    A_ij = A_ji and hence forces symmetrized ordering; it is also precisely the assumption
    Burt-Foreman breaks.

    Read off `kp_pryor.bulk_hamiltonian`'s valence branch:

        P = -Ev + g1 G k^2                      -> C = G * I,            channel P
        Q = g2 G (kx^2 + ky^2 - 2 kz^2)         -> C = G * diag(1,1,-2), channel Q
        Re R = -sqrt3 G g2 (kx^2 - ky^2)        -> C = G * diag(-s3,s3,0), channel R_re
        Im R = 2 sqrt3 G g3 kx ky               -> C_xy = C_yx = G s3,   channel R_im
        Re S = 2 sqrt3 G g3 kz kx               -> C_zx = C_xz = G s3,   channel S_re
        Im S = -2 sqrt3 G g3 kz ky              -> C_zy = C_yz = -G s3,  channel S_im

    Note the factor of two on the cross terms is absorbed by writing them symmetrically: the pair
    (C_xy, C_yx) = (s3, s3) reproduces 2 sqrt3 kx ky.

    Only gamma1, gamma2 and gamma3 appear -- there are three coefficient fields in the whole
    kinetic operator, which is what makes the matrix-free apply cheap.
    """
    terms = []

    def C(**kw):
        m = np.zeros((3, 3))
        for k, v in kw.items():
            i, j = 'xyz'.index(k[0]), 'xyz'.index(k[1])
            m[i, j] = v
        return G0 * m

    terms.append(('gamma1', 'P', C(xx=1.0, yy=1.0, zz=1.0)))
    terms.append(('gamma2', 'Q', C(xx=1.0, yy=1.0, zz=-2.0)))
    terms.append(('gamma2', 'R_re', C(xx=-SQRT3, yy=SQRT3)))
    terms.append(('gamma3', 'R_im', C(xy=SQRT3, yx=SQRT3)))
    terms.append(('gamma3', 'S_re', C(zx=SQRT3, xz=SQRT3)))
    terms.append(('gamma3', 'S_im', C(zy=-SQRT3, yz=-SQRT3)))
    return terms


# --------------------------------------------------------------------------------------
# Operator ordering: the DKK route, and Burt-Foreman
# --------------------------------------------------------------------------------------
#
# Ordering is a statement about which coefficient sits BETWEEN k_i and k_j, and it is natural in
# the Dresselhaus-Kip-Kittel orbital basis (X, Y, Z) x spin rather than in Pryor's |J, mJ> basis,
# because that is where the literature states it and where the cross terms are a single scalar.
# So the ordered tensor is built in the DKK basis and rotated into Pryor's -- with the rotation
# DERIVED numerically and checked to machine precision, never transcribed.

ORDERINGS = ('symmetrized', 'burt-foreman')

_SIGMA = (np.array([[0, 1], [1, 0]], dtype=complex),
          np.array([[0, -1j], [1j, 0]], dtype=complex),
          np.array([[1, 0], [0, -1]], dtype=complex))

_EPS = np.zeros((3, 3, 3))
for _a, _b, _c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _EPS[_a, _b, _c], _EPS[_a, _c, _b] = 1.0, -1.0


def dkk_kinetic_blocks(gamma1, gamma2, gamma3, ordering='symmetrized'):
    """{(i, j): 3x3 orbital matrix} for the kinetic tensor in the DKK (X, Y, Z) basis.

    The Dresselhaus-Kip-Kittel form of the valence-band kinetic operator is

        H_ab = sum_ij k_i A_ij[a, b] k_j,     a, b in {X, Y, Z}

    with three independent coefficients, here in Luttinger form (the free-electron hbar^2 k^2/2m0
    folded in, so there is no separate identity term):

        L = -(gamma1 + 4 gamma2) G      the XX element for k along x   (light hole along [001])
        M = (2 gamma2 - gamma1) G       the YY and ZZ elements         (heavy hole along [001])
        N = -6 gamma3 G                 the XY element, total

    and the diagonal blocks A_xx = diag(L, M, M), A_yy = diag(M, L, M), A_zz = diag(M, M, L).

    **The ordering enters only through how N is split.** With position-dependent parameters the XY
    element is written

        H_XY = k_x N+ k_y  +  k_y N- k_x ,          N+ + N- = N

    and any split reproduces the bulk Hamiltonian, because k_x and k_y commute. They differ only
    where the coefficients vary.

      'symmetrized' -- N+ = N- = N/2. Hermitian, obvious, and the source of every interface
          problem in this package: the resulting tensor has eigenvalues up to +1.74 (InSb), so the
          operator has no maximum at a step.

      'burt-foreman' -- **N- = M**. Burt's exact envelope-function theory (Burt, J. Phys.:
          Condens. Matter 4, 6651 (1992)) and Foreman's valence-band result (Foreman, Phys. Rev. B
          48, 4964 (1993)) fix the split rather than leaving it free. The consequence, measured in
          `strong_ellipticity`, is that lambda_max of the kinetic tensor drops to EXACTLY ZERO for
          every material here -- negative semidefinite, and hence bounded above by the local
          valence edge.

          PROVENANCE, because this matters and the citation policy requires it: the specific rule
          N- = M was NOT read out of Foreman 1993. It was taken from a secondary source stating the
          DKK-basis form (kx N+ ky + ky N- kx with L, M, N+, N- given explicitly), and it is
          corroborated here rather than trusted -- lambda_max landing on machine zero for four
          different materials is not something a wrong split would do, and the bulk Hamiltonian is
          verified unchanged to 1.7e-15 eV. Check it against the primary source before publishing
          a number, and be aware that a shift of any CONSTANT between N+ and N- is invisible to
          every test here because it leaves the operator identically unchanged.

    A note on the free-electron term, because the literature statement looks different. Foreman's
    condition is usually quoted as N- = M in DKK variables where M carries a -hbar^2/2m0 from the
    free-electron term and the Hamiltonian carries a separate +hbar^2 k^2/2m0 identity. Folding
    that identity into L and M (exact, since m0 is a universal constant) moves the same constant
    into M. Shifting a CONSTANT c between N+ and N- leaves the operator identically unchanged --
    k_x c k_y - k_y c k_x = 0, exactly, in the continuum and in this discretization -- so the two
    statements are the same operator. They are NOT the same tensor, though, and the Luttinger
    convention used here is the one that makes lambda_max land on zero rather than on
    +hbar^2/2m0 = +0.0381. The bound is a property of the representation, the operator is not.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    L = -(gamma1 + 4.0 * gamma2) * G0
    M = (2.0 * gamma2 - gamma1) * G0
    N = -6.0 * gamma3 * G0
    n_minus = M if ordering == 'burt-foreman' else N / 2.0
    n_plus = N - n_minus

    blocks = {}
    for i, diag in enumerate(([L, M, M], [M, L, M], [M, M, L])):
        blocks[(i, i)] = np.diag(diag).astype(complex)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        blk = np.zeros((3, 3), dtype=complex)
        blk[i, j], blk[j, i] = n_plus, n_minus
        blocks[(i, j)] = blk
        blocks[(j, i)] = blk.conj().T          # Hermiticity of the whole: A_ji = A_ij^dagger
    return blocks


def dkk_spin_orbit(delta_so):
    """The spin-orbit term in the DKK basis, ordered [X up, X dn, Y up, Y dn, Z up, Z dn].

    H_so = (2 delta / 3 hbar^2) L.S, shifted by -delta/3 so the J = 3/2 quadruplet sits at zero
    and the split-off pair at -delta, which is Pryor's convention. For p orbitals in the real
    (X, Y, Z) basis the angular momentum is (L_a)_bc = -i hbar eps_abc -- one line, and the
    eigenvalues (four at 0, two at -delta) check it.
    """
    so = np.zeros((6, 6), dtype=complex)
    for a in range(3):
        so += np.kron(-1j * _EPS[a], _SIGMA[a])
    return (delta_so / 3.0) * (so - np.eye(6))


def dkk_bulk_hamiltonian(kx, ky, kz, params, ordering='symmetrized', Ev=0.0):
    """The six-band bulk Hamiltonian assembled in the DKK basis. Equal to
    `kp_pryor.bulk_hamiltonian` up to the basis change `dkk_to_pryor_unitary` returns."""
    blocks = dkk_kinetic_blocks(params['gamma1L'], params['gamma2L'], params['gamma3L'],
                                ordering)
    k = (kx, ky, kz)
    H = np.zeros((6, 6), dtype=complex)
    for (i, j), blk in blocks.items():
        H += k[i] * k[j] * np.kron(blk, np.eye(2))
    return H + dkk_spin_orbit(params['delta_so']) + Ev * np.eye(6)


_UNITARY_CACHE = {}


def dkk_to_pryor_unitary(nk=12, seed=0, atol=1e-11):
    """The 6x6 unitary U with U^dagger H_DKK(k) U = kp_pryor.bulk_hamiltonian(k), DERIVED.

    Transcribing the Clebsch-Gordan coefficients between the (X, Y, Z) x spin basis and Pryor's
    |J, mJ> basis by hand is exactly the kind of step this package has been bitten by -- phase and
    ordering conventions differ between sources, a wrong one produces a plausible Hamiltonian, and
    nothing downstream would catch it. So U is not transcribed. It is solved for.

    The intertwining condition H_DKK(k) U = U H_Pryor(k) is LINEAR in U, so stacking it over
    several k gives a null-space problem. Two facts make it well posed:

      * H(k) is quadratic in k, so the six independent products k_i k_j plus the constant
        spin-orbit term make SEVEN structures that must intertwine separately. Fewer than seven
        k-points fits the sampled points and fails everywhere else -- measured, when this was first
        written with five: residual 1.4e-14 at the fitted k and 46 eV at a fresh one.
      * Kramers degeneracy makes the commutant at any SINGLE k large, so one k cannot determine U
        even in principle. Several generic k reduce it to scalars, which is the 1 in the assertion
        below.

    Verified here, not assumed: the null space is one-dimensional, U is unitary, and the
    intertwining holds at fresh k to ~1e-13 eV. The same U comes out for every material (they
    differ only by a global phase), as it must for a basis change.
    """
    key = (nk, seed)
    if key in _UNITARY_CACHE:
        return _UNITARY_CACHE[key]

    import materials as mt
    p = mt.kp_params(mt.material('InSb'))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nk):
        k = rng.normal(size=3)
        Hd = dkk_bulk_hamiltonian(k[0], k[1], k[2], p)
        Hj = kp.bulk_hamiltonian(k[0], k[1], k[2], p, n_bands=6, Ev=0.0)
        rows.append(np.kron(np.eye(6), Hd) - np.kron(Hj.T, np.eye(6)))
    _, s, Vh = np.linalg.svd(np.vstack(rows))
    n_null = int((s < 1e-8 * s.max()).sum())
    if n_null != 1:
        raise AssertionError(f"intertwiner null space has dimension {n_null}, expected 1; "
                             f"increase nk (currently {nk})")
    # Column-major unpacking: vec(A X B) = (B^T (x) A) vec(X) is the column-major identity, so a
    # C-order reshape here silently transposes U -- which fits the sampled k and nothing else.
    U = Vh[-1].conj().reshape(6, 6, order='F')
    U = U / np.sqrt(abs(np.trace(U.conj().T @ U)) / 6.0)

    err_u = float(np.abs(U.conj().T @ U - np.eye(6)).max())
    worst = 0.0
    for nm in ('InSb', 'InAs', 'GaAs', 'GaSb'):
        q = mt.kp_params(mt.material(nm))
        for _ in range(4):
            k = rng.normal(size=3) * 2.0
            Hd = dkk_bulk_hamiltonian(k[0], k[1], k[2], q)
            Hj = kp.bulk_hamiltonian(k[0], k[1], k[2], q, n_bands=6, Ev=0.0)
            worst = max(worst, float(np.abs(U.conj().T @ Hd @ U - Hj).max()))
    if err_u > 1e-12 or worst > atol:
        raise AssertionError(f"basis change failed: unitarity {err_u:.2e}, "
                             f"worst intertwining residual {worst:.2e} eV")
    _UNITARY_CACHE[key] = (U, dict(unitarity=err_u, residual=worst, n_null=n_null))
    return _UNITARY_CACHE[key]


def kinetic_blocks(ordering='symmetrized'):
    """[(i, j, alpha_name, B)] with B a 6x6 matrix in PRYOR'S band basis.

    The canonical form the solver consumes: the kinetic operator is

        sum over entries  k_i [ alpha(r) B ] k_j

    and Hermiticity holds because every entry (i, j, alpha, B) is accompanied by
    (j, i, alpha, B^dagger). Note B is NOT Hermitian for i != j under Burt-Foreman ordering, and
    that asymmetry IS the ordering -- a solver that assumes A_ij = A_ji cannot represent it.

    A_ij is linear and homogeneous in (gamma1, gamma2, gamma3), so the three coefficient matrices
    are read off by evaluating the DKK blocks at unit gamma and rotating. Which also means
    Burt-Foreman puts gamma1 and gamma2 into the CROSS terms, where symmetrized ordering has only
    gamma3: N- = M = (2 gamma2 - gamma1) G. That is the whole content of the reordering, and it is
    why holding gamma3 uniform appeared to fix things under the old scheme.
    """
    U, _ = dkk_to_pryor_unitary()
    out = []
    for n, name in enumerate(('gamma1', 'gamma2', 'gamma3')):
        unit = [0.0, 0.0, 0.0]
        unit[n] = 1.0
        for (i, j), blk in dkk_kinetic_blocks(*unit, ordering=ordering).items():
            B = U.conj().T @ np.kron(blk, np.eye(2)) @ U
            if np.abs(B).max() > 1e-14:
                out.append((i, j, name, np.ascontiguousarray(B)))
    return out


def validate_ordering(ordering='symmetrized', atol=1e-13):
    """Assert the DKK-built kinetic blocks reproduce the channel construction, and that
    Burt-Foreman differs from symmetrized ONLY in the bulk-invariant way it is allowed to.

    Two checks in one. For 'symmetrized' the blocks must equal `kinetic_terms`' assembly exactly,
    which ties the DKK route and the basis change back to `kp_pryor`. For either ordering the BULK
    Hamiltonian must be unchanged, since k_i and k_j commute and only the split differs -- if that
    fails, the reordering has changed the physics rather than the operator ordering.
    """
    import materials as mt
    A = {}
    for i, j, name, B in kinetic_blocks(ordering):
        A[(i, j)] = A.get((i, j), {})
        A[(i, j)][name] = B

    p = mt.kp_params(mt.material('InSb'))
    g = dict(gamma1=p['gamma1L'], gamma2=p['gamma2L'], gamma3=p['gamma3L'])
    k = np.array([0.31, -0.17, 0.23])
    H = np.zeros((6, 6), dtype=complex)
    for (i, j), d in A.items():
        for name, B in d.items():
            H += k[i] * k[j] * g[name] * B
    M = term_matrices(6)
    H += -p['gamma1L'] * 0.0 + p['delta_so'] * M['delta']

    ref = kp.bulk_hamiltonian(k[0], k[1], k[2], p, n_bands=6, Ev=0.0)
    bulk_err = float(np.abs(H - ref).max())

    herm = 0.0
    for (i, j), d in A.items():
        for name, B in d.items():
            other = A.get((j, i), {}).get(name)
            herm = max(herm, np.inf if other is None else
                       float(np.abs(B.conj().T - other).max()))

    out = dict(bulk_error=bulk_err, hermiticity=herm)
    if bulk_err > atol or herm > atol:
        raise AssertionError(f"ordering {ordering!r} failed: bulk error {bulk_err:.2e} eV, "
                             f"pairwise Hermiticity {herm:.2e}")
    return out


def validate_channels(params, kvec=(0.31, -0.17, 0.23), Ev=0.0, atol=1e-14):
    """Assert that the channel decomposition reproduces `kp_pryor.bulk_hamiltonian` exactly.

    This is the one test that ties this module's assembly to the validated transcription. It runs
    in microseconds and is called from `__main__`; run it after touching either file.
    """
    g1, g2, g3 = params['gamma1L'], params['gamma2L'], params['gamma3L']
    kx, ky, kz = kvec
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    P = -Ev + g1 * G0 * k2
    Q = g2 * G0 * (kx ** 2 + ky ** 2 - 2 * kz ** 2)
    R = SQRT3 * G0 * (-g2 * (kx ** 2 - ky ** 2) + 2j * g3 * kx * ky)
    S = 2 * SQRT3 * g3 * G0 * kz * (kx - 1j * ky)

    M = term_matrices(6)
    H = (P * M['P'] + Q * M['Q'] + R.real * M['R_re'] + R.imag * M['R_im']
         + S.real * M['S_re'] + S.imag * M['S_im'] + params['delta_so'] * M['delta'])
    ref = kp.bulk_hamiltonian(kx, ky, kz, params, n_bands=6, Ev=Ev)
    err = float(np.abs(H - ref).max())
    if err > atol:
        raise AssertionError(f"channel decomposition disagrees with kp_pryor by {err:.3e} eV")
    return err


# --------------------------------------------------------------------------------------
# Reciprocal-space grid and cutoff
# --------------------------------------------------------------------------------------

def reciprocal_grid(shape, h):
    """Cartesian components of G on the FFT grid, in 1/nm, matching `np.fft.fftn` ordering."""
    axes = [2.0 * np.pi * np.fft.fftfreq(n, d=h) for n in shape]
    return np.meshgrid(*axes, indexing='ij')


def nyquist(h):
    """The largest |G| the real-space grid can represent along one axis, pi/h."""
    return np.pi / h


def cutoff_mask(shape, h, gmax):
    """Boolean mask of the plane waves inside a sphere of radius `gmax` (1/nm)."""
    Gx, Gy, Gz = reciprocal_grid(shape, h)
    return (Gx ** 2 + Gy ** 2 + Gz ** 2) <= gmax ** 2


def gmax_from_ecut(ecut, mass=1.0):
    """|G| (1/nm) at which the free-carrier kinetic energy reaches `ecut` (eV).

    `mass` scales the prefactor so a cutoff can be quoted against the band that actually needs it:
    for a valence-only problem the LIGHT hole sets the basis requirement, not the heavy one.
    """
    return float(np.sqrt(ecut * mass / G0))


def ecut_from_gmax(gmax, mass=1.0):
    """Inverse of `gmax_from_ecut`."""
    return float(G0 * gmax ** 2 / mass)


# --------------------------------------------------------------------------------------
# Material fields in Fourier space
# --------------------------------------------------------------------------------------

def refine_bandlimited(f, factor=2):
    """Evaluate the trigonometric interpolant of a sampled field on a `factor`x finer grid.

    The plane-wave method does not see the sampled field, it sees this interpolant -- the unique
    band-limited periodic function through the samples. Between grid points it overshoots a step
    by the usual Gibbs amount, and that overshoot is invisible to any check performed on the
    samples themselves. `check_ellipticity` uses this to look at the field the solver actually
    integrates against.

    The zero-frequency bin must land where `ifftshift` expects it, and THAT IS NOT THE MIDDLE OF
    THE PADDED ARRAY for an odd-length axis. `fftshift` puts DC at index s//2; `ifftshift` on the
    padded array reads it from index S//2; so the low pad is `S//2 - s//2`, not `(S - s)//2`. The
    two agree whenever s is even and differ by one whenever s is odd, which made this a bug that
    was invisible on most grids and catastrophic on the rest.

    Measured, interpolating a field built from modes 1-3 (exactly representable, so the
    interpolant is the field itself and the error must be zero):

        n =  16, 18, 32, 10  (even)    max error 1e-15
        n =  17, 19, 33,  7  (odd)     max error 1.5      <- order unity, on a field of range ~2

    An off-by-one shift of the whole spectrum, i.e. multiplication by a linear phase ramp, which
    scrambles the interpolant rather than displacing it. It surfaced through `check_ellipticity`:
    a sweep failed at exactly the boxes with an odd axis ([18,18,17] and [10,10,7]) and passed at
    every all-even one, reporting gamma1 - 2 gamma3 = -1.87 where the true value is +1.55.
    """
    f = np.asarray(f, dtype=float)
    n = f.shape
    N = tuple(int(s * factor) for s in n)
    F = np.fft.fftshift(np.fft.fftn(f, norm='forward'))
    pad = []
    for s, S in zip(n, N):
        lo = S // 2 - s // 2
        pad.append((lo, S - s - lo))
    Fb = np.pad(F, pad)
    return np.fft.ifftn(np.fft.ifftshift(Fb), norm='forward').real


def kinetic_tensor(gamma1, gamma2, gamma3, ordering='symmetrized'):
    """The 18x18 matrix A of the kinetic form, A[(i,b), (j,b')] = A_ij[b, b'].

    The six-band kinetic operator is  sum_ij k_i A_ij(r) k_j, whose quadratic form is
    sum_ij integral (d_i psi)^dagger A_ij (d_j psi). Collecting the direction index i and the band
    index b into one index makes that form  <phi|A|phi>  with phi_(i,b) = d_i psi_b, so whether
    the operator is bounded above is a question about the eigenvalues of this one 18x18 Hermitian
    matrix. `strong_ellipticity` is what asks it.

    Hermitian for either ordering, because A_ij^dagger = A_ji holds in both -- but only the
    symmetrized one has A_ij = A_ji, and the difference is the whole point.
    """
    g = dict(gamma1=gamma1, gamma2=gamma2, gamma3=gamma3)
    A = np.zeros((18, 18), dtype=complex)
    for i, j, name, B in kinetic_blocks(ordering):
        A[6 * i:6 * i + 6, 6 * j:6 * j + 6] += g[name] * B
    return A


def strong_ellipticity(gamma1, gamma2, gamma3, ordering='symmetrized'):
    """lambda_max of the 18x18 kinetic tensor -- POSITIVE means the operator is unbounded above
    at an abrupt heterointerface. This is the central diagnostic in this module.

    There are two different ellipticity conditions and the six-band Luttinger operator satisfies
    only the weaker one.

    **Legendre-Hadamard (rank-one) ellipticity** asks that <phi|A|phi> <= 0 for the rank-one
    directions phi = k (x) v only. That is exactly the statement that every bulk band curves
    downward, and it is what gamma1 - 2 gamma2 > 0 and gamma1 - 2 gamma3 > 0 express. It holds
    comfortably for all the materials here.

    **Strong ellipticity** asks that <phi|A|phi> <= 0 for EVERY 18-vector. It does not hold. In
    these materials A has large positive eigenvalues:

        material   gamma1-2gamma2  gamma1-2gamma3   lambda_max(A)   worst rank-one
        InAs            +3.00           +1.60          +0.937          -0.082
        InSb            +3.80           +1.80          +1.741          -0.102
        GaAs            +2.86           +1.12          +0.226          -0.043
        GaSb            +4.00           +1.40          +0.533          -0.063

    Why that distinction is the whole story
    ---------------------------------------
    With CONSTANT coefficients, Fourier diagonalizes the form and only rank-one directions ever
    appear, so Legendre-Hadamard is enough and the spectrum is bounded by the band edges. The
    same is true for coefficients that vary CONTINUOUSLY, by freezing coefficients locally. But
    at a DISCONTINUOUS interface the argument fails, and the positive directions of A become
    accessible: the form can be made arbitrarily large using fields that oscillate across the
    interface on the scale of the discontinuity.

    So the interface modes that have plagued this project are not a discretization defect at all.
    **They are the true, divergent spectrum of the symmetrized operator with an abrupt
    interface.** Every solver that discretizes that operator correctly must reproduce them, which
    is precisely what was observed: the finite-difference scheme produces them with energy
    ~0.046/h^2, a staggered rewrite produced them, a Q1 finite-element rewrite produced them, and
    the plane-wave method here -- which has no stencil at all -- produces them too. The three
    "failed fixes" were not failures of the fixes.

    lambda_max is +0.23 for GaAs against +1.74 for InSb, an eightfold difference -- but that is a
    difference of DEGREE, not of kind, and it is worth being precise about what it does and does
    not buy. **InAs/GaAs runs away too.** Measured on a 12 nm InAs/GaAs ellipsoid at h = 0.40 with
    symmetrized ordering, growing the plane-wave basis: confinement 13.3 -> -68.5 -> -264.8 meV.
    The Pryor benchmark passes not because GaAs is immune but because it is run on a large island
    at a coarse spacing, where the 0.046/h^2 ceiling is small against the well. Read the eightfold
    ratio as "the antimonides hit the wall at sizes and spacings where InAs/GaAs has not yet",
    never as "the benchmark passing means the method is sound here".

    **gamma3 alone carries it**, which is worth knowing because gamma3 is also the only
    coefficient `kp_confined`'s cross-term operator is ever called with. Dropping one coefficient
    at a time:

                         full      gamma3 = 0    gamma2 = 0    both = 0
        InSb           +1.7412       -0.1448       +0.5601      -1.3259
        InAs           +0.9373       -0.1143       +0.2896      -0.7620

    Setting gamma3 to zero restores strong ellipticity; setting gamma2 to zero does not. So the
    R and S blocks are where the whole problem lives, and they are exactly the blocks Burt-Foreman
    reorders. `scripts/gamma3_diagnostic.py` had already isolated gamma3 by experiment, on a
    hypothesis about stencils that was wrong -- the right parameter for the wrong reason.

    THE REMEDY, AND IT WORKS
    ------------------------
    Burt-Foreman operator ordering (`ordering='burt-foreman'`, see `dkk_kinetic_blocks`) is a
    different OPERATOR, not a different discretization, and it fixes this exactly:

        material   symmetrized   burt-foreman
        GaAs         +0.2259        0.0000
        GaSb         +0.5334        0.0000
        InAs         +0.9373        0.0000
        InSb         +1.7412        0.0000

    Zero, to rounding, for every material -- the tensor becomes negative SEMIdefinite, so the
    operator is bounded above by the local valence edge and `spectrum_bound` becomes a real bound.
    That the answer is exactly zero rather than merely negative suggests an algebraic identity
    behind Foreman's N- = M rather than a numerical coincidence; it is reported as measured.

    Smoothing the coefficients over a finite width (`smooth_fields`) also restores boundedness --
    a continuous coefficient satisfies Garding under Legendre-Hadamard alone -- but the bound then
    depends on the width, so it is a regularization to report as a family, never a fix. Use the
    ordering.
    """
    A = kinetic_tensor(gamma1, gamma2, gamma3, ordering)
    w = np.linalg.eigvalsh(A)
    # Burt-Foreman lands lambda_max on zero to rounding, not below it, so the test has to admit
    # the semidefinite case. The tolerance is scaled by the tensor norm rather than absolute.
    tol = 1e-10 * max(1.0, float(np.abs(w).max()))
    return dict(lambda_max=float(w.max()), lambda_min=float(w.min()), ordering=ordering,
                strongly_elliptic=bool(w.max() <= tol))


def check_ellipticity(fields, refine=2, label='', raise_on_fail=True):
    """Assert gamma1 - 2 gamma2 > 0 and gamma1 - 2 gamma3 > 0 for the BAND-LIMITED fields.

    This is the LEGENDRE-HADAMARD condition -- every bulk band curves downward. It is NECESSARY
    and it is NOT SUFFICIENT: passing it says nothing about whether the spectrum is bounded at a
    heterointerface, which is governed by `strong_ellipticity` and fails for every material here.
    Read the two together and never this one alone. It is kept as a guard because violating it is
    a different and cruder failure: measured, forcing InSb's gamma3 = 16.5 into InAs (gamma1 =
    20.0) returned sixteen eigenvalues at +2.33 eV with 0% of their weight in the island.

    The margins are thin -- gamma1 - 2 gamma3 is +1.8 for InSb (5.2% of gamma1) and +1.6 for InAs
    (8.0%), against 16% for GaAs. Linear averaging across an interface preserves them, but Gibbs
    overshoot on a step need not, which is why this checks the trigonometric interpolant on a
    refined grid rather than the samples, where the inequality holds trivially by construction.

    Returns the two minima and the refinement factor used.
    """
    g1 = refine_bandlimited(fields['gamma1'], refine)
    g2 = refine_bandlimited(fields['gamma2'], refine)
    g3 = refine_bandlimited(fields['gamma3'], refine)
    m12 = float((g1 - 2 * g2).min())
    m13 = float((g1 - 2 * g3).min())
    out = dict(min_g1_2g2=m12, min_g1_2g3=m13, refine=refine,
               elliptic=bool(m12 > 0 and m13 > 0))
    if not out['elliptic'] and raise_on_fail:
        raise ValueError(
            f"non-elliptic band-limited Luttinger fields{' for ' + label if label else ''}: "
            f"min(gamma1-2gamma2) = {m12:+.3f}, min(gamma1-2gamma3) = {m13:+.3f} on a "
            f"{refine}x refined grid. Even the bulk bands no longer curve the right way, so "
            f"nothing about this spectrum is meaningful. Smooth the fields (`smooth_fields`) or "
            f"check the parameter set.")
    return out


def smooth_fields(fields, keys, sigma_nm, h):
    """Convolve selected fields with a Gaussian of width `sigma_nm` NANOMETRES, in Fourier space.

    A REGULARIZATION, NOT A FIX, and the width is in physical units for a reason. The symmetrized
    six-band operator is unbounded above at an abrupt interface because it is only
    Legendre-Hadamard elliptic, not strongly elliptic (`strong_ellipticity`). Grading the
    coefficients over a finite width restores boundedness -- Garding's inequality holds under
    Legendre-Hadamard alone once the coefficients are continuous -- but the bound, and hence the
    answer, depends on the width. So this must be reported as a family over sigma, never as a
    number at one sigma.

    Physical units are what makes the grid-refinement test meaningful. With sigma in nm the
    coefficient field is a FIXED continuum function as h shrinks, so a converging sequence in h
    means the discretization has converged to that operator. With sigma in cells the operator
    would sharpen as the grid refines and nothing could be concluded from the trend -- which is
    exactly the trap the finite-difference solver fell into.

    Only the named keys are smoothed. Band edges and strain enter as local multiplicative terms,
    which are bounded whatever they do at the interface; the unboundedness comes entirely from
    the kinetic coefficients.
    """
    out = dict(fields)
    if sigma_nm <= 0:
        return out
    shape = np.asarray(fields[keys[0]]).shape
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=h) for n in shape]
    KX, KY, KZ = np.meshgrid(*ks, indexing='ij')
    damp = np.exp(-0.5 * sigma_nm ** 2 * (KX ** 2 + KY ** 2 + KZ ** 2))
    for k in keys:
        F = np.fft.fftn(np.asarray(fields[k], dtype=float), norm='forward')
        out[k] = np.fft.ifftn(F * damp, norm='forward').real
    return out


# --------------------------------------------------------------------------------------
# The Hamiltonian
# --------------------------------------------------------------------------------------

class SixBandPlaneWave:
    """Matrix-free six-band valence Hamiltonian in a plane-wave basis on a periodic supercell.

    Built from three kinetic coefficient fields (gamma1, gamma2, gamma3) and a set of LOCAL scalar
    fields that multiply the constant channel matrices: -Ev, delta_so, and the Bir-Pikus Q_eps,
    R_eps, S_eps. Nothing else enters, because the six-band Hamiltonian has no other structure --
    see `kinetic_terms` and `term_matrices`.

    The apply is
        chi_j(r)  = IFFT[ G_j c(G) ]                      for j = x, y, z
        rho_i(r)  = sum_terms  alpha(r) M [ sum_j C_ij chi_j(r) ]
        (Hc)(G)  += G_i FFT[ rho_i ]                      for i = x, y, z
        (Hc)(G)  += FFT[ sum_local  f(r) M psi(r) ]
    which is ~48 FFTs of the real-space grid per matvec -- O(N log N), and cheap enough that the
    whole solve is a Lanczos run on an operator that is never formed.

    THE CUTOFF MUST NOT EXCEED HALF THE NYQUIST WAVENUMBER. Multiplying alpha(r) by chi(r) on the
    grid computes the CYCLIC convolution sum_G' alpha(G - G' mod N) c(G'). Only if every kept pair
    satisfies |G - G'| <= pi/h does the wrap never happen, and only then is the result the exact
    matrix element of the band-limited coefficient field -- which is what makes this a Galerkin
    method with a variational bound. `safe` reports whether the cutoff respects it; exceeding it
    still gives a Hermitian operator, but not one with a clean variational interpretation.
    """

    def __init__(self, alphas, local, shape, h, gmax, ordering='burt-foreman',
                 allow_aliasing=False, workers=_FFT_WORKERS):
        self._fftkw = {} if workers is None else dict(workers=workers, overwrite_x=True)
        self.shape = tuple(int(s) for s in shape)
        self.h = float(h)
        self.gmax = float(gmax)
        self.g_nyquist = min(nyquist(self.h) for _ in self.shape)

        self.safe = self.gmax <= 0.5 * self.g_nyquist + 1e-12
        if not self.safe and not allow_aliasing:
            raise ValueError(
                f"cutoff gmax = {self.gmax:.3f}/nm exceeds half the Nyquist wavenumber "
                f"{0.5*self.g_nyquist:.3f}/nm (h = {self.h} nm). The coefficient convolution "
                f"would wrap and the variational bound would no longer hold. Refine h to "
                f"{np.pi/(2*self.gmax):.3f} nm or lower gmax, or pass allow_aliasing=True and "
                f"say so wherever the number is quoted.")

        self.M = term_matrices(6)
        self.ordering = ordering
        self.alphas = {k: np.ascontiguousarray(alphas[k], dtype=float)
                       for k in ('gamma1', 'gamma2', 'gamma3')}
        self.local = [(np.ascontiguousarray(np.asarray(f), dtype=float), self.M[ch])
                      for ch, f in local]

        # The apply works on a flat (bands, points) view, so each band mixing is one BLAS call on
        # contiguous memory. `kinetic_blocks` already delivers one 6x6 per (direction pair, field),
        # and NOTHING here assumes those blocks are Hermitian or that A_ij = A_ji -- that
        # assumption is exactly what would make Burt-Foreman inexpressible.
        self._local = [(np.asarray(f, dtype=float).reshape(1, -1), M) for f, M in self.local]
        self._kin = [(i, j, self.alphas[name].reshape(1, -1), B)
                     for i, j, name, B in kinetic_blocks(ordering)]

        keep = cutoff_mask(self.shape, self.h, self.gmax)
        self.kept = np.flatnonzero(keep.ravel())
        self.n_pw = int(self.kept.size)
        self.n_bands = 6
        self.dim = self.n_bands * self.n_pw

        Gx, Gy, Gz = reciprocal_grid(self.shape, self.h)
        self.Gvec = [g[None, ...] for g in (Gx, Gy, Gz)]

        # Scratch reused across calls. The apply allocated ~60 MB of temporaries per matvec
        # before this, which at these array sizes cost more than the arithmetic. NOT thread safe;
        # the block solvers here call the apply one column at a time.
        self._S = np.empty((4, self.n_bands) + self.shape, dtype=complex)
        self._R = np.empty((4, self.n_bands) + self.shape, dtype=complex)
        self._w = np.empty((self.n_bands, int(np.prod(self.shape))), dtype=complex)

    # -- basis plumbing ------------------------------------------------------------------
    def _scatter(self, x):
        c = np.zeros((self.n_bands,) + self.shape, dtype=complex)
        c.reshape(self.n_bands, -1)[:, self.kept] = x.reshape(self.n_bands, self.n_pw)
        return c

    def _gather(self, c):
        return c.reshape(self.n_bands, -1)[:, self.kept].ravel()

    def _ifft(self, c):
        return _fftmod.ifftn(c, axes=(-3, -2, -1), norm='forward', **self._fftkw)

    def _fft(self, f):
        return _fftmod.fftn(f, axes=(-3, -2, -1), norm='forward', **self._fftkw)

    # -- the operator --------------------------------------------------------------------
    def matvec(self, x):
        """H c, for one coefficient vector.

        Two batched FFT calls, not fifty. The four transforms the operator needs -- psi itself for
        the local term and G_j psi for each of the three kinetic directions -- are stacked into
        one array and transformed together, and the four outputs likewise. That matters: at the
        grids this method actually needs, per-call numpy overhead and temporaries were most of the
        runtime, not arithmetic.

        Everything after the transform is done on a flat (bands, points) view, which is free
        because the stacked array is C-contiguous with the band axis outermost. Each band mixing
        is then one gemm on contiguous memory.
        """
        c = self._scatter(np.asarray(x, dtype=complex).ravel())

        S, R, w, nb = self._S, self._R, self._w, self.n_bands
        S[0] = c
        for j in range(3):
            np.multiply(self.Gvec[j], c, out=S[j + 1])
        S = self._ifft(S)
        flat = S.reshape(4 * nb, -1)
        psi, chi = flat[:nb], flat[nb:]

        R.fill(0.0)
        Rf = R.reshape(4 * nb, -1)
        for f, M in self._local:
            np.matmul(M, psi, out=w)
            np.multiply(f, w, out=w)
            Rf[:nb] += w
        for i, j, alpha, B in self._kin:
            np.matmul(B, chi[nb * j:nb * (j + 1)], out=w)
            np.multiply(alpha, w, out=w)
            Rf[nb * (i + 1):nb * (i + 2)] += w

        R = self._fft(R)
        out = R[0]
        for i in range(3):
            out += self.Gvec[i] * R[i + 1]
        return self._gather(out)

    def diagonal(self):
        """<G, b| H |G, b> exactly, for every kept plane wave and band.

        Exact because alpha(G - G') at G = G' is just the mean of alpha over the cell, so no
        approximation enters. Used as the preconditioner, where it matters that it distinguishes
        the bands: the heavy and light hole diagonals differ by a factor of order ten in these
        materials, and a preconditioner built from gamma1 alone (the obvious choice, and the one
        tried first here) treats them alike and converges accordingly.

        The per-term contribution is complex when the ordering is asymmetric, since B need not be
        Hermitian then. Only the SUM over (i, j) is real -- A_ji = A_ij^dagger makes the pair
        conjugate -- so the real part is taken at the end, never per term.
        """
        Gk = np.stack([g.ravel()[self.kept]
                       for g in reciprocal_grid(self.shape, self.h)])
        d = np.zeros((self.n_bands, self.n_pw), dtype=complex)
        for i, j, alpha, B in self._kin:
            d += np.diag(B)[:, None] * (float(alpha.mean()) * Gk[i] * Gk[j])[None, :]
        for f, M in self.local:
            d += np.diag(M)[:, None] * float(f.mean())
        imag = float(np.abs(d.imag).max())
        if imag > 1e-9 * max(1.0, float(np.abs(d.real).max())):
            raise AssertionError(f"Hamiltonian diagonal is not real ({imag:.2e}); the kinetic "
                                 f"blocks do not satisfy A_ji = A_ij^dagger")
        return np.ascontiguousarray(d.real).ravel()

    def matmat(self, X):
        """Apply to several vectors at once. LOBPCG works on blocks, and the FFTs batch over the
        extra axis for free, so this is most of the reason the block solver is affordable."""
        X = np.asarray(X, dtype=complex)
        if X.ndim == 1:
            return self.matvec(X)
        return np.column_stack([self.matvec(X[:, j]) for j in range(X.shape[1])])

    def as_linear_operator(self, sign=1.0):
        """`scipy.sparse.linalg.LinearOperator` view. `sign = -1` gives -H, whose SMALLEST
        eigenvalues are the hole ladder -- the form LOBPCG wants."""
        import scipy.sparse.linalg as spla
        s = 1.0 if sign > 0 else -1.0
        return spla.LinearOperator((self.dim, self.dim), dtype=complex,
                                   matvec=lambda v: s * self.matvec(v),
                                   matmat=lambda X: s * self.matmat(X))

    def dense(self):
        """The explicit matrix, by applying to each basis vector. O(dim^2) -- diagnostics only,
        but the only way to check Hermiticity and the bulk limit to machine precision."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        e = np.zeros(self.dim, dtype=complex)
        for j in range(self.dim):
            e[:] = 0.0
            e[j] = 1.0
            H[:, j] = self.matvec(e)
        return H

    def hermiticity(self, n_probe=4, seed=0):
        """max |<phi|H psi> - <H phi|psi>| over random probes, relative to the operator scale.

        Cheap and matrix-free, so it can run on the production-sized problem where `dense` cannot.
        """
        rng = np.random.default_rng(seed)
        worst, scale = 0.0, 0.0
        for _ in range(n_probe):
            a = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
            b = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
            Ha, Hb = self.matvec(a), self.matvec(b)
            worst = max(worst, abs(np.vdot(b, Ha) - np.vdot(Hb, a)))
            scale = max(scale, abs(np.vdot(b, Ha)))
        return worst / max(scale, 1e-300)

    # -- real space ----------------------------------------------------------------------
    def density(self, x):
        """|psi(r)|^2 summed over bands, on the real-space grid, normalized to sum to 1."""
        psi = self._ifft(self._scatter(np.asarray(x, dtype=complex)))
        rho = (np.abs(psi) ** 2).sum(axis=0)
        return rho / rho.sum()



# --------------------------------------------------------------------------------------
# Building one from a heterostructure environment
# --------------------------------------------------------------------------------------

def fields_from_env(env, smooth_nm=0.0):
    """(alphas, local, diagnostics) for `SixBandPlaneWave` from a `heterostructure.build` env.

    Reuses `kp_pryor.material_fields` for the parameter fields and `kp_pryor.bir_pikus_terms` for
    the strain terms, so the physics comes from the same place the finite-difference solver takes
    it from and a disagreement between the two can only be the discretization.

    The local channels are exactly the k = 0 content of the Hamiltonian:

        P     <- -Ev(r)          the band edge, already carrying hydrostatic strain and piezo
        delta <- delta_so(r)
        Q, R, S <- the Bir-Pikus shear terms, which are purely multiplicative and so raise no
                   ordering question at all

    `smooth_nm` tapers ONLY the three Luttinger fields (see `smooth_fields`); the band edges and
    strain are left alone.
    """
    import materials as mt
    m = env['mask']
    fields = kp.material_fields(m, mt.kp_params(env['dot']), mt.kp_params(env['matrix']),
                                env['Ev'], env['Ec'], n_bands=6)
    if smooth_nm:
        fields = smooth_fields(fields, ('gamma1', 'gamma2', 'gamma3'),
                               smooth_nm, env['h'])

    ell = check_ellipticity(fields, refine=2,
                            label=f"{env['dot']['name']} in {env['matrix']['name']}")

    Q_eps, R_eps, S_eps = kp.bir_pikus_terms(env['strain'], fields['b'], fields['d'])
    Q_eps = np.broadcast_to(np.asarray(Q_eps), m.shape)
    R_eps = np.broadcast_to(np.asarray(R_eps, dtype=complex), m.shape)
    S_eps = np.broadcast_to(np.asarray(S_eps, dtype=complex), m.shape)

    alphas = dict(gamma1=fields['gamma1'], gamma2=fields['gamma2'], gamma3=fields['gamma3'])
    local = [('P', -np.asarray(fields['Ev'], dtype=float)),
             ('delta', np.asarray(fields['delta_so'], dtype=float)),
             ('Q', Q_eps.real),
             ('R_re', R_eps.real), ('R_im', R_eps.imag),
             ('S_re', S_eps.real), ('S_im', S_eps.imag)]
    return alphas, local, dict(ellipticity=ell, fields=fields)


def hamiltonian_from_env(env, gmax=None, ecut=None, mass=1.0, smooth_nm=0.0,
                         ordering='burt-foreman', allow_aliasing=False):
    """`SixBandPlaneWave` for an env. Give `gmax` (1/nm) or `ecut` (eV, via `mass`).

    Defaults to the largest cutoff the real-space grid supports without wrap-around,
    gmax = pi/(2h), which is the most basis the material resolution can honestly carry. Refine
    `h` in `heterostructure.build` to go higher.
    """
    h = env['h']
    if gmax is None:
        gmax = gmax_from_ecut(ecut, mass) if ecut is not None else 0.5 * nyquist(h)
    alphas, local, diag = fields_from_env(env, smooth_nm=smooth_nm)
    H = SixBandPlaneWave(alphas, local, env['mask'].shape, h, gmax, ordering=ordering,
                         allow_aliasing=allow_aliasing)
    H.diagnostics = diag
    return H


# --------------------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------------------

def spectrum_bound(H, refine=2):
    """max over r of the local k = 0 valence edge -- the bound that holds ONLY IF the operator is
    strongly elliptic.

    For H = sum_ij k_i A_ij(r) k_j + W(r), if the kinetic form is negative semidefinite then every
    Rayleigh quotient satisfies <psi|H|psi> <= max_r lambda_max(W(r)), and a Galerkin method can
    only produce Rayleigh quotients -- so no eigenvalue could lie above the highest local valence
    edge anywhere in the box.

    **That premise is false for these materials.** Negative semidefiniteness of the kinetic form
    requires STRONG ellipticity, and the six-band Luttinger operator only satisfies the weaker
    Legendre-Hadamard condition (`strong_ellipticity`, and the module docstring for the
    measurement). So this is not a bound on the real problem; it is the bound the problem WOULD
    obey if the ordering were fixed, and comparing the computed spectrum against it is the
    cleanest way to see by how much the operator has run away. Measured on the 2.5 nm island at
    h = 0.20: this returns 0.763 eV while the top of the spectrum reaches 1.951 eV.

    It stays useful in two places. With gamma2 = gamma3 = 0, or with any parameter set that is
    strongly elliptic, it IS a bound and section 1 of `scripts/planewave_validation.py` asserts
    it. And once Burt-Foreman ordering is in place it becomes a bound again, which is the
    acceptance test for that work.

    The `refine` argument matters for a second reason. The method does not see the sampled
    coefficient fields, it sees their trigonometric interpolant, which overshoots each step by the
    usual Gibbs amount between grid points -- measured +86 meV on this structure. Both numbers are
    returned so the two effects are never confused: `gibbs` is tens of meV, the ellipticity
    runaway is over an eV.
    """
    fields_r, fields_s, Ms = [], [], []
    for f, M in H.local:
        fields_r.append(refine_bandlimited(f, refine))
        fields_s.append(f)
        Ms.append(M)

    def edge(fields):
        W = np.zeros(fields[0].shape + (6, 6), dtype=complex)
        for f, M in zip(fields, Ms):
            W += f[..., None, None] * M
        return float(np.linalg.eigvalsh(W)[..., -1].max())

    b_ref, b_smp = edge(fields_r), edge(fields_s)
    return dict(bound=b_ref, bound_sampled=b_smp, gibbs=b_ref - b_smp, refine=refine)


def prolong(H_from, V_from, H_to):
    """Re-express coefficient vectors from one cutoff in the basis of another, same FFT grid.

    Both bases are subsets of the same set of plane waves, so this is a pure index remap: every
    coefficient carries over untouched and the new ones start at zero. No interpolation, so a
    converged vector stays converged in the enlarged basis and only the newly available high-G
    content has to be found.

    This is what makes a cutoff-convergence ladder cheap. Each rung starts from the answer at the
    rung below instead of from noise, which is worth roughly a factor of three in iterations here
    -- and the ladder has to be run anyway, since the convergence trend IS the result.
    """
    if H_from.shape != H_to.shape:
        raise ValueError("prolongation needs the same real-space grid on both sides")
    V_from = np.atleast_2d(np.asarray(V_from))
    if V_from.shape[0] != H_from.dim:
        V_from = V_from.T
    ncol = V_from.shape[1]

    pos = np.full(int(np.prod(H_to.shape)), -1, dtype=np.int64)
    pos[H_to.kept] = np.arange(H_to.n_pw)
    where = pos[H_from.kept]
    keep = where >= 0

    out = np.zeros((H_to.dim, ncol), dtype=complex)
    src = V_from.reshape(H_from.n_bands, H_from.n_pw, ncol)
    dst = out.reshape(H_to.n_bands, H_to.n_pw, ncol)
    dst[:, where[keep], :] = src[:, keep, :]
    return out


def crop_env(env, pad, verbose=False):
    """A copy of `env` on a tighter box, by SUBSETTING the grid -- never by interpolating.

    Strain and the confined state want different boxes. The strain field of an inclusion decays
    as 1/r^3, so it needs generous padding or the periodic images stiffen the island and shift
    every band edge; a hole bound by hundreds of meV decays in under a nanometre and needs almost
    none. Solving both on the strain box wastes most of the plane-wave basis on empty matrix, and
    the cost of that is cubic.

    So: solve the strain once on a well-padded box with `heterostructure.build`, then crop to
    `pad` nm of matrix around the island for the k.p solve. Because the crop is a subset of the
    same co-registered grid, the strain field inside it is bit-identical to the well-padded solve
    -- which is the whole point, and is not true of any scheme that resamples.

    The cropped box is still periodic, which is fine here only because its faces sit in
    essentially unstrained far-field matrix. `verbose` reports how far the fields at the new
    faces are from their far-field values so that assumption is checked rather than assumed.
    """
    import strain_fourier as sf

    # The window comes from the MASK's own bounding box, not from `shape['extent']`. Only
    # `ellipsoid` is centred on z = 0; `lens`, `spherical_lens` and `dash` sit with their base
    # there and extend upward, so a window centred on the origin would cut the island in half.
    m = env['mask']
    if not m.any():
        raise ValueError("empty mask")
    sel = []
    for ax in range(3):
        axis = env[f"c{'xyz'[ax]}"]
        hit = np.flatnonzero(m.any(axis=tuple(i for i in range(3) if i != ax)))
        lo, hi = axis[hit[0]] - pad, axis[hit[-1]] + pad
        idx = np.flatnonzero((axis >= lo - 1e-9) & (axis <= hi + 1e-9))
        if idx.size < 4:
            raise ValueError(f"crop along {'xyz'[ax]} keeps only {idx.size} points; "
                             f"pad is too small")
        sel.append(slice(int(idx[0]), int(idx[-1]) + 1))
    sel = tuple(sel)

    out = dict(env)
    for key in ('mask', 'phi', 'Ec', 'Ev', 'X', 'Y', 'Z'):
        out[key] = np.ascontiguousarray(env[key][sel])
    for ax, s in zip('xyz', sel):
        out[f'c{ax}'] = env[f'c{ax}'][s]
    out['strain'] = sf.StrainTensor(**{k: np.ascontiguousarray(getattr(env['strain'], k)[sel])
                                       for k in sf.StrainTensor.COMPONENTS})
    out['cropped_from'] = env['mask'].shape
    out['pad'] = pad

    if verbose:
        tr = out['strain'].trace
        face = max(abs(float(tr[0].mean())), abs(float(tr[-1].mean())),
                   abs(float(tr[:, 0].mean())), abs(float(tr[:, -1].mean())),
                   abs(float(tr[:, :, 0].mean())), abs(float(tr[:, :, -1].mean())))
        print(f"  cropped {env['mask'].shape} -> {out['mask'].shape} (pad {pad} nm); "
              f"worst face-mean Tr(eps) = {face:.2e} (0 is unstrained far field)", flush=True)
    return out


def _preconditioner(H, shift=0.5):
    """Inverse of the shifted exact diagonal of -H -- the plane-wave preconditioner.

    The spectrum is enormously spread: gamma1 hbar^2 G^2/2m0 reaches tens of eV at the cutoff,
    while the hole levels of interest sit within ~0.1 eV of each other at the very top.
    Unpreconditioned iteration on that extremal end converges at the rate set by the whole spread
    and is unusable; damping by the inverse diagonal flattens it. This is the standard plane-wave
    preconditioner (Payne et al., Rev. Mod. Phys. 64, 1045 (1992)).

    `H.diagonal()` is used rather than the gamma1 kinetic energy alone. That was tried first and
    is much worse: the heavy- and light-hole diagonals differ by a factor of order ten here, so a
    single-mass preconditioner mistreats one of them and the block converges at its rate.

    `shift` (eV) keeps the inverse bounded where the diagonal touches its own minimum; it is of
    order the well depth.
    """
    import scipy.sparse.linalg as spla
    d = -H.diagonal()                       # diagonal of -H; smallest entries are the hole end
    d = 1.0 / (d - d.min() + shift)
    return spla.LinearOperator(
        (H.dim, H.dim), dtype=complex,
        matvec=lambda v: (d * np.asarray(v).ravel()).reshape(np.shape(v)),
        matmat=lambda X: d[:, None] * np.asarray(X))


def _solve_lobpcg(H, k, tol, maxiter, seed, shift=0.5, X0=None):
    """Smallest eigenvalues of -H by preconditioned LOBPCG, returned as eigenvalues of H.

    -H is used rather than H because LOBPCG is a minimization: in the electron convention the
    hole ladder is at the TOP of the valence spectrum, so negating puts it at the bottom where the
    method wants it. Extra columns are carried beyond the `k` requested because LOBPCG converges
    the interior of a block last.

    `X0` is a starting subspace -- from `prolong`, when climbing a cutoff ladder. Columns are
    topped up with noise if there are fewer than the block needs.

    **It thrashes to `maxiter` when the operator has no maximum**, which is the present state with
    the symmetrized ordering: there is nothing to minimise, so it wanders. That is informative but
    it is not a measurement -- use `method='eigsh'` (Lanczos, which always converges to the
    extremal eigenvalue of the finite matrix) whenever the number itself matters. LOBPCG is kept
    as the default because it is several times faster once the operator is bounded, which is the
    regime this module is meant to end up in.
    """
    import scipy.sparse.linalg as spla
    block = max(2, min(H.dim // 3, k + 4))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(H.dim, block)) + 1j * rng.normal(size=(H.dim, block))
    if X0 is not None:
        X0 = np.atleast_2d(np.asarray(X0, dtype=complex))
        n = min(block, X0.shape[1])
        X[:, :n] = X0[:, :n]
    w, V = spla.lobpcg(H.as_linear_operator(sign=-1.0), X, M=_preconditioner(H, shift),
                       tol=tol, maxiter=maxiter, largest=False)
    return -np.asarray(w), np.asarray(V)


def _solve_eigsh(H, k, tol, maxiter):
    import scipy.sparse.linalg as spla
    n_solve = min(max(2 * k, k + 8), H.dim - 2)
    E, V = spla.eigsh(H.as_linear_operator(), k=n_solve, which='LA', tol=tol, maxiter=maxiter)
    return E, V


def six_band_holes_pw(env, k=8, gmax=None, ecut=None, mass=1.0, smooth_nm=0.0,
                      ordering='burt-foreman', method='lobpcg', tol=1e-5, maxiter=800,
                      seed=0, verbose=True, allow_aliasing=False, X0=None, H=None,
                      top_reduce='max', erode=0):
    """Hole states from the six-band Hamiltonian in a plane-wave basis.

    Same signature and same returned dict as `heterostructure.six_band_holes`, so the two are
    interchangeable at the call site and can be compared point by point.

    `ordering` defaults to 'burt-foreman'. With `'symmetrized'` this Hamiltonian has no maximum at
    an abrupt interface (module docstring), so a single solve returns whatever the basis happened
    to reach and NOTHING it returns is quotable.

    **Even with the right ordering, one call is not a result.** Three axes have to be shown to
    converge, and only the first is cheap to forget:

      * `gmax`, the plane-wave cutoff. Monotone from below -- each eigenvalue is a maximum of the
        Rayleigh quotient over the basis, so enlarging the basis can only raise it and confinement
        can only fall. `cutoff_ladder` runs it and warm-starts each rung.
      * `h`, the grid the coefficient fields live on. Independent of gmax, and it caps gmax at
        pi/2h.
      * the k.p BOX. The one that bites: a six-band hole carries a light-hole admixture with a
        3.7 nm decay length even when the state looks deep, so `crop_env` padding has to be run
        out. Measured on the 2.5 nm island, widening it 1.5 -> 2 -> 3 -> 4 nm moved the level by
        -7.3, -5.8, -1.9 meV.

    `n_spurious` counts eigenvalues above the island valence top, the same filter the FD routine
    applies. Under Burt-Foreman it should be zero and a nonzero count means an assumption broke;
    under 'symmetrized' it is a MEASUREMENT of how far the operator has run, not a cleanup -- the
    states it removes are genuine eigenvalues, and the answer below them is contaminated too.

    `top` is the same reference as the FD routine's -- `heterostructure.valence_edge_top` -- so
    confinement energies are on the same footing. `top_reduce='mean'` is the h-independent
    estimator and is the right one for an ellipsoid; `'max'` is the default only because it
    matches the FD routine. On a small dot prefer neither: quote `Ec_far - E`, which is the
    transition energy in these broken-gap systems and never touches `v_top` at all.
    """
    import heterostructure as hs
    t0 = time.time()
    if H is None:
        H = hamiltonian_from_env(env, gmax=gmax, ecut=ecut, mass=mass, smooth_nm=smooth_nm,
                                 ordering=ordering, allow_aliasing=allow_aliasing)
    # Both estimators of the island valence top are carried. `max` is what the finite-difference
    # routine uses and is the safe FILTER (an over-tight bound would discard real states), but it
    # is h-dependent and biased high -- measured here at 0.648 eV at h = 0.40 against 0.677 at
    # h = 0.20 on the same ellipsoid, where Eshelby's theorem says the quantity is exactly
    # h-independent. For an ellipsoid quote confinement from `mean`; see
    # `heterostructure.valence_edge_top`.
    tr = hs.valence_edge_top(env, reduce=top_reduce, erode=erode, report=True)
    top = tr['top']
    top_filter = max(tr['top_max'], tr['top_mean'])

    if verbose:
        print(f"  plane-wave vb: {H.n_pw:,} plane waves x 6 bands = {H.dim:,} unknowns, "
              f"gmax = {H.gmax:.2f}/nm (ecut {ecut_from_gmax(H.gmax):.2f} eV), "
              f"grid {H.shape}, h = {H.h} nm, ordering {H.ordering}", flush=True)
        e = H.diagnostics['ellipticity']
        print(f"    band-limited ellipticity: min(g1-2g2) = {e['min_g1_2g2']:+.3f}, "
              f"min(g1-2g3) = {e['min_g1_2g3']:+.3f}  "
              f"({'ok' if e['elliptic'] else 'FAILS'})", flush=True)

    if method == 'lobpcg':
        E, V = _solve_lobpcg(H, k, tol, maxiter, seed, X0=X0)
    elif method == 'eigsh':
        E, V = _solve_eigsh(H, k, tol, maxiter)
    else:
        raise ValueError(f"method must be 'lobpcg' or 'eigsh', got {method!r}")

    order = np.argsort(-E)
    E, V = E[order], V[:, order]
    E_all, V_all = E.copy(), V.copy()

    m = env['mask']
    inside = np.array([float(H.density(V[:, j])[m].sum()) for j in range(V.shape[1])])

    keep = E <= top_filter + 1e-6
    n_spurious = int((~keep).sum())
    E, V, inside = E[keep][:k], V[:, keep][:, :k], inside[keep][:k]

    if verbose:
        print(f"    {time.time()-t0:.0f}s, island valence top ({top_reduce}) = {top:.4f} eV "
              f"[max {tr['top_max']:.4f}, mean {tr['top_mean']:.4f}, "
              f"interior std {tr['interior_std']*1e3:.1f} meV]", flush=True)
        if n_spurious:
            print(f"    *** {n_spurious} eigenvalue(s) ABOVE the band edge. In this method that "
                  f"is not an artifact to filter -- it means the variational bound was violated. "
                  f"Check ellipticity and aliasing before reading anything below.", flush=True)
        for j, (e, f) in enumerate(zip(E, inside)):
            print(f"    h{j}: E = {e:+.4f} eV, confinement {(top-e)*1e3:6.1f} meV, "
                  f"{f*100:5.1f}% inside the dot", flush=True)
    return dict(E=E, V=V, inside=inside, loc=inside, top=top, n_bands=6,
                n_spurious=n_spurious, seconds=time.time() - t0,
                gmax=H.gmax, n_pw=H.n_pw, dim=H.dim, H=H,
                E_all=E_all, V_all=V_all,
                ellipticity=H.diagnostics['ellipticity'])


def cutoff_ladder(env, gmax_list=None, k=2, smooth_nm=0.0, ordering='burt-foreman',
                  verbose=True, **kw):
    """Solve at a rising sequence of cutoffs, warm-starting each rung from the one below.

    THIS, NOT A SINGLE SOLVE, IS THE RESULT -- and with the symmetrized ordering its job is to
    tell you whether there is a result at all.

    Each eigenvalue is a maximum of the Rayleigh quotient over the basis, so enlarging the basis
    can only raise it and the confinement energies can only FALL. Two outcomes are then
    distinguishable, and no single number distinguishes them:

      * the steps collapse -- the state is converged, and the last step bounds what is left,
        since the remainder is the tail of a monotone sequence;
      * the steps do not collapse -- the operator has no maximum and nothing here is a state.
        Measured on Yeap's 2.5 nm island with the real parameters: 0.457 -> 1.951 eV over
        gmax 3 -> 7.85, steps of +0.12, +0.43, +0.19, +0.38, +0.36 eV. With gamma2, gamma3 scaled
        by 0.25, which restores strong ellipticity and changes nothing else, the same ladder is
        flat to 0.1 meV.

    A non-monotone rung means the eigensolver failed there, not that the physics changed -- warm
    starting from the rung below makes that rare, but check it rather than assume it.

    Defaults to a ladder ending at the largest cutoff the grid supports without wrap-around.

    **The low rungs are not as cheap as their basis size suggests**, and it is worth knowing why
    before optimising the wrong thing. The apply transforms the FULL real-space grid whatever the
    cutoff, so cost is set by `h`, not by `gmax`: measured on the 2.5 nm island at h = 0.25, the
    gmax = 3.77 rung (1,021 plane waves) took 112 s, against 29 s for a comparable basis on a grid
    8x smaller. Resampling the coefficient fields onto a cutoff-matched grid for the early rungs
    would fix it; not implemented.
    """
    gmax_top = 0.5 * nyquist(env['h'])
    if gmax_list is None:
        gmax_list = [f * gmax_top for f in (0.45, 0.6, 0.75, 0.9, 1.0)]

    rows, prev_H, prev_V = [], None, None
    for g in gmax_list:
        H = hamiltonian_from_env(env, gmax=g, smooth_nm=smooth_nm, ordering=ordering)
        X0 = None if prev_H is None else prolong(prev_H, prev_V, H)
        r = six_band_holes_pw(env, k=k, H=H, X0=X0, verbose=False, **kw)
        conf = (r['top'] - r['E'][0]) * 1e3 if len(r['E']) else float('nan')
        rows.append(dict(gmax=g, ecut=ecut_from_gmax(g), n_pw=H.n_pw, dim=H.dim,
                         E=float(r['E'][0]) if len(r['E']) else float('nan'),
                         conf=conf, loc=float(r['inside'][0]) if len(r['inside']) else float('nan'),
                         top=r['top'], n_spurious=r['n_spurious'], seconds=r['seconds']))
        if verbose:
            d = '' if len(rows) < 2 else f"{rows[-1]['conf'] - rows[-2]['conf']:+8.1f}"
            print(f"  {g:6.2f} {ecut_from_gmax(g):7.2f} {H.n_pw:9,} {H.dim:10,} "
                  f"{rows[-1]['E']:+9.4f} {conf:9.1f} {rows[-1]['loc']*100:6.1f}% "
                  f"{r['n_spurious']:>5} {d:>9} {r['seconds']:7.0f}s", flush=True)
        prev_H, prev_V = H, r['V_all']
    return rows


if __name__ == '__main__':
    import materials as mt

    err = validate_channels(mt.kp_params(mt.material('InSb')))
    print(f"channel decomposition vs kp_pryor.bulk_hamiltonian: {err:.2e} eV (must be ~1e-16)")

    # Uniform material: the plane-wave Hamiltonian must be block diagonal in G, each block equal
    # to the bulk matrix at that G. This checks the kinetic assembly, the FFT normalization and
    # the channel matrices together.
    shape, h = (8, 8, 8), 0.6
    p = mt.kp_params(mt.material('InAs'))
    alphas = {k: np.full(shape, p[f'gamma{i}L']) for i, k in
              zip((1, 2, 3), ('gamma1', 'gamma2', 'gamma3'))}
    local = [('P', np.zeros(shape)), ('delta', np.full(shape, p['delta_so'])),
             ('Q', np.zeros(shape)), ('R_re', np.zeros(shape)), ('R_im', np.zeros(shape)),
             ('S_re', np.zeros(shape)), ('S_im', np.zeros(shape))]
    Hpw = SixBandPlaneWave(alphas, local, shape, h, gmax=0.5 * nyquist(h))
    print(f"uniform test: {Hpw.n_pw} plane waves, {Hpw.dim} unknowns, "
          f"hermiticity {Hpw.hermiticity():.2e}")

    Hd = Hpw.dense()
    Gx, Gy, Gz = (g.ravel() for g in reciprocal_grid(shape, h))
    worst = 0.0
    for a, g in enumerate(Hpw.kept):
        ref = kp.bulk_hamiltonian(Gx[g], Gy[g], Gz[g], p, n_bands=6, Ev=0.0)
        blk = Hd[a::Hpw.n_pw, a::Hpw.n_pw]
        worst = max(worst, float(np.abs(blk - ref).max()))
    print(f"bulk limit, worst block error over all {Hpw.n_pw} G vectors: {worst:.2e} eV "
          f"(must be ~1e-15)")

    # With alpha uniform, alpha(G != 0) = 0, so DIFFERENT plane waves must not couple at all.
    off = Hd.copy()
    for a in range(Hpw.n_pw):
        off[a::Hpw.n_pw, a::Hpw.n_pw] = 0.0
    print(f"uniform material, G != G' coupling: {np.abs(off).max():.2e} eV (must be 0)")

    # The variational bound, in the simplest case there is: with Ev = 0 everywhere the local
    # valence edge is 0, and no eigenvalue may exceed it.
    ev = np.linalg.eigvalsh(Hd)
    print(f"max eigenvalue {ev.max():+.3e} eV against the local valence edge 0.0 eV "
          f"(must not exceed it)")

    # A HETEROGENEOUS field, where every part of the assembly is exercised: the exact diagonal
    # against the built matrix, and the variational bound against the true local valence edge.
    q = mt.kp_params(mt.material('InSb'))
    cc = (np.arange(shape[0]) - (shape[0] - 1) / 2) * h
    XX, YY, ZZ = np.meshgrid(cc, cc, cc, indexing='ij')
    isl = ((XX ** 2 + YY ** 2 + ZZ ** 2) < 1.4 ** 2).astype(float)
    ah = {f'gamma{i}': np.where(isl > 0, q[f'gamma{i}L'], p[f'gamma{i}L']) for i in (1, 2, 3)}
    Ev = np.where(isl > 0, 0.59, 0.0)
    lh = [('P', -Ev), ('delta', np.where(isl > 0, q['delta_so'], p['delta_so']))]
    lh += [(ch, np.zeros(shape)) for ch in ('Q', 'R_re', 'R_im', 'S_re', 'S_im')]
    Hh = SixBandPlaneWave(ah, lh, shape, h, gmax=0.5 * nyquist(h))
    Hhd = Hh.dense()
    print(f"heterogeneous: hermiticity of the built matrix "
          f"{np.abs(Hhd - Hhd.conj().T).max():.2e} eV, "
          f"exact diagonal vs built {np.abs(np.diag(Hhd).real - Hh.diagonal()).max():.2e} eV")
    print(f"    max eigenvalue {np.linalg.eigvalsh(Hhd).max():+.4f} eV against the local "
          f"valence edge {Ev.max():.4f} eV (must not exceed it)")
