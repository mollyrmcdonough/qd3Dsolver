"""Four-, six- and eight-band k.p Hamiltonians, transcribed from Pryor's eight-band matrix.

Source: C. Pryor, Phys. Rev. B 57, 7190 (1998), preprint arXiv:cond-mat/9710304 -- the
strain-dependent eight-band k.p Hamiltonian H_k of his Sec. II, together with the definitions
of A, P, Q, R, S, U, V and the modified Luttinger parameters that follow it. The preprint was
fetched and the matrix read off directly, so this transcription has a verifiable provenance
chain; per the citation policy in qdsolver_core.py the equation numbers are still not quoted,
since the preprint's numbering need not match the published version's.

Why this module replaces kp_luttinger.py / kp_confined.py's matrix
-----------------------------------------------------------------
The six-band matrix previously used here failed two checks that the earlier bulk validation
was not sensitive to:

1. Split-off placement. It put the SO band at -delta on the hole-convention diagonal; a hole
   in the split-off band costs delta MORE energy than one in HH/LH, so it belongs at +delta.
   With the wrong sign the split-off states are the algebraically lowest eigenvalues of the
   hole Hamiltonian, so any "lowest states" search returns spurious SO states instead of the
   hole ground state -- observed as bound-state energies ~300 meV below the potential floor.
2. Off-diagonal placement. In the spherical approximation (gamma2 = gamma3) the cubic warping
   vanishes and the valence bands must be exactly isotropic. The old matrix gave 21 meV of
   anisotropy at |k| = 0.3/nm; the matrix below gives 1e-16 eV. The two matrices have
   identical diagonals and identical off-diagonal *magnitudes* (Tr H and Tr H^2 agree to
   machine precision) but differ in Tr H^3 -- the elements were in the wrong places.

The earlier validation only checked effective masses along [001], where R and S vanish by
symmetry, which is why a wrong R/S arrangement passed it. The isotropy test in the spherical
approximation is the cheap check that catches this class of error, and it is the one to run
on any k.p matrix before trusting it.

Band ordering
-------------
Pryor's basis, read off the diagonal (at k along [001], -P+Q is the light hole and -P-Q the
heavy hole):

    [ CB, CB, LH, HH, HH, LH, SO, SO ]

Sub-models are contiguous slices of that ordering, which is what makes one transcription serve
all three:
  - 8-band: the whole matrix (conduction + valence, coupled through P0).
  - 6-band: rows/columns 2..7, the valence block. Exactly the P0 -> 0 limit of the 8-band
    valence sub-block, i.e. the six-band Luttinger-Kohn Hamiltonian.
  - 4-band: rows/columns 2..5, dropping the split-off pair. Per Pryor's Sec. VI this is the
    physically sensible intermediate model for InAs/GaAs holes: four- and eight-band ground
    states agree to within 3 meV, while six-band differs from both by ~40 meV, because InAs
    has Delta ~ E_g so including the split-off band while leaving the conduction band
    decoupled is inconsistent.

Sign convention
---------------
The matrix is assembled in Pryor's ELECTRON-energy convention (valence bands curve downward;
E_c and E_v are the unstrained band edges on a common scale). `hole_convention=True` negates
the valence-only models (4- and 6-band) so their eigenvalues are hole energies increasing
downward, matching the rest of this package. The 8-band model must NOT be negated -- it holds
electron and hole states in one spectrum, and its bound states are interior eigenvalues.
"""
import numpy as np
import scipy.sparse as sp

from qdsolver_core import HBAR2_OVER_2M0 as G

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
SQRT32 = np.sqrt(1.5)

BAND_LABELS_8 = ['CB+1/2', 'CB-1/2', 'LH+1/2', 'HH+3/2', 'HH-3/2', 'LH-1/2', 'SO+1/2', 'SO-1/2']
BAND_LABELS_6 = BAND_LABELS_8[2:]
BAND_LABELS_4 = BAND_LABELS_8[2:6]

BAND_SLICES = {8: slice(0, 8), 6: slice(2, 8), 4: slice(2, 6)}
BAND_LABELS = {8: BAND_LABELS_8, 6: BAND_LABELS_6, 4: BAND_LABELS_4}


def bir_pikus_terms(strain, b, d):
    """Bir-Pikus SHEAR strain contributions (Q_eps, R_eps, S_eps) to add to the kinetic Q, R, S.

    `strain` is a strain_fourier.StrainTensor (arrays for the confined problem, scalars for
    bulk); `b` and `d` are the shear deformation potentials, per grid point or scalar.

        Q_eps = -(b/2) (exx + eyy - 2 ezz)
        R_eps = (sqrt(3)/2) b (exx - eyy) - i d exy
        S_eps = -d (ezx - i eyz)

    These are purely local (multiplicative) operators, so unlike the kinetic terms they raise no
    operator-ordering question at all where b and d jump across the interface.

    Why these three formulas can be trusted
    ---------------------------------------
    They are fixed by the standard Bir-Pikus correspondence between the strain Hamiltonian and
    the kinetic one -- the two have identical structure under eps_ij <-> k_i k_j. Matching term
    by term against the kinetic Q, R and S already transcribed in this module gives

        G gamma2  <->  -b/2          2 sqrt(3) G gamma3  <->  -d

    and, crucially, the SAME two substitutions reproduce all three of Q, R and S. That is an
    over-determined consistency check: three complex expressions, two free constants. It is run
    as an exact numerical test in strain_validation.py by setting eps_ij = k_i k_j with
    b = -2 G gamma2 and d = -2 sqrt(3) G gamma3, which must return the kinetic matrix exactly.

    The overall sign is fixed independently by physics rather than by that correspondence: for
    compressive biaxial strain the heavy hole must be the topmost valence band. With HH on the
    -P-Q diagonal and InAs-on-GaAs strain (exx = eyy = -0.0669, ezz = +0.0605, b = -1.8 eV),
    Q_eps = -0.229 eV, so HH shifts by -Q_eps = +0.229 eV and LH by -0.229 eV. HH on top. ✓

    Reference: G. L. Bir and G. E. Pikus, "Symmetry and Strain-Induced Effects in
    Semiconductors" (Wiley, 1974) -- the strain-dependent valence-band Hamiltonian and the
    definitions of the a, b, d deformation potentials.

    Tensor (not engineering) shear convention, matching strain_fourier.StrainTensor.
    """
    Q_eps = -(b / 2.0) * (strain.exx + strain.eyy - 2.0 * strain.ezz)
    R_eps = (SQRT3 / 2.0) * b * (strain.exx - strain.eyy) - 1j * d * strain.exy
    S_eps = -d * (strain.ezx - 1j * strain.eyz)
    return Q_eps, R_eps, S_eps


def interband_strain_scalars(strain, P0, kx, ky, kz):
    """Pryor's strain-dependent CONDUCTION-VALENCE coupling terms u and v (bulk, scalar k).

    These are the strain analogues of the kinetic U and V: where U and V are P0 times k, u and
    v are P0 times (strain . k). Pryor writes them with derivatives,

        u = (-i/sqrt(3)) P0 sum_j e_zj d_j        v = (-i/sqrt(6)) P0 sum_j (e_xj - i e_yj) d_j

    and substituting d_j -> i k_j gives what is returned here:

        u = P0 (e_zx kx + e_zy ky + e_zz kz) / sqrt(3)
        v = P0 [(e_xx - i e_xy) kx + (e_xy - i e_yy) ky + (e_zx - i e_yz) kz] / sqrt(6)

    How they enter the matrix
    -------------------------
    Comparing Pryor's H_s against his H_k slot by slot, every conduction-row entry of H_s is
    the corresponding entry of H_k with U -> -u and V -> -v. So the full Hamiltonian H_k + H_s
    is obtained by the single substitution

        U -> U - u ,   V -> V - v

    which is how `bulk_hamiltonian` and `confined_hamiltonian` apply them. That pattern was
    checked against all ten conduction-row couplings of the two matrices, not inferred from
    one; it holds in every slot.

    Because they are linear in k they vanish at k = 0, which is why Pryor can say the k = 0
    band structure of his Fig. 2 "reduces to a six-band model with a decoupled conduction
    band" -- and why `local_band_edges` need not know about them.

    Eight-band only: with no explicit conduction band there is nothing for them to couple to.
    """
    e = strain
    u = P0 * (e.ezx * kx + e.eyz * ky + e.ezz * kz) / SQRT3
    v = P0 * ((e.exx - 1j * e.exy) * kx
              + (e.exy - 1j * e.eyy) * ky
              + (e.ezx - 1j * e.eyz) * kz) / np.sqrt(6.0)
    return u, v


def hole_sigma(Ev_field, strain, b_field, d_field, delta_so_field, inside_mask=None,
               margin=0.0, hydrostatic_applied=True):
    """A shift-invert target sigma that sits AT the top of the local valence band.

    Not from any paper -- this is a numerical-methods helper for this package, and it exists
    because of a specific failure that is easy to miss. A folded-spectrum or shift-invert solve
    returns the eigenvalues NEAREST sigma; it does not return the extremal ones, and its
    residual certifies only that what came back is an eigenpair, not that it is the one wanted
    (Parlett, "The Symmetric Eigenvalue Problem"). So a sigma placed below the states of
    interest yields converged, certified, WRONG answers with no warning.

    That is not hypothetical here. Before the Bir-Pikus terms existed, the band-edge field V_h
    from pryor1998.band_edge_fields WAS the top of the valence band, so `V_h.max()` was a
    correct sigma. Adding shear strain silently invalidated it: the shear raises the local
    valence edge well above V_h (by ~145 meV in the b = 14 nm pyramid), so eight-band hole
    solves seeded that way returned states ~55 meV below the true ground state.

    The edge is computed EXACTLY, as the largest k = 0 eigenvalue of the valence block
    (`local_band_edges`). An earlier version used the heavy-hole diagonal element alone,
    E_v - Q_eps, which is not the edge: unlike the kinetic R and S, the strain terms R_eps and
    S_eps do NOT vanish at k = 0, so they mix the levels and push the true edge higher. That
    approximation errs in the dangerous direction -- too low -- which is the same failure mode
    it was written to prevent.

    `margin` is added to the result. Even with the exact edge, ALWAYS confirm a hole solve by
    re-running with sigma raised further and checking the spectrum is unchanged: the edge
    bounds where the states are, but the solver still has to find them.
    """
    edges = local_band_edges(strain, Ev=Ev_field, Ec=np.zeros_like(np.asarray(Ev_field)),
                             delta_so=delta_so_field, a_c=0.0, a_v=0.0,
                             b=b_field, d=d_field,
                             hydrostatic_applied=hydrostatic_applied)
    v1 = edges['v1']
    if inside_mask is not None:
        v1 = v1[inside_mask]
    return float(np.max(v1)) + margin


def modified_luttinger(gamma1L, gamma2L, gamma3L, Eg, delta_so, Ep):
    """The modified Luttinger parameters for an EIGHT-band calculation.

    In an eight-band model the conduction band is explicit, so its contribution has to be
    removed from the Luttinger parameters, which were defined to include it perturbatively.
    Pryor gives gamma1 = gamma1L - Ep/(3Eg + Delta), and gamma2, gamma3 = gamma2L, gamma3L
    minus half that same quantity.

    Only for the 8-band model: the 4- and 6-band models keep the unmodified gamma^L, because
    there the conduction band really is absent and its contribution must stay folded in.
    """
    shift = Ep / (3 * Eg + delta_so)
    return gamma1L - shift, gamma2L - 0.5 * shift, gamma3L - 0.5 * shift


def kane_P0(Ep):
    """Kane interband momentum matrix element P0 (eV*nm) from Ep (eV), via Ep = 2 m0 P0^2/hbar^2
    (stated directly after Pryor's definitions of the modified Luttinger parameters), i.e.
    P0 = sqrt(Ep * hbar^2/(2 m0))."""
    return np.sqrt(Ep * G)


def _assemble(blocks8, n_bands, adjoint, zero, negate=False):
    """Take the upper triangle of Pryor's 8x8 as a nested list, slice out the requested
    sub-model, and Hermitian-complete it.

    Writing only the upper triangle and completing by adjoint means Hermiticity is guaranteed
    by construction and can never mask a transcription error -- a wrong element shows up as a
    wrong eigenvalue instead (the same "Hermiticity from construction, physics from validation"
    split used throughout this package).
    """
    sl = BAND_SLICES[n_bands]
    idx = list(range(8))[sl]
    m = len(idx)

    out = [[None] * m for _ in range(m)]
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            if j >= i:
                blk = blocks8[i][j]
                out[a][b] = zero if blk is None else blk
    for a in range(m):
        for b in range(a):
            out[a][b] = adjoint(out[b][a])

    if negate:
        out = [[-out[a][b] for b in range(m)] for a in range(m)]
    return out


def _blocks(A, P, Q, R, S, U, V, delta, adj):
    """Pryor's 8x8 upper triangle. `adj` is complex conjugation for bulk (scalars) or the
    conjugate transpose for the discretized operators; U is Hermitian so U* == U either way."""
    B = [[None] * 8 for _ in range(8)]

    B[0][0] = A
    B[0][1] = None
    B[0][2] = adj(V)
    B[0][3] = None
    # NOTE: read as sqrt(3)*V* off the preprint's rendered matrix, but that makes the
    # eight-band spectrum anisotropic in the spherical approximation (2e-2 eV at |k|=0.3/nm)
    # while every other element checks out. Enumerating all 65536 sign/conjugation
    # arrangements of the ten conduction-row couplings gives 24 that are exactly isotropic
    # (they differ only by basis phase conventions, i.e. they are the same Hamiltonian), and
    # one of them agrees with the reading here in every slot except this one, where it
    # requires V rather than V*. Isotropy is a physical constraint no wrong arrangement can
    # satisfy accidentally, so V is used. Most likely a misread of a small rendered glyph
    # rather than an error in the paper -- flagged rather than silently "corrected".
    B[0][4] = SQRT3 * V
    B[0][5] = -SQRT2 * U
    B[0][6] = -U
    B[0][7] = SQRT2 * adj(V)

    B[1][1] = A
    B[1][2] = -SQRT2 * U
    B[1][3] = -SQRT3 * adj(V)
    B[1][4] = None
    B[1][5] = -V
    B[1][6] = SQRT2 * V
    B[1][7] = U

    B[2][2] = -P + Q
    B[2][3] = -adj(S)
    B[2][4] = R
    B[2][5] = None
    B[2][6] = SQRT32 * S
    B[2][7] = -SQRT2 * Q

    B[3][3] = -P - Q
    B[3][4] = None
    B[3][5] = R
    B[3][6] = -SQRT2 * R
    B[3][7] = S / SQRT2

    B[4][4] = -P - Q
    B[4][5] = adj(S)
    B[4][6] = adj(S) / SQRT2
    B[4][7] = SQRT2 * adj(R)

    B[5][5] = -P + Q
    B[5][6] = SQRT2 * Q
    B[5][7] = SQRT32 * adj(S)

    B[6][6] = -P - delta
    B[6][7] = None

    B[7][7] = -P - delta
    return B


# ---------------------------------------------------------------------------------------
# Bulk (k is a number)
# ---------------------------------------------------------------------------------------

def bulk_hamiltonian(kx, ky, kz, params, n_bands=8, hole_convention=False,
                     Ec=None, Ev=0.0, strain=None, include_hydrostatic=False,
                     interband_strain=True):
    """Bulk k.p Hamiltonian (eV) at wavevector (kx, ky, kz) in 1/nm.

    `params` is an entry of pryor1998.PRYOR_TABLE_I (or anything with the same keys). For
    n_bands = 8 the modified Luttinger parameters and P0 are used; for 4 and 6 the unmodified
    gamma^L are used and the conduction coupling is absent.

    `strain`, if given, is a strain_fourier.StrainTensor of scalars; its shear part enters
    through bir_pikus_terms. See confined_hamiltonian for what include_hydrostatic does and
    why it defaults to False.
    """
    Eg, delta, Ep = params['Eg'], params['delta_so'], params['Ep']
    if Ec is None:
        Ec = Ev + Eg

    if n_bands == 8:
        g1, g2, g3 = modified_luttinger(params['gamma1L'], params['gamma2L'],
                                        params['gamma3L'], Eg, delta, Ep)
        P0 = kane_P0(Ep)
    else:
        g1, g2, g3 = params['gamma1L'], params['gamma2L'], params['gamma3L']
        P0 = 0.0

    k2 = kx**2 + ky**2 + kz**2
    A = Ec + G * k2
    P = -Ev + g1 * G * k2
    Q = g2 * G * (kx**2 + ky**2 - 2 * kz**2)
    R = SQRT3 * G * (-g2 * (kx**2 - ky**2) + 2j * g3 * kx * ky)
    S = 2 * SQRT3 * g3 * G * kz * (kx - 1j * ky)
    U = P0 * kz / SQRT3
    V = P0 * (kx - 1j * ky) / np.sqrt(6.0)

    if strain is not None:
        Q_eps, R_eps, S_eps = bir_pikus_terms(strain, params['b'], params['d'])
        Q, R, S = Q + Q_eps, R + R_eps, S + S_eps
        if include_hydrostatic:
            tr = strain.trace
            A = A + params['a_c'] * tr
            P = P + params['a_v'] * tr
        if interband_strain and n_bands == 8:
            u_s, v_s = interband_strain_scalars(strain, P0, kx, ky, kz)
            U, V = U - u_s, V - v_s

    blocks = _blocks(A, P, Q, R, S, U, V, delta, np.conj)
    negate = hole_convention and n_bands != 8
    rows = _assemble(blocks, n_bands, np.conj, 0.0, negate=negate)
    return np.array(rows, dtype=complex)


# ---------------------------------------------------------------------------------------
# Confined (k -> operator on a grid)
# ---------------------------------------------------------------------------------------

def confined_hamiltonian(ops, fields, n_bands=8, hole_convention=False, strain=None,
                         include_hydrostatic=False, interband_strain=True):
    """Discretized k.p Hamiltonian (eV) as a sparse matrix.

    `ops` is a kp_confined.GridOperators for the grid; `fields` is a dict of per-grid-point
    arrays with keys gamma1, gamma2, gamma3, delta_so, Ev, Ec and (8-band only) P0.

    Strain
    ------
    Pass `strain` (a strain_fourier.StrainTensor of grid-shaped arrays) to include the
    Bir-Pikus shear terms; `fields` must then also carry the b and d deformation potentials,
    which material_fields supplies. These are the terms that were missing entirely under the
    hydrostatic-only strain model, and Pryor's Sec. VI attributes hole confinement mainly to
    them.

    `include_hydrostatic` defaults to False on purpose, and this is a double-counting trap
    worth stating plainly: the standard path builds Ev/Ec with pryor1998.band_edge_fields,
    which has ALREADY applied a_c Tr(eps) and -a_v Tr(eps) to the band edges. Since P is built
    as -Ev + kinetic, the hydrostatic strain is therefore already in the Hamiltonian, and
    setting include_hydrostatic=True on top of that applies it twice. Set it True only when
    Ev/Ec are the UNSTRAINED band edges.

    For n_bands = 8, `interband_strain` additionally includes Pryor's strain-dependent
    conduction-valence coupling u and v (see `interband_strain_scalars`), via U -> U - u and
    V -> V - v. They are linear in k, so they have no effect on k = 0 quantities such as
    `local_band_edges`; set the flag False to isolate their contribution to a bound state.

    Still not included: the strain renormalization of P0 itself, which is conventionally
    dropped and is dropped here.

    Reuses the discretization already validated in kp_confined.py: the direct three-point
    Ben Daniel-Duke stencil for the k_i^2 terms and symmetrized Hermitian central differences
    for the cross terms k_i k_j. The 8-band model additionally needs FIRST-order terms (U and
    V are linear in k), which use the same Hermitian central-difference operators.

    The operator-ordering caveat from kp_confined.py's docstring applies unchanged: plain
    symmetrization is used where the material parameters vary, which is Hermitian but is not
    the Burt-Foreman envelope-function ordering (Burt, J. Phys.: Condens. Matter 4, 6651
    (1992); Foreman, Phys. Rev. B 48, 4964 (1993)).
    """
    g1, g2, g3 = fields['gamma1'], fields['gamma2'], fields['gamma3']
    delta_f, Ev_f, Ec_f = fields['delta_so'], fields['Ev'], fields['Ec']

    n = int(np.prod(ops.shape))
    I = sp.identity(n, dtype=complex, format='csr')
    ones = np.ones(ops.shape)

    A = sp.diags(Ec_f.ravel()).astype(complex) + G * (ops.k2(ones, 0) + ops.k2(ones, 1)
                                                      + ops.k2(ones, 2))
    P = -sp.diags(Ev_f.ravel()).astype(complex) + G * (ops.k2(g1, 0) + ops.k2(g1, 1)
                                                       + ops.k2(g1, 2))
    Q = G * (ops.k2(g2, 0) + ops.k2(g2, 1) - 2 * ops.k2(g2, 2))
    R = G * (-SQRT3 * (ops.k2(g2, 0) - ops.k2(g2, 1)) + 2j * SQRT3 * ops.cross(g3, 0, 1))
    S = 2 * SQRT3 * G * (ops.cross(g3, 2, 0) - 1j * ops.cross(g3, 2, 1))
    delta = sp.diags(delta_f.ravel()).astype(complex)

    if strain is not None:
        Q_eps, R_eps, S_eps = bir_pikus_terms(strain, fields['b'], fields['d'])
        # Local multiplicative operators -- diagonal in real space, so no ordering ambiguity.
        Q = Q + sp.diags(np.asarray(Q_eps).ravel()).astype(complex)
        R = R + sp.diags(np.asarray(R_eps).ravel().astype(complex))
        S = S + sp.diags(np.asarray(S_eps).ravel().astype(complex))
        if include_hydrostatic:
            tr = strain.trace
            A = A + sp.diags((fields['a_c'] * tr).ravel()).astype(complex)
            P = P + sp.diags((fields['a_v'] * tr).ravel()).astype(complex)

    if n_bands == 8:
        P0 = fields['P0']
        P0d = sp.diags(P0.ravel()).astype(complex)
        # Symmetrized so each stays Hermitian (U) / consistently adjointed (V) where P0 varies.
        def sym(D):
            return (P0d @ D + D @ P0d) / 2.0
        U = sym(ops.Dz) / SQRT3
        V = (sym(ops.Dx) - 1j * sym(ops.Dy)) / np.sqrt(6.0)

        if strain is not None and interband_strain:
            # Pryor's u and v: the same P0*k structure, but contracted with the strain tensor.
            # Each coefficient P0*e_ij now varies from point to point, so every term is
            # symmetrized the same way `sym` handles a varying P0 -- Hermitian, though (as in
            # the module docstring) not the Burt-Foreman ordering.
            def symf(field, D):
                Fd = sp.diags(np.asarray(field).ravel().astype(complex))
                return (Fd @ D + D @ Fd) / 2.0

            e = strain
            u_s = (symf(P0 * e.ezx, ops.Dx) + symf(P0 * e.eyz, ops.Dy)
                   + symf(P0 * e.ezz, ops.Dz)) / SQRT3
            v_s = (symf(P0 * (e.exx - 1j * e.exy), ops.Dx)
                   + symf(P0 * (e.exy - 1j * e.eyy), ops.Dy)
                   + symf(P0 * (e.ezx - 1j * e.eyz), ops.Dz)) / np.sqrt(6.0)
            U, V = U - u_s, V - v_s
    else:
        U = sp.csr_matrix((n, n), dtype=complex)
        V = sp.csr_matrix((n, n), dtype=complex)

    def adj(M):
        return M.conj().T

    zero = sp.csr_matrix((n, n), dtype=complex)
    blocks = _blocks(A, P, Q, R, S, U, V, delta, adj)
    negate = hole_convention and n_bands != 8
    rows = _assemble(blocks, n_bands, adj, zero, negate=negate)
    return sp.bmat(rows, format='csr')


def local_band_edges(strain, Ev, Ec, delta_so, a_c, a_v, b, d, hydrostatic_applied=False):
    """Band energies from the LOCAL value of the strain -- the eigenvalues of H at k = 0.

    This is the construction behind Pryor's Fig. 2 ("band structure based on the local value of
    the strain"), which his Sec. IV defines as the eigenvalues of his strain Hamiltonian H_s
    taken with k = 0. Since the conduction-valence coupling (U, V and the strain terms u, v) is
    linear in k, at k = 0 the eight-band model collapses to a decoupled conduction band plus a
    six-band valence problem -- Pryor says exactly this. The conduction band is therefore
    returned in closed form and only the 6x6 valence block is diagonalized, which avoids having
    to assume anything about eigenvalue ordering between the two.

    Every argument is broadcast against the others, so scalars and grid-shaped arrays mix
    freely.

    Parameters
    ----------
    strain : strain_fourier.StrainTensor
        The ELASTIC strain, which is what the deformation potentials act on.
    Ev, Ec : array or float
        Valence and conduction band edges. By default these are taken to be the UNSTRAINED
        edges and the hydrostatic shift is applied here; pass `hydrostatic_applied=True` if
        they already carry it (as pryor1998.band_edge_fields returns them) so it is not
        counted twice. Both routes are checked against each other in strain_validation.py.
    a_c, a_v : array or float
        Hydrostatic deformation potentials in PRYOR's sign convention: the conduction edge
        shifts by +a_c*Tr(eps) and the valence edge by -a_v*Tr(eps). See pryor1998's module
        docstring -- this is not the Van de Walle convention.
    b, d : array or float
        Shear deformation potentials, fed to `bir_pikus_terms`.

    Returns
    -------
    dict with keys 'cb', 'v1', 'v2', 'v3' (arrays of the broadcast shape) and 'kramers'.
    'v1' >= 'v2' >= 'v3' are the three doubly-degenerate valence levels, so 'v1' is the local
    valence-band edge; away from high-symmetry points the shear terms mix HH and LH, so they
    are numbered rather than labelled (use `local_band_character` for the mixing). 'kramers'
    is the largest splitting within a pair that should be exactly degenerate -- a free
    correctness check, since Kramers degeneracy holds at k = 0 for any strain.
    """
    Q_eps, R_eps, S_eps = bir_pikus_terms(strain, b, d)
    tr = strain.trace

    Ev_s = np.asarray(Ev, dtype=float) if hydrostatic_applied else np.asarray(Ev) - a_v * tr
    Ec_s = np.asarray(Ec, dtype=float) if hydrostatic_applied else np.asarray(Ec) + a_c * tr

    # P enters the matrix as -P on the valence diagonal, so P = -Ev_s reproduces Pryor's -p.
    P, A = -Ev_s, Ec_s
    shape = np.broadcast_shapes(np.shape(Q_eps), np.shape(R_eps), np.shape(S_eps),
                                np.shape(P), np.shape(np.asarray(delta_so)))

    def bc(x):
        return np.broadcast_to(np.asarray(x, dtype=complex), shape)

    blocks = _blocks(bc(A), bc(P), bc(Q_eps), bc(R_eps), bc(S_eps),
                     0.0, 0.0, bc(delta_so), np.conj)
    rows = _assemble(blocks, 6, np.conj, bc(0.0))
    Hv = np.stack([np.stack([bc(c) for c in row], axis=-1) for row in rows], axis=-2)

    ev = np.linalg.eigvalsh(Hv)          # ascending, shape (..., 6)
    kramers = float(np.max([np.abs(ev[..., 2 * i + 1] - ev[..., 2 * i]).max() for i in range(3)]))

    return dict(cb=np.broadcast_to(Ec_s, shape).copy(),
                v1=ev[..., 5], v2=ev[..., 3], v3=ev[..., 1], kramers=kramers)


def local_band_character(strain, Ev, delta_so, a_v, b, d, hydrostatic_applied=False):
    """Basis-state weights of the three local valence levels, for labelling HH vs LH vs SO.

    Returns an array of shape (..., 3, 3): for each level (v1, v2, v3, in that order) the
    fraction of its weight on the HH, LH and SO basis pairs. Pure states give (1,0,0) etc.;
    the shear terms r and s mix them, and Pryor's Fig. 2 shows a band crossing along [001]
    where the ordering of the mixed levels changes.
    """
    Q_eps, R_eps, S_eps = bir_pikus_terms(strain, b, d)
    Ev_s = np.asarray(Ev, dtype=float) if hydrostatic_applied else np.asarray(Ev) - a_v * strain.trace
    shape = np.broadcast_shapes(np.shape(Q_eps), np.shape(R_eps), np.shape(S_eps),
                                np.shape(Ev_s), np.shape(np.asarray(delta_so)))

    def bc(x):
        return np.broadcast_to(np.asarray(x, dtype=complex), shape)

    blocks = _blocks(bc(0.0), bc(-Ev_s), bc(Q_eps), bc(R_eps), bc(S_eps),
                     0.0, 0.0, bc(delta_so), np.conj)
    rows = _assemble(blocks, 6, np.conj, bc(0.0))
    Hv = np.stack([np.stack([bc(c) for c in row], axis=-1) for row in rows], axis=-2)

    _, vec = np.linalg.eigh(Hv)
    w = np.abs(vec) ** 2                                    # (..., basis, level)
    # BAND_LABELS_6 order is [LH, HH, HH, LH, SO, SO]; group into HH / LH / SO pairs.
    grouped = np.stack([w[..., 1, :] + w[..., 2, :],        # HH
                        w[..., 0, :] + w[..., 3, :],        # LH
                        w[..., 4, :] + w[..., 5, :]], -2)   # SO
    return np.stack([grouped[..., 5], grouped[..., 3], grouped[..., 1]], axis=-1).swapaxes(-1, -2)


def material_fields(inside_mask, dot, matrix, Ev_field, Ec_field, n_bands=8):
    """Build the per-grid-point `fields` dict for `confined_hamiltonian` from two entries of
    pryor1998.PRYOR_TABLE_I plus the already-strain-shifted band-edge fields.

    Ev_field/Ec_field carry the band offsets AND the hydrostatic strain shift, so they must be
    built with pryor1998.band_edge_fields (which uses Pryor's a_v sign convention).
    """
    def pick(key):
        return np.where(inside_mask, dot[key], matrix[key])

    if n_bands == 8:
        g1d, g2d, g3d = modified_luttinger(dot['gamma1L'], dot['gamma2L'], dot['gamma3L'],
                                           dot['Eg'], dot['delta_so'], dot['Ep'])
        g1m, g2m, g3m = modified_luttinger(matrix['gamma1L'], matrix['gamma2L'],
                                           matrix['gamma3L'], matrix['Eg'],
                                           matrix['delta_so'], matrix['Ep'])
        g1 = np.where(inside_mask, g1d, g1m)
        g2 = np.where(inside_mask, g2d, g2m)
        g3 = np.where(inside_mask, g3d, g3m)
        P0 = np.where(inside_mask, kane_P0(dot['Ep']), kane_P0(matrix['Ep']))
    else:
        g1, g2, g3 = pick('gamma1L'), pick('gamma2L'), pick('gamma3L')
        P0 = None

    fields = dict(gamma1=g1, gamma2=g2, gamma3=g3, delta_so=pick('delta_so'),
                  Ev=Ev_field, Ec=Ec_field,
                  b=pick('b'), d=pick('d'), a_c=pick('a_c'), a_v=pick('a_v'))
    if P0 is not None:
        fields['P0'] = P0
    return fields
