"""Reproduce Pryor's Fig. 4: bound-state energies as a function of island size.

Source: C. Pryor, Phys. Rev. B 57, 7190 (1998), preprint arXiv:cond-mat/9710304. Fig. 4's
caption, verbatim:

    FIG. 4. Bound state energies as a function of island size. The dotted line indicates the
    energy for a one-monolayer biaxially strained InAs wetting layer, also computed in the
    envelope approximation. (a) Conduction band. (b) Valence band.

and the opening sentence of his Sec. V:

    "The bound state energies were computed as a function of island size using the full
    eight-band Hamiltonian (Fig. 4)."

So: the full eight-band Hamiltonian at every island size, nothing reduced. The wetting layer is
deliberately NOT modelled here -- Pryor draws it only as horizontal reference lines computed
separately as an independent quantum well in the envelope approximation, and his own quantum-dot
calculation also "performed assuming no wetting layer". The lines are omitted; everything else
in the figure is a dot calculation and is reproduced.

Energy zero is the unstrained GaAs valence-band edge, which is the zero of Pryor's Figs. 4 and 7.
On that scale the GaAs conduction edge sits at Eg(GaAs) = 1.519 eV, so panel (a) is read
downward from 1.519 and panel (b) upward from 0.

What Pryor states about this figure, and what is therefore checkable here
------------------------------------------------------------------------
Conduction band (his Sec. V):
  * "For the conduction-band there are only a few bound states in the island."
  * "The first excited state is accompanied by a nearly degenerate state. The splitting between
    these two states varies from 2 meV to 6 meV for 10 nm < b < 18 nm. The near degeneracy
    reflects the C4 symmetry of the square island, with the splitting due to the piezoelectric
    effect."  -- so the piezoelectric potential must be included, and the splitting is a
    quantitative target, not decoration.
  * "The gap between the ground state and first excited state varies from 60 meV to 95 meV over
    the range 10 nm < b < 18 nm."
Valence band:
  * "The valence-band states are more strongly confined, due to their larger effective mass.
    Only the first four states are shown in Fig. 4b ... The energy spacings vary from a few meV
    to 30 meV over the range of island sizes considered."

Kramers degeneracy
------------------
Time reversal makes every eigenvalue of this Hamiltonian exactly twofold degenerate, so the
eigenvalues come out in pairs and a "level" in Pryor's sense is one pair. That is used here as a
free correctness check rather than assumed: `kramers_levels` pairs the sorted spectrum and
reports the within-pair splitting, which must be at the residual level. The 2-6 meV splitting
Pryor discusses is a splitting BETWEEN pairs (the two C4 partner states), not within one.

Run: python pryor_fig4.py            (full sweep, several hours; writes fig4_levels.json)
     python pryor_fig4.py 10 12      (just those sizes)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import qdsolver_core as qd
import elasticity_fd as ef
import piezoelectric as pz
import kp_confined as kpc
import kp_pryor as kp
import pryor1998 as pr
import eigensolvers as eig

#: Island base widths (nm). Pryor's Fig. 4 x-axis runs 6 -> 18 nm.
SIZES = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)

#: Band edges of the unstrained GaAs barrier on this energy scale (Table I).
GAAS_CB = pr.PRYOR_TABLE_I['GaAs']['Eg']      # 1.519 eV
GAAS_VB = 0.0

DOT = pr.PRYOR_TABLE_I['InAs']
MATRIX = pr.PRYOR_TABLE_I['GaAs']
EPS_STAR = qd.eigenstrain(DOT['a0'], MATRIX['a0'])
C_DOT = (DOT['C11'], DOT['C12'], DOT['C44'])
C_MAT = (MATRIX['C11'], MATRIX['C12'], MATRIX['C44'])

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_levels.json')


# --------------------------------------------------------------------------------------
# Environment: grid, strain, piezoelectric potential, Hamiltonian
# --------------------------------------------------------------------------------------

def clean_grid(base, h, pad):
    """Mirror-symmetric grid with the island base on a grid plane, as in pryor_fig6.

    The x/y axis is centred (odd count) so `mirror_asymmetry` is exactly zero -- the {101}
    facets fall on grid points, so an axis that is not symmetric to machine precision flips
    voxels between +x and -x and manufactures a C4 violation that would be mistaken for the
    piezoelectric splitting this figure is about.
    """
    nx = int(round(2 * (base / 2 + pad) / h)) // 2 * 2 + 1
    kz0 = int(round(pad / h))
    nz = kz0 + int(round((base / 2 + pad) / h)) + 1
    cx = qd.centered_axis(nx, h)
    cz = (np.arange(nz) - kz0) * h
    X, Y, Z = np.meshgrid(cx, cx, cz, indexing='ij')
    pyr = qd.pyramid_mask(X, Y, Z, base)
    assert qd.mirror_asymmetry(pyr, 0) == 0 and qd.mirror_asymmetry(pyr, 1) == 0
    return cx, cz, pyr


def environment(base, h=1.0, pad=8.0, use_piezo=True, verbose=True):
    """Strain, piezoelectric potential and eight-band Hamiltonian for one island size."""
    cx, cz, pyr = clean_grid(base, h, pad)
    exact_vol = base ** 2 * (base / 2) / 3

    t0 = time.time()
    strain = ef.solve_strain_fd(pyr, EPS_STAR, C_DOT, C_MAT, h, tol=1e-10)
    g = strain.at(pyr)
    tr_mean = g['exx'] + g['eyy'] + g['ezz']

    phi = np.zeros(pyr.shape)
    if use_piezo:
        e14 = np.where(pyr, DOT['e14'], MATRIX['e14'])
        phi = pz.potential(strain, e14, MATRIX['eps_R'], h)

    # Electrostatic potential ENERGY of an electron is -e*phi. Both band edges shift together
    # under it, which is why this is applied to the eight-band Hamiltonian as a whole rather
    # than patched into E_c and E_v in opposite directions.
    V_e, V_h, _, _ = pr.band_edge_fields(pyr, strain.trace)
    U = -phi
    Ec, Ev = V_e + U, V_h + U

    ops = kpc.GridOperators(pyr.shape, h, periodic=False)
    fields = kp.material_fields(pyr, DOT, MATRIX, Ev, Ec, n_bands=8)
    H = kp.confined_hamiltonian(ops, fields, n_bands=8, strain=strain)

    sig_e = float(Ec[pyr].min())
    sig_h = kp.hole_sigma(Ev, strain, np.where(pyr, DOT['b'], MATRIX['b']),
                          np.where(pyr, DOT['d'], MATRIX['d']),
                          np.where(pyr, DOT['delta_so'], MATRIX['delta_so']), inside_mask=pyr)

    if verbose:
        print(f"b = {base:.0f} nm | grid {pyr.shape} = {pyr.size:,} pts, h = {h}, "
              f"box {(len(cx)*h):.0f} nm, vol err {qd.mask_volume_error(pyr, h, exact_vol):+.3f}")
        print(f"    strain: CG {strain.cg_iterations} iters, rel {strain.cg_residual:.1e}, "
              f"{time.time()-t0:.0f}s; <Tr eps> = {tr_mean:+.5f}")
        if use_piezo:
            print(f"    piezo: {phi.min()*1e3:+.1f} .. {phi.max()*1e3:+.1f} meV, "
                  f"C4 antisymmetry {pz.c4_antisymmetry_error(phi):.1e}")
        print(f"    H {H.shape[0]:,} x {H.shape[0]:,}, nnz {H.nnz:,}; "
              f"sigma_e = {sig_e:.4f} eV, sigma_h = {sig_h:.4f} eV")

    return dict(base=base, h=h, pad=pad, cx=cx, cz=cz, pyr=pyr, strain=strain, phi=phi,
                Ec=Ec, Ev=Ev, H=H, sigma_e=sig_e, sigma_h=sig_h,
                tr_mean=float(tr_mean), vol_err=float(qd.mask_volume_error(pyr, h, exact_vol)))


def ilu_folded_preconditioner(H, sigma, drop_tol=1e-3, fill_factor=8, verbose=True):
    """M ~ ((H - sigma)^2)^-1, built as two applications of an incomplete LU of (H - sigma).

    LOBPCG requires a positive-definite preconditioner, and (H - sigma)^-1 is not: its
    eigenvalues 1/(lambda - sigma) straddle zero. Applying the same approximate inverse twice
    gives ~(H - sigma)^-2, which is positive definite, and is the right operator for the folded
    problem `eigensolvers.solve_interior` actually solves. Passing the single ILU solve would be
    a preconditioner for the wrong operator with the wrong sign structure.
    """
    t0 = time.time()
    shifted = (H - sigma * sp.identity(H.shape[0], dtype=H.dtype, format='csr')).tocsc()
    ilu = spla.spilu(shifted, drop_tol=drop_tol, fill_factor=fill_factor)
    if verbose:
        nnz = ilu.L.nnz + ilu.U.nnz
        print(f"    ILU: {time.time()-t0:.0f}s, nnz(L)+nnz(U) = {nnz:,} ({nnz/H.nnz:.1f}x)")
    twice = lambda x: ilu.solve(ilu.solve(x))
    return spla.LinearOperator(H.shape, matvec=twice, matmat=twice, dtype=H.dtype)


# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------

def localization(V, pyr):
    """Fraction of sum_{i=1}^{8} |psi_i|^2 lying inside the island, per column of V.

    With no wetting layer there are states right up to the GaAs band edges (Pryor says so
    explicitly), and in a finite box those unbound states are quantized into discrete levels
    that a solver returns alongside the real ones. Energy alone cannot tell them apart near the
    edge; this can.
    """
    V = np.asarray(V)
    n = pyr.size
    out = []
    for j in range(V.shape[1]):
        rho = (np.abs(V[:, j].reshape(8, n)) ** 2).sum(axis=0)
        out.append(float(rho.reshape(pyr.shape)[pyr].sum() / rho.sum()))
    return np.array(out)


def bound_states(env, band, k=8, tol=1e-7, maxiter=8000, precondition=False, verbose=True):
    """Eigenstates of the eight-band Hamiltonian nearest the relevant band edge.

    `band` is 'cb' or 'vb'. The target sigma is the bottom of the local conduction well or the
    exact k = 0 top of the local valence band respectively -- see `kp_pryor.hole_sigma` for why
    the latter is computed rather than guessed.
    """
    H, pyr = env['H'], env['pyr']
    sigma = env['sigma_e'] if band == 'cb' else env['sigma_h']
    M = ilu_folded_preconditioner(H, sigma, verbose=verbose) if precondition else None

    t0 = time.time()
    E, V, info = eig.solve_interior(H, k=k, sigma=sigma, M=M, tol=tol, maxiter=maxiter,
                                    verbose=verbose)
    loc = localization(V, pyr)

    # Report in the direction each band is read: conduction states upward from the well bottom,
    # valence states downward from the valence edge (i.e. most-bound hole first).
    order = np.argsort(E) if band == 'cb' else np.argsort(-E)
    E, loc = E[order], loc[order]

    if verbose:
        print(f"    {band}: {time.time()-t0:.0f}s, residual {info['residuals'].max():.1e} eV")
    return dict(E=E, V=V[:, order], loc=loc, sigma=sigma, info=info, band=band)


def kramers_levels(E, loc, split_tol=1e-4):
    """Collapse the exactly-twofold spectrum into physical levels.

    Time reversal guarantees the pairing; this checks it instead of trusting it. Returns
    (level energies, mean localization per level, largest within-pair splitting in eV). A
    within-pair splitting above the residual means the solve has not resolved the degeneracy and
    the level count below it is unreliable.
    """
    E = np.asarray(E)
    n = len(E) // 2 * 2
    pairs = E[:n].reshape(-1, 2)
    lvl = pairs.mean(axis=1)
    worst = float(np.abs(np.diff(pairs, axis=1)).max()) if n else 0.0
    lloc = np.asarray(loc)[:n].reshape(-1, 2).mean(axis=1)
    return lvl, lloc, worst


# --------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------

def run_size(base, h=1.0, pad=8.0, k=8, use_piezo=True, precondition=False, verbose=True,
             **kw):
    env = environment(base, h=h, pad=pad, use_piezo=use_piezo, verbose=verbose)
    out = dict(base=base, h=h, pad=pad, k=k, piezo=use_piezo, npts=int(env['pyr'].size),
               tr_mean=env['tr_mean'], vol_err=env['vol_err'],
               sigma_e=env['sigma_e'], sigma_h=env['sigma_h'])
    for band in ('cb', 'vb'):
        st = bound_states(env, band, k=k, precondition=precondition, verbose=verbose, **kw)
        lvl, lloc, worst = kramers_levels(st['E'], st['loc'])
        out[band] = dict(E=st['E'].tolist(), loc=st['loc'].tolist(),
                         levels=lvl.tolist(), level_loc=lloc.tolist(),
                         kramers_split=worst,
                         residual=float(st['info']['residuals'].max()),
                         iterations=int(st['info']['iterations']),
                         seconds=float(st['info']['time']))
        if verbose:
            edge = GAAS_CB if band == 'cb' else GAAS_VB
            binding = (edge - lvl) if band == 'cb' else (lvl - edge)
            print(f"    {band} levels (eV): " +
                  ", ".join(f"{e:.4f}[{b*1e3:.0f}meV,{L*100:.0f}%]"
                            for e, b, L in zip(lvl, binding, lloc)))
            print(f"    Kramers splitting within pairs: {worst*1e3:.2e} meV")
    del env
    return out


def sweep(sizes=SIZES, cache=CACHE, recompute=False, **kw):
    """Run (or load) the whole size sweep. Results are appended to `cache` as they finish, so an
    interrupted run resumes rather than restarting."""
    done = {}
    if cache and os.path.exists(cache) and not recompute:
        with open(cache) as fh:
            done = {r['base']: r for r in json.load(fh)}
    for b in sizes:
        if b in done:
            print(f"b = {b:.0f} nm: loaded from {os.path.basename(cache)}")
            continue
        done[b] = run_size(b, **kw)
        if cache:
            with open(cache, 'w') as fh:
                json.dump([done[x] for x in sorted(done)], fh, indent=1)
    return [done[b] for b in sorted(done) if b in sizes]


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def _levels_array(results, band, loc_min=0.0):
    """(array of b, list of level arrays), keeping only the genuinely bound levels.

    Two conditions, and a level has to pass both. It must lie inside the barrier gap -- below
    the unstrained GaAs conduction edge for electrons, above the GaAs valence edge for holes,
    those being the asymptotic barrier edges far from the island -- and it must be localized in
    the dot. Neither test alone is enough: a finite box quantizes the continuum into discrete
    levels that can sit below the edge, and a small box can hold an above-barrier state with most
    of its amplitude still over the island.

    The result is truncated at the first failure rather than filtered, because the levels come
    in order of binding: once one is not bound, nothing above it is either. Dropping a level from
    the middle would silently renumber the ones above it, and "the n-th bound state" is exactly
    what each curve in Pryor's figure is.
    """
    bs, lv = [], []
    for r in results:
        lvl = np.array(r[band]['levels'])
        lloc = np.array(r[band]['level_loc'])
        in_gap = (lvl < GAAS_CB) if band == 'cb' else (lvl > GAAS_VB)
        bad = np.flatnonzero(~in_gap | (lloc < loc_min))
        n = int(bad[0]) if len(bad) else len(lvl)
        bs.append(r['base'])
        lv.append(lvl[:n])
    return np.array(bs), lv


def plot_conduction(results, loc_min=0.35, figsize=(6.2, 4.6), ax=None):
    """Pryor's Fig. 4(a): conduction-band bound-state energies vs island size."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    bs, lv = _levels_array(results, 'cb', loc_min)
    nmax = max(len(v) for v in lv)
    for j in range(nmax):
        x = [b for b, v in zip(bs, lv) if len(v) > j]
        y = [v[j] for v in lv if len(v) > j]
        ax.plot(x, y, 'o-', ms=3.5, lw=1.0, color='k', mfc='w' if j else 'k')
    ax.axhline(GAAS_CB, color='0.5', lw=0.9, ls='--')
    ax.text(bs[0], GAAS_CB, 'GaAs $E_c$ ', va='bottom', ha='left', fontsize=8, color='0.4')
    ax.set(xlabel='$b$  (nm)', ylabel='$E$  (eV)')
    ax.margins(x=0.06, y=0.08)
    ax.set_title('(a) Conduction band', fontsize=10)
    return ax


def plot_valence(results, loc_min=0.35, figsize=(6.2, 4.6), ax=None):
    """Pryor's Fig. 4(b): valence-band bound-state energies vs island size.

    Plotted as E_v measured upward from the unstrained GaAs valence edge, as Pryor does, so a
    HIGHER point is a MORE bound hole.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    bs, lv = _levels_array(results, 'vb', loc_min)
    nmax = max(len(v) for v in lv)
    for j in range(nmax):
        x = [b for b, v in zip(bs, lv) if len(v) > j]
        y = [v[j] for v in lv if len(v) > j]
        ax.plot(x, y, 'o-', ms=3.5, lw=1.0, color='k', mfc='w' if j else 'k')
    ax.axhline(GAAS_VB, color='0.5', lw=0.9, ls='--')
    ax.text(bs[0], GAAS_VB, 'GaAs $E_v$ ', va='bottom', ha='left', fontsize=8, color='0.4')
    ax.set(xlabel='$b$  (nm)', ylabel='$E_v$  (eV)')
    ax.margins(x=0.06, y=0.08)
    ax.set_title('(b) Valence band', fontsize=10)
    return ax


# --------------------------------------------------------------------------------------
# Pryor's quantitative claims about Fig. 4
# --------------------------------------------------------------------------------------

def check_claims(results, loc_min=0.35):
    """Print Pryor's Sec. V statements about Fig. 4 beside what this calculation gives."""
    bs, cb = _levels_array(results, 'cb', loc_min)
    _, vb = _levels_array(results, 'vb', loc_min)

    print(f"{'b (nm)':>7} {'n_cb':>5} {'E0':>8} {'E1-E0':>8} {'E2-E1':>8} "
          f"{'n_vb':>5} {'Ev0':>8} {'spacings (meV)':>26}")
    for b, c, v in zip(bs, cb, vb):
        g10 = (c[1] - c[0]) * 1e3 if len(c) > 1 else np.nan
        g21 = (c[2] - c[1]) * 1e3 if len(c) > 2 else np.nan
        sp_v = ' '.join(f"{d*1e3:.1f}" for d in -np.diff(v)) if len(v) > 1 else '-'
        e0 = f"{c[0]:.4f}" if len(c) else 'unbound'
        v0 = f"{v[0]:.4f}" if len(v) else 'unbound'
        print(f"{b:>7.0f} {len(c):>5} {e0:>8} {g10:>8.1f} {g21:>8.1f} "
              f"{len(v):>5} {v0:>8} {sp_v:>26}")

    print("\nPryor, Sec. V, on the conduction band:")
    rng = [(b, c) for b, c in zip(bs, cb) if 10 <= b <= 18 and len(c)]
    gaps = [(b, (c[1] - c[0]) * 1e3) for b, c in rng if len(c) > 1]
    if gaps:
        print(f'  "The gap between the ground state and first excited state varies from 60 meV '
              f'to 95 meV over the range 10 nm < b < 18 nm."')
        print(f"    here: {min(g for _, g in gaps):.0f} - {max(g for _, g in gaps):.0f} meV "
              f"({', '.join(f'b={b:.0f}: {g:.0f}' for b, g in gaps)})")
    splits = [(b, (c[2] - c[1]) * 1e3) for b, c in rng if len(c) > 2]
    if splits:
        print(f'  "The first excited state is accompanied by a nearly degenerate state. The '
              f'splitting between\n   these two states varies from 2 meV to 6 meV for '
              f'10 nm < b < 18 nm ... due to the piezoelectric effect."')
        print(f"    here: {min(s for _, s in splits):.1f} - {max(s for _, s in splits):.1f} meV "
              f"({', '.join(f'b={b:.0f}: {s:.1f}' for b, s in splits)})")

    print("\nPryor, Sec. V, on the valence band:")
    all_sp = [d * 1e3 for v in vb if len(v) > 1 for d in -np.diff(v)]
    if all_sp:
        print(f'  "The energy spacings vary from a few meV to 30 meV over the range of island '
              f'sizes considered."')
        print(f"    here: {min(all_sp):.1f} - {max(all_sp):.1f} meV")
    print(f'  "The valence-band states are more strongly confined, due to their larger '
          f'effective mass."')
    for b, c, v in zip(bs, cb, vb):
        print(f"    b = {b:>4.0f} nm: {len(c)} conduction level(s), {len(v)} valence level(s) "
              f"localized in the island")


if __name__ == '__main__':
    args = [float(a) for a in sys.argv[1:]]
    sizes = tuple(args) if args else SIZES
    t0 = time.time()
    res = sweep(sizes)
    print(f"\ntotal {time.time()-t0:.0f}s")
    check_claims(res)
