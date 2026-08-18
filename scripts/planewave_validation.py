"""Validate the plane-wave six-band solver, and use it to settle what the interface modes are.

Run from the repository root:

    PYTHONIOENCODING=utf-8 python -u scripts/planewave_validation.py            # sections 1-2
    PYTHONIOENCODING=utf-8 python -u scripts/planewave_validation.py --all      # everything
    PYTHONIOENCODING=utf-8 python -u scripts/planewave_validation.py --only 3

Sections
--------
1. **Assembly.** Exact checks with no physics in them: the channel decomposition against
   `kp_pryor.bulk_hamiltonian`, the derived DKK -> Pryor basis change, both orderings reducing to
   the same bulk matrix, the uniform-material limit block by block, Hermiticity, and the exact
   diagonal. All at machine precision. Seconds.

2. **Ellipticity.** The 18x18 kinetic tensor for each material: Legendre-Hadamard (passes
   everywhere), strong ellipticity under symmetrized ordering (fails everywhere) and under
   Burt-Foreman (passes everywhere). Seconds, no solve.

3. **Boundedness.** The measurement that settled what the interface modes ARE. Two ladders on
   Yeap's 2.5 nm island -- growing the basis at fixed grid, and refining the grid at fixed basis
   fraction -- each run twice, once with the real parameters and once with gamma2, gamma3 scaled
   until strong ellipticity is restored. Minutes.

4. **Cross-validation against the finite-difference solver.** Only meaningful where the problem is
   well posed, so it runs in the strongly elliptic regime. Tens of minutes.

5. **Burt-Foreman ordering: the acceptance test.** Section 3(a) again with the ordering as the
   variable instead of an artificial gamma-scaling, plus h-convergence and box convergence.
   Tens of minutes.

6. **Regression on InAs/GaAs**, where the symmetrized solver was trusted. Tens of minutes.

Why sections 3 and 5 are the point
----------------------------------
Every attempt to remove the interface modes assumed they were a discretization defect: the
compact-vs-central sampling mismatch, then a staggered cross term, then a Q1 finite-element cross
term. All three changed nothing. Section 3 tests the alternative -- that the SYMMETRIZED six-band
operator with an abrupt interface simply has no maximum, and every correct discretization must
reproduce that. A plane-wave basis has no stencil at all, so if the modes appear here too, no
choice of grid can be the cause. They do.

Section 5 is the consequence: change the ORDERING and the same ladder converges. The two sections
share a design. Each holds everything fixed except one thing that moves lambda_max of the kinetic
tensor through zero -- an artificial parameter scaling in section 3, the physical Burt-Foreman
reordering in section 5 -- so the effect cannot be attributed to anything else.
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse.linalg as spla

import materials as mt
import heterostructure as hs
import kp_pryor as kp
import kp_planewave as pw

warnings.filterwarnings('ignore', category=UserWarning)

BASE, AR, PAD, CROP = 2.5, 1.0, 6.0, 2.0          # Yeap et al. Fig. 6, AR = 1
T_K = 80.0

_fails = []


def check(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ''), flush=True)
    if not ok:
        _fails.append(name)


# ---------------------------------------------------------------------------------------
def section1():
    print("\n1. ASSEMBLY -- exact checks, no physics", flush=True)
    err = pw.validate_channels(mt.kp_params(mt.material('InSb')))
    check("channel decomposition == kp_pryor.bulk_hamiltonian", err < 1e-13, f"{err:.2e} eV")

    _, u = pw.dkk_to_pryor_unitary()
    check("DKK -> Pryor basis change is derived, unitary and unique",
          u['n_null'] == 1 and u['unitarity'] < 1e-12 and u['residual'] < 1e-11,
          f"null dim {u['n_null']}, unitarity {u['unitarity']:.2e}, "
          f"worst intertwining residual at fresh k {u['residual']:.2e} eV")
    for o in pw.ORDERINGS:
        v = pw.validate_ordering(o)
        check(f"ordering {o!r}: bulk Hamiltonian unchanged, blocks pair up Hermitian",
              v['bulk_error'] < 1e-13 and v['hermiticity'] < 1e-13,
              f"bulk {v['bulk_error']:.2e} eV, A_ji vs A_ij^dagger {v['hermiticity']:.2e}")

    shape, h = (8, 8, 8), 0.6
    p = mt.kp_params(mt.material('InAs'))
    alphas = {f'gamma{i}': np.full(shape, p[f'gamma{i}L']) for i in (1, 2, 3)}
    local = [('P', np.zeros(shape)), ('delta', np.full(shape, p['delta_so']))]
    local += [(c, np.zeros(shape)) for c in ('Q', 'R_re', 'R_im', 'S_re', 'S_im')]
    # Both orderings, because both must reduce to the SAME bulk matrix -- that is what makes the
    # reordering an ordering rather than a change of physics.
    Gx, Gy, Gz = (g.ravel() for g in pw.reciprocal_grid(shape, h))
    for o in pw.ORDERINGS:
        H = pw.SixBandPlaneWave(alphas, local, shape, h, gmax=0.5 * pw.nyquist(h), ordering=o)
        Hd = H.dense()
        worst = 0.0
        for a, g in enumerate(H.kept):
            ref = kp.bulk_hamiltonian(Gx[g], Gy[g], Gz[g], p, n_bands=6, Ev=0.0)
            worst = max(worst, float(np.abs(Hd[a::H.n_pw, a::H.n_pw] - ref).max()))
        check(f"uniform material, {o}: all {H.n_pw} G blocks == the bulk matrix",
              worst < 1e-13, f"{worst:.2e} eV")
        off = Hd.copy()
        for a in range(H.n_pw):
            off[a::H.n_pw, a::H.n_pw] = 0.0
        check(f"uniform material, {o}: G != G' blocks vanish",
              float(np.abs(off).max()) == 0.0)
        check(f"Hermiticity from random probes, {o}", H.hermiticity() < 1e-13,
              f"{H.hermiticity():.2e}")

    # Heterogeneous, so the convolution, the local terms and the diagonal are all exercised.
    q = mt.kp_params(mt.material('InSb'))
    c = (np.arange(shape[0]) - (shape[0] - 1) / 2) * h
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    isl = (X ** 2 + Y ** 2 + Z ** 2) < 1.4 ** 2
    ah = {f'gamma{i}': np.where(isl, q[f'gamma{i}L'], p[f'gamma{i}L']) for i in (1, 2, 3)}
    Ev = np.where(isl, 0.59, 0.0)
    lh = [('P', -Ev), ('delta', np.where(isl, q['delta_so'], p['delta_so']))]
    lh += [(ch, np.zeros(shape)) for ch in ('Q', 'R_re', 'R_im', 'S_re', 'S_im')]
    for o in pw.ORDERINGS:
        Hh = pw.SixBandPlaneWave(ah, lh, shape, h, gmax=0.5 * pw.nyquist(h), ordering=o)
        Hhd = Hh.dense()
        d = float(np.abs(np.diag(Hhd).real - Hh.diagonal()).max())
        check(f"heterogeneous, {o}: exact diagonal == built matrix", d < 1e-13, f"{d:.2e} eV")
        check(f"heterogeneous, {o}: built matrix is Hermitian",
              float(np.abs(Hhd - Hhd.conj().T).max()) < 1e-13)
        if o == 'burt-foreman':
            # The bound is real now, so it is a hard check on a genuine heterostructure.
            top = float(np.linalg.eigvalsh(Hhd).max())
            check("burt-foreman obeys the variational bound on a stepped interface",
                  top <= pw.spectrum_bound(Hh)['bound'] + 1e-9,
                  f"top {top:+.4f} eV vs bound {pw.spectrum_bound(Hh)['bound']:.4f} eV")

    # A control that is strongly elliptic for a different reason: with gamma2 = gamma3 = 0 the
    # SYMMETRIZED tensor is -gamma1 times the identity, so its bound holds too. Independent of the
    # reordering, and it was the only such control before Burt-Foreman existed here.
    a0 = dict(ah)
    a0['gamma2'] = np.zeros(shape)
    a0['gamma3'] = np.zeros(shape)
    se = pw.strong_ellipticity(30.0, 0.0, 0.0, 'symmetrized')
    H0 = pw.SixBandPlaneWave(a0, lh, shape, h, gmax=0.5 * pw.nyquist(h), ordering='symmetrized')
    top = float(np.linalg.eigvalsh(H0.dense()).max())
    check("gamma2 = gamma3 = 0 control obeys the variational bound",
          se['strongly_elliptic'] and top <= Ev.max() + 1e-9,
          f"lambda_max(A) = {se['lambda_max']:+.3f}, top eigenvalue {top:+.4f} eV "
          f"vs local edge {Ev.max():.4f} eV")


# ---------------------------------------------------------------------------------------
def section2():
    print("\n2. ELLIPTICITY -- the condition that holds, the one that does not, and the fix",
          flush=True)
    print(f"    {'material':<8} {'g1-2g2':>8} {'g1-2g3':>8} {'LH':>4} | "
          f"{'lmax sym':>10} {'strong?':>8} | {'lmax B-F':>11} {'strong?':>8}", flush=True)
    ok_bf = True
    for name in ('GaAs', 'GaSb', 'InAs', 'InSb'):
        p = mt.kp_params(mt.material(name))
        g1, g2, g3 = p['gamma1L'], p['gamma2L'], p['gamma3L']
        a = pw.strong_ellipticity(g1, g2, g3, 'symmetrized')
        b = pw.strong_ellipticity(g1, g2, g3, 'burt-foreman')
        ok_bf &= b['strongly_elliptic']
        print(f"    {name:<8} {g1-2*g2:+8.2f} {g1-2*g3:+8.2f} "
              f"{'ok' if min(g1-2*g2, g1-2*g3) > 0 else 'FAIL':>4} | "
              f"{a['lambda_max']:+10.4f} {'ok' if a['strongly_elliptic'] else 'FAIL':>8} | "
              f"{b['lambda_max']:+11.3e} {'ok' if b['strongly_elliptic'] else 'FAIL':>8}",
              flush=True)
    print("    Legendre-Hadamard passes everywhere -- every bulk band curves downward.", flush=True)
    print("    Symmetrized ordering fails strong ellipticity everywhere: an abrupt interface then",
          flush=True)
    print("    unbinds the spectrum. Burt-Foreman lands lambda_max on zero, so the operator is",
          flush=True)
    print("    bounded above by the local valence edge.", flush=True)
    check("Burt-Foreman ordering is strongly elliptic for every material here", ok_bf)

    # Which coefficient carries it? gamma3 alone -- and gamma3 is also the only coefficient the
    # finite-difference cross-term operator is ever called with, which is why
    # scripts/gamma3_diagnostic.py isolated the right parameter from the wrong hypothesis.
    print(f"\n    lambda_max(A) dropping one coefficient at a time:", flush=True)
    print(f"    {'material':<8} {'full':>10} {'gamma3=0':>10} {'gamma2=0':>10} {'both=0':>10}",
          flush=True)
    for name in ('InSb', 'InAs'):
        p = mt.kp_params(mt.material(name))
        g1, g2, g3 = p['gamma1L'], p['gamma2L'], p['gamma3L']
        v = [pw.strong_ellipticity(g1, a, b)['lambda_max']
             for a, b in ((g2, g3), (g2, 0.0), (0.0, g3), (0.0, 0.0))]
        print(f"    {name:<8} {v[0]:+10.4f} {v[1]:+10.4f} {v[2]:+10.4f} {v[3]:+10.4f}", flush=True)
    print("    gamma3 = 0 restores strong ellipticity; gamma2 = 0 does not. The R and S blocks "
          "carry it all.", flush=True)

    # Where does s put lambda_max through zero? That crossing is what section 3 brackets.
    print(f"\n    scaling gamma2, gamma3 by s:", flush=True)
    print(f"    {'s':>5} {'lmax(A) InSb':>13} {'lmax(A) InAs':>13}", flush=True)
    for s in (0.0, 0.25, 0.4, 0.5, 0.75, 1.0):
        vals = []
        for name in ('InSb', 'InAs'):
            p = mt.kp_params(mt.material(name))
            vals.append(pw.strong_ellipticity(p['gamma1L'], s * p['gamma2L'],
                                              s * p['gamma3L'])['lambda_max'])
        print(f"    {s:5.2f} {vals[0]:+13.3f} {vals[1]:+13.3f}", flush=True)


# ---------------------------------------------------------------------------------------
_ENVS = {}


def _env(h):
    if h not in _ENVS:
        mt.set_temperature(T_K)
        e = hs.build(hs.ellipsoid(BASE, BASE / AR), 'InSb', matrix='InAs', h=h, pad=PAD,
                     vol_tol=0.30, verbose=False)
        _ENVS[h] = pw.crop_env(e, CROP)
    return _ENVS[h]


def _ham(sub, h, gmax, s):
    a, loc, _ = pw.fields_from_env(sub)
    a = dict(a)
    a['gamma2'] = s * a['gamma2']
    a['gamma3'] = s * a['gamma3']
    # Explicitly symmetrized: sections 3 and 4 measure that operator, so they must not
    # follow the module default, which is now burt-foreman.
    return pw.SixBandPlaneWave(a, loc, sub['mask'].shape, h, gmax=gmax,
                               ordering='symmetrized')


def _top(H, v0=None):
    """Largest eigenvalue, by Lanczos.

    NOT LOBPCG. LOBPCG minimises a Rayleigh quotient, and when the operator has no maximum there
    is nothing to minimise -- it runs to maxiter and returns whatever it reached, which is
    informative but is not a measurement. Lanczos converges to the extremal eigenvalue of the
    finite matrix, which always exists and is exactly the quantity in question.
    """
    t0 = time.time()
    E, V = spla.eigsh(H.as_linear_operator(), k=3, which='LA', tol=1e-7, v0=v0,
                      ncv=min(H.dim - 1, 40))
    j = int(np.argmax(E))
    return float(E[j]), V[:, j], time.time() - t0


def section3():
    print("\n3. BOUNDEDNESS -- what the interface modes actually are", flush=True)
    for s in (1.0, 0.25):
        p = mt.kp_params(mt.material('InSb'))
        lm = pw.strong_ellipticity(p['gamma1L'], s * p['gamma2L'],
                                   s * p['gamma3L'])['lambda_max']
        print(f"\n  s = {s}: lambda_max(A, InSb) = {lm:+.3f}  "
              f"({'STRONGLY ELLIPTIC' if lm <= 0 else 'not strongly elliptic'})", flush=True)

        sub = _env(0.20)
        edge = hs.valence_edge_top(sub)
        print(f"  (a) fixed grid h = 0.20 nm {sub['mask'].shape}, growing basis. The coefficient "
              f"field is\n      fixed, so only the basis changes and a Galerkin maximum can only "
              f"rise. Does it stop?\n      sampled island valence edge = {edge:.4f} eV", flush=True)
        Hp, vp, prev = None, None, None
        for gm in (3.0, 4.0, 5.0, 6.0, 7.0, 7.85):
            H = _ham(sub, 0.20, gm, s)
            v0 = None if Hp is None else pw.prolong(Hp, vp, H)[:, 0]
            t, vp, dt = _top(H, v0)
            d = '' if prev is None else f"{t - prev:+8.4f}"
            print(f"      gmax {gm:5.2f}  n_pw {H.n_pw:6,}  top E {t:+9.4f}  step {d:>9}  "
                  f"{dt:5.0f}s", flush=True)
            Hp, prev = H, t

        print("  (b) fixed basis fraction gmax = nyquist/2, refining the grid. Now the "
              f"coefficient\n      sharpens too, so this asks how the ceiling itself moves.",
              flush=True)
        for h in (0.40, 0.30, 0.25, 0.20):
            sub = _env(h)
            gm = 0.5 * pw.nyquist(h)
            H = _ham(sub, h, gm, s)
            t, _, dt = _top(H)
            print(f"      h {h:4.2f}  gmax {gm:5.2f}  n_pw {H.n_pw:6,}  top E {t:+9.4f}  "
                  f"E*h^2 = {t*h*h:7.4f} eV nm^2  {dt:5.0f}s", flush=True)


# ---------------------------------------------------------------------------------------
def section4():
    """Do the two discretizations agree? Only askable where the problem is well posed.

    Comparing them at s = 1 would be meaningless: the operator has no maximum, so neither method
    converges and neither number is the answer to anything. Agreement or disagreement would say
    nothing about the discretizations.

    So the comparison is run in the strongly elliptic regime (gamma2, gamma3 scaled by 0.25),
    where a right answer exists and both methods must find it. That validates the plane-wave
    assembly against the finite-difference one on a genuine 3D heterostructure -- two completely
    different discretizations of the same operator -- which is the check that makes either of them
    a referee for the other once the ordering is fixed.

    Both are run PERIODIC so they solve literally the same boundary-value problem; the finite
    difference solver's usual Dirichlet box would otherwise contribute a difference that has
    nothing to do with the discretization. The island is large (12 nm base) because at s = 0.25
    the carrier is light and a 2.5 nm well would not bind it, leaving a box state whose energy is
    all boundary condition.
    """
    print("\n4. CROSS-VALIDATION against the finite-difference solver", flush=True)
    print("  In the STRONGLY ELLIPTIC regime (gamma2, gamma3 x 0.25), where a right answer "
          "exists.\n  Periodic on both sides, so this is discretization against discretization "
          "and nothing else.", flush=True)
    mt.set_temperature(T_K)
    s, base, ar = 0.25, 12.0, 2.0

    for h in (0.5, 0.4):
        env = hs.build(hs.ellipsoid(base, base / ar), 'InSb', matrix='InAs', h=h, pad=PAD,
                       vol_tol=0.30, verbose=False)
        sub = pw.crop_env(env, 3.0)
        fields = kp.material_fields(sub['mask'], mt.kp_params(sub['dot']),
                                    mt.kp_params(sub['matrix']), sub['Ev'], sub['Ec'], n_bands=6)
        fields['gamma2'] = s * fields['gamma2']
        fields['gamma3'] = s * fields['gamma3']

        import kp_confined as kpc
        ops = kpc.GridOperators(sub['mask'].shape, h, periodic=True)
        Hfd = kp.confined_hamiltonian(ops, fields, n_bands=6, strain=sub['strain'])
        t0 = time.time()
        Efd, _ = spla.eigsh(Hfd, k=3, which='LA', tol=1e-9)
        tfd = time.time() - t0
        print(f"    h {h:4.2f}  grid {sub['mask'].shape}  FD {Hfd.shape[0]:,} unknowns  "
              f"top E {Efd.max():+.5f} eV  {tfd:5.0f}s", flush=True)

        Hp, vp = None, None
        for frac in (0.6, 0.8, 1.0):
            gm = frac * 0.5 * pw.nyquist(h)
            H = _ham(sub, h, gm, s)
            v0 = None if Hp is None else pw.prolong(Hp, vp, H)[:, 0]
            t, vp, dt = _top(H, v0)
            print(f"    h {h:4.2f}  gmax {gm:5.2f}  PW {H.dim:,} unknowns "
                  f"({H.n_pw:,} waves)  top E {t:+.5f} eV  "
                  f"diff from FD {(t - Efd.max())*1e3:+7.2f} meV  {dt:5.0f}s", flush=True)
            Hp = H
        check(f"FD and plane-wave agree at h = {h} (strongly elliptic)",
              abs(t - Efd.max()) < 5e-3, f"{(t - Efd.max())*1e3:+.2f} meV apart")


def section5():
    """Burt-Foreman ordering: does it flatten the ladder the symmetrized one runs away on?

    This is the acceptance test for the reordering, and it is the same experiment as section 3(a)
    with `ordering` in place of the artificial gamma-scaling. Same geometry, same grid, same
    strain, same band edges, same solver, real parameters throughout -- only the operator ordering
    differs. If the symmetrized column climbs and the Burt-Foreman column flattens, the fix works
    and small dots are reachable.

    Confinement is quoted from the MEAN island valence edge, not the max: for an ellipsoid the
    strain is homogeneous by Eshelby's theorem, so the mean is the estimator of a constant while
    the max tracks the upper tail of the staircase noise and drifts with h. See
    `heterostructure.valence_edge_top`.
    """
    print("\n5. BURT-FOREMAN ORDERING -- the acceptance test", flush=True)
    sub = _env(0.20)
    tr = hs.valence_edge_top(sub, report=True)
    print(f"  Yeap's 2.5 nm AR 1 island, grid {sub['mask'].shape} at h = 0.20 nm. Island valence "
          f"edge:\n  max {tr['top_max']:.4f}, mean {tr['top_mean']:.4f} eV "
          f"(interior std {tr['interior_std']*1e3:.1f} meV). Confinement quoted from the mean.",
          flush=True)

    finals = {}
    for ordering in ('symmetrized', 'burt-foreman'):
        H0 = pw.hamiltonian_from_env(sub, gmax=3.0, ordering=ordering)
        bound = pw.spectrum_bound(H0)['bound']
        print(f"\n  {ordering}: spectrum bound (band-limited local valence edge) "
              f"{bound:.4f} eV", flush=True)
        Hp, vp, prev = None, None, None
        for gm in (3.0, 4.0, 5.0, 6.0, 7.0, 7.85):
            H = pw.hamiltonian_from_env(sub, gmax=gm, ordering=ordering)
            v0 = None if Hp is None else pw.prolong(Hp, vp, H)[:, 0]
            E, vp, dt = _top(H, v0)
            Hp = H
            loc = float(H.density(vp)[sub['mask']].sum())
            d = '' if prev is None else f"{E - prev:+8.4f}"
            print(f"    gmax {gm:5.2f}  n_pw {H.n_pw:6,}  top E {E:+9.4f}  step {d:>9}  "
                  f"loc {loc*100:5.1f}%  conf {(tr['top_mean']-E)*1e3:7.1f} meV  {dt:5.0f}s",
                  flush=True)
            prev = E
        finals[ordering] = (prev, bound)

    e_bf, bound_bf = finals['burt-foreman']
    check("Burt-Foreman respects the variational bound", e_bf <= bound_bf + 1e-9,
          f"top {e_bf:+.4f} eV vs bound {bound_bf:.4f} eV")
    check("symmetrized does NOT (this is the defect, not a failure of the test)",
          finals['symmetrized'][0] > finals['symmetrized'][1],
          f"top {finals['symmetrized'][0]:+.4f} eV vs bound {finals['symmetrized'][1]:.4f} eV")

    print("\n  h-convergence under Burt-Foreman, each grid at the largest cutoff it supports.",
          flush=True)
    print("  The symmetrized solver could not be converged here at ANY spacing: 430 -> 174 meV "
          "over\n  h = 0.50 -> 0.25 with no limit.", flush=True)
    print("  E_trans = Ec(matrix, far) - E_hole is the quantity to read. This is a broken-gap "
          "type-II\n  dot, so that IS the transition energy -- and unlike a confinement energy it "
          "does not\n  involve v_top, whose mean over a 2.5 nm mask is mostly boundary cells and "
          "drifts with h.\n  Yeap et al. Fig. 6(b) at AR 1 reports 0.16 eV.", flush=True)
    prev = None
    for h in (0.40, 0.30, 0.25, 0.20):
        s = _env(h)
        t = hs.valence_edge_top(s, reduce='mean')
        H = pw.hamiltonian_from_env(s, gmax=0.5 * pw.nyquist(h), ordering='burt-foreman')
        E, V, dt = _top(H)
        loc = float(H.density(V)[s['mask']].sum())
        etr = s['Ec_far'] - E
        d = '' if prev is None else f"{(etr - prev)*1e3:+8.1f}"
        print(f"    h {h:4.2f}  gmax {0.5*pw.nyquist(h):5.2f}  n_pw {H.n_pw:6,}  E {E:+.4f}  "
              f"E_trans {etr:.4f} eV  step {d:>9} meV  |  v_top(mean) {t:.4f} conf "
              f"{(t-E)*1e3:6.1f}  loc {loc*100:5.1f}%  {dt:4.0f}s", flush=True)
        prev = etr

    print("\n  Box convergence: the k.p box is cropped out of the strain box, so its padding is a"
          "\n  free parameter and a tight one squeezes the state upward. The hole is only ~54% "
          "inside\n  a 2.5 nm island, so this is not a formality.", flush=True)
    for cp in (1.5, 2.0, 3.0, 4.0):
        mt.set_temperature(T_K)
        big = hs.build(hs.ellipsoid(BASE, BASE / AR), 'InSb', matrix='InAs', h=0.25, pad=PAD,
                       vol_tol=0.30, verbose=False)
        s = pw.crop_env(big, cp)
        t = hs.valence_edge_top(s, reduce='mean')
        H = pw.hamiltonian_from_env(s, gmax=0.5 * pw.nyquist(0.25), ordering='burt-foreman')
        E, V, dt = _top(H)
        print(f"    crop pad {cp:4.1f} nm  grid {str(s['mask'].shape):>14}  n_pw {H.n_pw:6,}  "
              f"conf {(t-E)*1e3:7.1f} meV  loc {float(H.density(V)[s['mask']].sum())*100:5.1f}%  "
              f"{dt:5.0f}s", flush=True)


def section6():
    """Does the reordering hold the case the symmetrized solver was trusted on?

    InAs/GaAs is where this package's k.p is validated against Pryor, and where lambda_max is
    eightfold smaller than InSb's (+0.226 against +1.741). The question is whether Burt-Foreman
    holds that case.

    Note what this section CANNOT do. The obvious comparison -- Burt-Foreman against symmetrized in
    the same plane-wave basis -- is not available, because symmetrized does not converge here
    either: at h = 0.40 it runs away just as it does in the antimonides, only later. So the
    reference has to be the FINITE-DIFFERENCE solver at a coarse spacing, which is the regime the
    Pryor benchmark was actually run in and where the runaway ceiling (~0.046/h^2) is small against
    the well. The two are then different operators solved by different methods, so they are not
    required to agree exactly -- they differ by an interface term, which should matter less as the
    island grows. A Burt-Foreman ladder that failed to converge, or a difference that grew with
    island size, would be damning.
    """
    print("\n6. REGRESSION -- InAs/GaAs, where the symmetrized solver is trusted", flush=True)
    mt.set_temperature(T_K)
    for base, ar in ((12.0, 2.0), (6.0, 2.0)):
        env = hs.build(hs.ellipsoid(base, base / ar), 'InAs', matrix='GaAs', h=0.4, pad=PAD,
                       vol_tol=0.30, verbose=False)
        sub = pw.crop_env(env, 3.0)
        t = hs.valence_edge_top(sub, reduce='mean')
        out = {}
        for ordering in ('symmetrized', 'burt-foreman'):
            Hp, vp, seq = None, None, []
            for frac in (0.6, 0.8, 1.0):
                H = pw.hamiltonian_from_env(sub, gmax=frac * 0.5 * pw.nyquist(0.4),
                                            ordering=ordering)
                v0 = None if Hp is None else pw.prolong(Hp, vp, H)[:, 0]
                E, vp, _ = _top(H, v0)
                Hp = H
                seq.append((t - E) * 1e3)
            out[ordering] = seq
            conv = abs(seq[-1] - seq[-2]) < 5.0
            print(f"    base {base:4.1f} nm  {ordering:13s} conf "
                  + ' -> '.join(f"{c:7.1f}" for c in seq) + " meV  "
                  f"(last step {seq[-1]-seq[-2]:+8.1f}, "
                  f"{'converged' if conv else 'NOT CONVERGED'})", flush=True)
        check(f"base {base:g} nm: Burt-Foreman converges on InAs/GaAs",
              abs(out['burt-foreman'][-1] - out['burt-foreman'][-2]) < 5.0,
              f"last step {out['burt-foreman'][-1]-out['burt-foreman'][-2]:+.1f} meV")

        # The reference: the finite-difference solver at a COARSE spacing, its trusted regime.
        env_c = hs.build(hs.ellipsoid(base, base / ar), 'InAs', matrix='GaAs', h=1.0, pad=PAD,
                         vol_tol=0.30, verbose=False)
        t_c = hs.valence_edge_top(env_c, reduce='mean')
        r = hs.six_band_holes(env_c, k=4, verbose=False)
        fd = (t_c - r['E'][0]) * 1e3 if len(r['E']) else float('nan')
        print(f"    base {base:4.1f} nm  FD h = 1.0 nm  conf {fd:7.1f} meV  "
              f"loc {r['inside'][0]*100:5.1f}%  spurious {r['n_spurious']}  |  "
              f"plane-wave Burt-Foreman {out['burt-foreman'][-1]:7.1f} meV  "
              f"-> {out['burt-foreman'][-1]-fd:+.1f} meV apart", flush=True)


def section7():
    """`refine_bandlimited` on ODD-length axes -- a regression test for a real, shipped bug.

    The function evaluates the trigonometric interpolant on a finer grid, and `check_ellipticity`
    is built on it: it is how the guard sees the field the solver actually integrates against
    rather than the samples, where the inequalities hold trivially. It padded the shifted spectrum
    with `(S - s)//2`, which puts DC one bin away from where `ifftshift` reads it whenever s is
    odd -- a linear phase ramp across the spectrum, so the interpolant is scrambled, not shifted.

    It was invisible on most grids because most boxes came out even, and catastrophic on the rest.
    In the 77 K production sweep it failed exactly the six points whose cropped box had an odd axis
    ([18,18,17] at base 6, [12,12,9] at 15, [10,10,7] at 20), reporting gamma1 - 2 gamma3 = -1.87
    against a true +1.52, i.e. a spurious non-elliptic verdict that aborted each solve. The lesson
    worth keeping is that the guard was the thing that broke, and it broke LOUDLY in a direction
    that looked like a physics failure -- read as "these materials are marginal" (which they are,
    +1.6 on gamma1 = 20) it is entirely plausible.

    Test: a field built from modes 1-3 is exactly representable, so its interpolant is itself and
    the refinement error must be zero at every length and every factor.
    """
    print("\n=== 7. refine_bandlimited: odd-length axes (regression) ===", flush=True)

    def f(t):
        return 1.0 + 0.7 * np.cos(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t) \
            - 0.4 * np.cos(6 * np.pi * t)

    worst = 0.0
    for n in (7, 10, 11, 16, 17, 18, 19, 32, 33):
        for factor in (2, 3):
            coarse = np.repeat(np.repeat(f(np.arange(n) / n)[:, None, None], 4, 1), 4, 2)
            got = pw.refine_bandlimited(coarse, factor)[:, 0, 0]
            e = float(np.abs(got - f(np.arange(n * factor) / (n * factor))).max())
            worst = max(worst, e)
            if n in (17, 18):
                print(f"    n = {n:3d} ({'odd' if n % 2 else 'even'}), factor {factor}: "
                      f"max error {e:.2e}", flush=True)
    check("refine_bandlimited is exact for band-limited input at any length/factor",
          worst < 1e-12, f"worst error {worst:.2e} (was 1.5 for odd n before the fix)")


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4, 5: section5, 6: section6,
            7: section7}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, nargs='*', help='section numbers to run')
    ap.add_argument('--all', action='store_true', help='include the slow sections 3 and 4')
    args = ap.parse_args()
    want = args.only if args.only else (sorted(SECTIONS) if args.all else [1, 2, 7])
    for n in want:
        SECTIONS[n]()
    print(f"\n{'ALL CHECKS PASSED' if not _fails else 'FAILURES: ' + ', '.join(_fails)}",
          flush=True)
