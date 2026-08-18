"""Transition energies for InSb and In(x)Ga(1-x)Sb islands, vs size, aspect ratio and temperature.

The production sweep. Resumable: one JSON record per point, written as each completes, so an
interrupted run resumes instead of restarting.

    E_transition = E_c(matrix, far) - E_hole

E_c(far) is the electron reference because **there is no bound electron state to use instead** --
not as an approximation, but as a measured result. The island expels the electron (its conduction
edge sits ~810 meV above the matrix) and the only candidate well is the thin tensile shell it digs
around itself. `scripts/electron_binding.py` runs that to ground: box series out to L = 120 nm on
2.5, 10 and 20 nm islands, and at every size the level decays to E_c(far) as 1/L^2 FROM ABOVE,
tracking the empty-box quantum, never settling exponentially and never crossing. The 3D shallow-
well threshold says why -- 150 meV deep but ~1 nm thick is V0*R^2 ~ 0.15 against the ~3.6 eV.nm^2
needed at m = 0.026 -- and the larger islands are MORE repulsive, not less, so nothing in this
sweep's size range binds one. Yeap et al.'s E1, ~17 meV below E_c(far), is on this reading the
lowest state of their finite FEM domain.

So E_trans as computed here has no electron binding subtracted, and that is the physical answer
rather than an upper bound. `--electron` remains for diagnosis only and is NOT the basis of any
number quoted: it solves on the strain box, which is far too small, and will report box
quantisation. Use `scripts/electron_binding.py` for the real thing.

Plane-wave Burt-Foreman, and why the `ratio` column is gone
------------------------------------------------------------
This driver used `heterostructure.six_band_holes`, i.e. finite differences on the SYMMETRIZED
six-band operator. That operator has no maximum at an abrupt interface -- not a discretization
defect but a property of the operator, which is Legendre-Hadamard elliptic and not strongly
elliptic (`kp_planewave` module docstring). The old `ratio = (0.046/h^2)/V0` column existed to say
how badly contaminated each point was, and it capped this sweep at large islands on coarse grids.

Burt-Foreman ordering removes the cause, so the sweep now runs on `kp_planewave` and the column is
replaced by real convergence evidence per point: `last_step` (the final rung of the cutoff ladder,
which bounds what is left because the ladder is monotone from below) and `n_above_bound` (which
must be zero, and is the acceptance test for the ordering).

The cost does not blow up at large islands, which is why one method can cover the whole range.
`h` is set by the island's smallest feature, so `gmax = pi/2h` SHRINKS as the island grows and
cancels the growing box. Estimated plane-wave counts at 8 cells across the height and a 4 nm crop:
2.5 nm dot 4,849; 10 nm island 3,754; 20 nm island 4,926; the 37 x 3.1 nm island 12,032. Flat to
within a factor of three over a 15x range in size.

The box is the binding axis now, and it bites hardest exactly where the answer is
---------------------------------------------------------------------------------
A six-band hole carries a light-hole admixture whose decay length is sqrt(G0/(m E)) -- 3.7 nm at
m ~ 0.015 against 0.73 nm for the heavy component -- so the k.p box has to be run out even for a
state that looks deeply bound. Measured on the 2.5 nm island at h = 0.30, E_trans over crop pad
2 -> 3 -> 4 -> 6 nm: 0.2279 -> 0.2365 -> 0.2383 -> 0.2391 -> 0.2395 eV.

That decay length diverges as the binding goes to zero, which is a problem for the question this
sweep exists to answer. The critical size is where binding -> 0, and there the required pad grows
without bound; solving at successive sizes and reading off where the level crosses zero would
measure the BOX, the same failure as the 0 meV hole well that grew 23,357 -> 124,418 nm^3 with
padding. So: converge the pad where the state is comfortably bound (`E_hole` above ~30 meV), fit
across that range, and quote the critical size as an extrapolated bracket. `E_hole <= 0` is
reported as `unbound` rather than as a number, because below the far-field matrix valence edge the
level is a resonance and its energy belongs to the box.

Aspect ratio is AR = d/h
------------------------
Diameter over height, the literature convention (Yeap, Rybchenko), so a LARGER AR is a FLATTER
island. This script and its output used h/d until 2026-08-05; records written before then carry
`aspect` and records since carry `AR`. `scripts/migrate_aspect_to_AR.py` converts the old files.

Run:  python scripts/transition_energies.py [--300K] [--electron] [--buffer] [--out FILE]
      python scripts/transition_energies.py --bases 30 --ar 16,12.5,10,8,6.67,5,4
      python scripts/transition_energies.py --bases 30 --ar 4,5,6.67,8 --h 0.5 --out FILE

`--h` pins the grid spacing instead of deriving it from `CELLS_MIN`. **An AR sweep needs it** --
see `run_point` for the 40 meV grid step it removes -- and it needs its own `--out`, because the
resume key carries no `h` and would otherwise collide with the default-grid record.

`--bases` / `--ar` override the default grids and write into the SAME output file, so extra points
densify an existing sweep rather than starting a new one -- the resume key carries base and AR, so
nothing already computed is recomputed. That is how the aspect-ratio curves in
`transition_energies.ipynb` were filled in after the initial 3x3 sweep.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import heterostructure as hs
import kp_planewave as pw
import materials as mt

HC = 1.23984193                 # eV.um
CELLS_MIN, H_FLOOR = 8.0, 0.15  # cells across the height; h never finer than the floor
#: ...and never COARSER than this. `h = height/CELLS_MIN` alone ties the grid to the island, but
#: the plane-wave cutoff is gmax = pi/2h, so that rule SHRINKS the basis as the island grows --
#: while the scale the basis has to resolve (the interface, and the ~0.7 nm heavy-hole decay) is
#: set by the material and does not grow with the dot. Measured in the first production run: the
#: entire AR = 1 column at base >= 6 came back UNCONVERGED, with the last cutoff rung at +7.8 meV
#: (h = 0.75) rising to +25.4 meV (h = 2.5), while every point at h <= 0.625 converged to under
#: 2 meV. The ladder is monotone from below, so those E_hole were UNDERestimates and their E_trans
#: OVERestimates -- which is why this had to be fixed rather than flagged: base 6 AR 1 reported
#: E_trans = +0.011 eV, i.e. a just-open gap, and correcting it can only close it.
#: Raising GMAX_FRACS past 1.0 is not an alternative -- gmax = pi/2h already puts |G - G'| at the
#: grid Nyquist, so a larger cutoff would alias rather than add information.
H_CEIL = 0.5
PAD_XY, PAD_Z = 8.0, 6.0        # STRAIN box. Calibrated in scripts/sweep_calibration.py: at
                                # constant mask, v_top moves only 1.9 meV over pad/base 0.6 -> 4.8
                                # and Ec_far not at all, so 8 nm (= pad/base 0.4 at base 20) is
                                # thin by ratio but worth 1-2 meV. The staircase dominates it --
                                # a 12% vol_err moved v_top 17 meV, which is why E_trans, which
                                # never touches v_top, is what this script quotes.
CROP_PAD = 4.0                  # k.p box, cropped from the strain grid. Calibrated in
                                # scripts/sweep_calibration.py over crop 2/3/4/6 at three corner
                                # geometries: worth about -2 meV on a 2.5 nm island and under
                                # -0.5 meV on a 10 nm one, always UNDERSTATING E_trans. Crop 6
                                # costs 3x the time to buy 1 meV. It is NOT enough near the
                                # critical size, where the binding -> 0 and the light-hole tail
                                # diverges -- hence the extrapolated bracket, see the docstring.
GMAX_FRACS = (0.6, 0.8, 1.0)    # cutoff ladder, as a fraction of pi/2h
STEP_TOL = 5.0                  # meV; flag a point whose last ladder rung exceeds this

#: Island DIAMETER, nm. The old grid was (20, 30, 40) -- entirely inside the regime where these
#: systems do not emit at all (the 37.5 x 3.1 nm island gives E_c - E_hole = -240 meV). The
#: question here is where confinement TURNS ON, so the grid runs down to Yeap's size.
BASES = (2.5, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0)
ASPECT_RATIOS = (1.0, 2.0, 4.0)        # AR = d/h; LARGER IS FLATTER
X_INGASB = (0.50, 1.00)         # In(x)Ga(1-x)Sb; x = 1 IS InSb

#: Matrix sets. `--buffer` switches to the metamorphic InAsSb buffers (Stage 6).
#:
#: Expect the buffer to help less than the misfit reduction suggests. It relieves strain (InSb
#: goes -6.48% -> -4.53% at y = 0.3) but lowers the broken-gap offset and the confinement in
#: near-equal measure, so the two largely cancel in the transition energy -- measured that way
#: earlier in this project. The strained offset is dominated by the biaxial shear raising the
#: heavy hole, which the buffer only partly relieves.
MATRICES_INAS = (('InAs', 'InAs'),)
MATRICES_BUFFER = (('InAsSb0.15', ('InAsSb', 0.15)),
                   ('InAsSb0.30', ('InAsSb', 0.30)))


def hole_ladder(env, crop=CROP_PAD, k=4):
    """Top hole state on the cropped k.p box, with the cutoff ladder that makes it quotable.

    The ladder is monotone from below -- each eigenvalue is a maximum of the Rayleigh quotient
    over the basis, so enlarging the basis can only raise it -- which is what lets `last_step`
    bound the remainder instead of merely suggesting it.

    ARPACK rather than LOBPCG: measured on the 2.5 nm island at h = 0.35, `eigsh` took 29 s
    against 67-71 s for preconditioned LOBPCG for the identical eigenvalue.
    """
    sub = pw.crop_env(env, crop)
    tr = hs.valence_edge_top(sub, reduce='mean', report=True)
    Hp, vp, ladder = None, None, []
    for frac in GMAX_FRACS:
        gmax = frac * 0.5 * pw.nyquist(env['h'])
        H = pw.hamiltonian_from_env(sub, gmax=gmax, ordering='burt-foreman')
        v0 = None if Hp is None else pw.prolong(Hp, vp, H)[:, 0]
        E, V = spla.eigsh(H.as_linear_operator(), k=k, which='LA', tol=1e-8, v0=v0,
                          ncv=min(H.dim - 1, 8 * k))
        order = np.argsort(-E)
        E, V = E[order], V[:, order]
        vp, Hp = V[:, 0], H
        ladder.append(dict(gmax=float(gmax), n_pw=int(H.n_pw), E=float(E[0])))
    bnd = float(pw.spectrum_bound(Hp)['bound'])
    return dict(E=E, sub=sub, ladder=ladder, bound=bnd,
                v_top=float(tr['top_mean']), interior_std=float(tr['interior_std']),
                loc=float(Hp.density(V[:, 0])[sub['mask']].sum()),
                n_above_bound=int((E > bnd + 1e-9).sum()),
                last_step=float((ladder[-1]['E'] - ladder[-2]['E']) * 1e3))


def run_point(x, matrix_spec, base, AR, T, want_electron=False, cells=CELLS_MIN, h_fixed=None):
    """One (composition, matrix, diameter, aspect-ratio, temperature) point. AR = d/h.

    `h_fixed` overrides the usual `h = height / cells` rule and pins the grid spacing instead.

    **Use it for any sweep in AR.** The default rule holds the number of cells across the HEIGHT
    fixed, which means a flatter island is automatically solved on a finer grid: over AR 4 -> 12.5
    at d = 30 nm the in-plane resolution goes 24 -> 60 cells across the diameter. That confounds
    shape with resolution, and it is not a small effect -- `v_top` for InSb jumps 40 meV between
    AR 5 (h = 1.00) and AR 6.67 (h = 0.75), which is a grid step, not a shape step, since the
    Eshelby interior strain varies smoothly with AR. Pinning h makes the AR axis a controlled
    experiment at the cost of losing vertical resolution at high AR -- which the `cells` column
    then reports honestly instead of hiding.
    """
    height = base / AR
    h = float(h_fixed) if h_fixed else min(max(height / cells, H_FLOOR), H_CEIL)
    mt.set_temperature(T)                       # MUST precede build: build stores material copies
    dot = 'InSb' if x >= 1.0 else ('InGaSb', x)
    env = hs.build(hs.ellipsoid(base, height), dot, matrix=matrix_spec, h=h,
                   pad=PAD_XY, z_pad=PAD_Z, use_piezo=False, vol_tol=0.25, verbose=False)

    # The grid must resolve the HEIGHT. Below ~4 cells neighbouring aspect ratios collapse onto
    # the same discrete mask and return bit-identical energies, which reads as a converged
    # plateau; that happened at AR 3.5 and 4 in the Yeap sweep before the guard existed.
    nz = int(env['mask'].any(axis=(0, 1)).sum())
    if nz < 4:
        return dict(status='unresolved', nz=nz, height=height, h=h)

    t0 = time.time()
    hole = hole_ladder(env, crop=CROP_PAD, k=4)
    Eh = float(hole['E'][0])
    v_top = hole['v_top']
    V0 = v_top - env['Ev_far']
    Et = env['Ec_far'] - Eh

    rec = dict(
        status='ok', method='planewave-burt-foreman',
        x=x, matrix=env['matrix']['name'], dot=env['dot']['name'],
        base=base, AR=AR, height=height, T=T, h=h, cells=height / h, nz=nz,
        pad=PAD_XY, z_pad=PAD_Z, crop_pad=CROP_PAD, grid=list(hole['sub']['mask'].shape),
        vol_err=float(env['vol_err']), misfit=float(mt.misfit(env['dot'], env['matrix'])),
        v_top=v_top, interior_std=hole['interior_std'], V0=float(V0),
        Ec_far=float(env['Ec_far']),
        E_hole=Eh, conf=float((v_top - Eh) * 1e3), loc=hole['loc'],
        # Ev_far is the energy zero, so E_hole IS the binding above the far-field matrix valence
        # edge. At or below zero the level is degenerate with the matrix valence continuum: a
        # resonance whose energy belongs to the box, not a bound state.
        binding=float(Eh * 1e3), unbound=bool(Eh <= 0.0),
        levels=[float(hole['E'][i]) for i in range(0, len(hole['E']), 2)],
        E_trans=float(Et), lam=float(HC / Et) if Et > 0 else None,
        ladder=hole['ladder'], last_step=hole['last_step'],
        bound=hole['bound'], n_above_bound=hole['n_above_bound'],
        seconds=time.time() - t0,
    )
    rec['suspect'] = bool(rec['last_step'] > STEP_TOL or rec['n_above_bound']
                          or rec['unbound'] or nz < 6)

    if want_electron:
        # DIAGNOSTIC ONLY -- the strain box is nowhere near large enough for this state, and the
        # number it returns is box quantisation. `electron_bound` is recorded as False regardless
        # of the comparison, because scripts/electron_binding.py has settled it: no bound electron
        # exists at any size in this sweep. Nothing here feeds E_trans.
        el = hs.electron_states(env, k=2, verbose=False, mass_mode='differential')
        Ee = float(np.min(el['E']))
        rec['E_electron_boxlimited'] = Ee
        rec['electron_bound'] = False
        rec['electron_note'] = 'box-limited diagnostic; see scripts/electron_binding.py'
    return rec


def key(x, m, b, AR, T):
    return f"x{x:.2f}_{m}_b{b:g}_AR{AR:.3f}_T{T:g}"


if __name__ == '__main__':
    T = 300.0 if '--300K' in sys.argv else 77.0
    want_e = '--electron' in sys.argv
    MATRICES = MATRICES_BUFFER if '--buffer' in sys.argv else MATRICES_INAS
    suffix = '_buffer' if '--buffer' in sys.argv else ''
    out = (sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv
           else f"_transitions_{T:g}K{suffix}.json")

    bases = ([float(v) for v in sys.argv[sys.argv.index('--bases') + 1].split(',')]
             if '--bases' in sys.argv else list(BASES))
    ars = ([float(v) for v in sys.argv[sys.argv.index('--ar') + 1].split(',')]
           if '--ar' in sys.argv else list(ASPECT_RATIOS))
    h_fixed = float(sys.argv[sys.argv.index('--h') + 1]) if '--h' in sys.argv else None

    done = json.load(open(out)) if os.path.exists(out) else {}
    # Convergence is a property of the SETTINGS, not of the point, so when a setting that controls
    # it changes (H_CEIL, GMAX_FRACS, CROP_PAD) the affected records have to be dropped or resume
    # will keep serving the old, under-converged answers indefinitely.
    if '--redo-unconverged' in sys.argv:
        drop = [k for k, r in done.items()
                if r.get('status') == 'ok' and r.get('last_step', 0.0) > STEP_TOL]
        for k in drop:
            del done[k]
        print(f"--redo-unconverged: dropped {len(drop)} record(s) with last_step > "
              f"{STEP_TOL:g} meV")
    if done:
        print(f"resuming: {len(done)} point(s) already in {out}")
    if any('aspect' in r for r in done.values()):
        sys.exit(f"{out} holds pre-2026-08-05 h/d records; run "
                 f"scripts/migrate_aspect_to_AR.py {out} first")

    print(f"T = {T:g} K   |   E_trans = E_c(matrix, far) - E_hole   |   plane-wave "
          f"Burt-Foreman, strain pad {PAD_XY:g}/{PAD_Z:g} nm, k.p crop {CROP_PAD:g} nm")
    print(f"{'dot':>16} {'base':>6} {'AR=d/h':>7} {'height':>7} {'h':>5} {'cells':>6} "
          f"{'conf':>8} {'bind':>8} {'loc':>6} {'step':>7} {'E_trans':>9} {'lambda':>8}  flag")
    print("-" * 121)
    t_all = time.time()
    for mlabel, mspec in MATRICES:
        for base in bases:
            for AR in ars:
                for x in X_INGASB:
                    k = key(x, mlabel, base, AR, T)
                    # Resume skips COMPLETED points, not merely recorded ones. An 'error' record
                    # used to satisfy `k in done` and so became permanent: the six points that hit
                    # the refine_bandlimited odd-axis bug would have stayed missing on every rerun
                    # after the fix, with nothing in the output to say why. 'unresolved' is kept,
                    # being a deterministic verdict on the geometry (nz < 4) rather than a failure.
                    if done.get(k, {}).get('status') in ('ok', 'unresolved'):
                        continue
                    try:
                        rec = run_point(x, mspec, base, AR, T, want_e, h_fixed=h_fixed)
                    except Exception as exc:
                        rec = dict(status='error', error=f"{type(exc).__name__}: {exc}")
                    done[k] = rec
                    json.dump(done, open(out, 'w'), indent=1)
                    if rec['status'] != 'ok':
                        print(f"{'':>16} {base:6.0f} {AR:7.2f}   {rec['status']}"
                              f" {rec.get('error', '')}", flush=True)
                        continue
                    lam = f"{rec['lam']:7.2f}u" if rec['lam'] else " no gap"
                    flag = ' '.join(f for f, on in (
                        ('UNBOUND', rec['unbound']), ('THIN', rec['nz'] < 6),
                        ('UNCONVERGED', rec['last_step'] > STEP_TOL),
                        ('ABOVE-BOUND', rec['n_above_bound'])) if on)
                    print(f"{rec['dot']:>16} {base:6.1f} {AR:7.2f} {rec['height']:7.2f} "
                          f"{rec['h']:5.2f} {rec['cells']:6.1f} {rec['conf']:7.1f}m "
                          f"{rec['binding']:+7.1f}m {rec['loc']*100:5.1f}% "
                          f"{rec['last_step']:+6.2f}m {rec['E_trans']:+9.3f} {lam:>8}  "
                          f"{flag}", flush=True)
    ok = [r for r in done.values() if r.get('status') == 'ok']
    good = [r for r in ok if not r['suspect']]
    print(f"\n{len(ok)} point(s), {len(good)} not flagged, {time.time()-t_all:.0f}s total -> {out}")
