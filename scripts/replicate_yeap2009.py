"""Replicate Yeap, Rybchenko, Itskevich & Haywood, Phys. Rev. B 79, 075305 (2009).

Their method is this package's method, which is what makes this a real validation rather than a
loose comparison:

    strain      continuum elasticity, cubic anisotropy, separate dot/matrix elastic constants
                (they use FEM/COMSOL, this uses finite difference; they cite FD as equivalent)
    band edges  standard deformation-potential theory
    holes       six-band k.p
    electrons   single-band effective mass, band-gap change folded into the electron mass
    couplings   conduction and valence DECOUPLED; exciton binding neglected
    omitted     wetting layer ("existence still doubtful"), piezoelectric potential ("negligible
                for the dot sizes considered" -- measured here as 0.2 meV, so: agreed)
    parameters  their Ref. 9 = Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001),
                i.e. exactly `materials.py`, on the same zero (unstrained InSb valence edge)

Geometry: base b = 2.5 nm FIXED, aspect ratio AR = d/h, so height = 2.5/AR nm. AR = 4 is a FLAT
island, not a tall one. This repo now uses the same convention throughout (it quoted the
reciprocal, h/d, until 2026-08-05), so no conversion is needed here any more.

Composition: their x is the ARSENIC fraction in InAs(x)Sb(1-x). `materials.alloy('InAsSb', x)`
takes x as the ANTIMONY fraction. So x_mine = 1 - x_theirs. Getting this backwards silently
produces a plausible-looking curve running the wrong way.

Targets read from their figures at 80 K:
  Fig 6(b), pure InSb vs AR:      1.0->0.16  1.5->0.22  2.0->0.29  2.5->0.36  3.0->0.42
                                  3.5->0.46  4.0->0.48 eV
  Fig 7(b), AR = 2 vs x_As:       0.0->0.29  0.1->0.32  0.2->0.35  0.3->0.38  0.4->0.42 eV
  Their text: "An AR of ~2.5 is needed for a transition energy of 0.34 eV at 80 K".

Run:  python scripts/replicate_yeap2009.py [--fine] [--300K]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import heterostructure as hs
import materials as mt

HC = 1.23984193
BASE = 2.5                      # nm, their fixed base diameter
CELLS_PER_HEIGHT = 8.0
PAD_XY = PAD_Z = 2.5            # a hole bound by ~800 meV decays well within this

FIG6 = {1.0: 0.16, 1.5: 0.22, 2.0: 0.29, 2.5: 0.36, 3.0: 0.42, 3.5: 0.46, 4.0: 0.48}
FIG7 = {0.0: 0.29, 0.1: 0.32, 0.2: 0.35, 0.3: 0.38, 0.4: 0.42}   # keyed by x_As, AR = 2


def run_point(AR, x_as=0.0, T=80.0, cells=CELLS_PER_HEIGHT, shape='ellipsoid'):
    """One (aspect ratio, composition) point. Returns the type-II transition energy in eV."""
    mt.set_temperature(T)
    height = BASE / AR
    h = max(height / cells, 0.08)
    geom = (hs.ellipsoid(BASE, height) if shape == 'ellipsoid'
            else hs.spherical_lens(BASE / 2.0, height))
    dot = 'InSb' if x_as <= 0.0 else ('InAsSb', 1.0 - x_as)      # their x_As -> our x_Sb
    env = hs.build(geom, dot, matrix='InAs', h=h, pad=PAD_XY, z_pad=PAD_Z,
                   use_piezo=False, vol_tol=0.60, verbose=False)

    holes = hs.six_band_holes(env, k=12, verbose=False)
    E_h = float(holes['E'][0])

    # Their electron is "loosely bound around the dot" in shallow strain-induced wells in the
    # InAs -- a matrix state a few meV below E_c(far), not a dot state. Take the computed level
    # if it is bound, else the far-field edge, which is their own approximation.
    try:
        el = hs.electron_states(env, k=4, verbose=False)
        E_e = float(np.min(np.asarray(el['E'])))
        if not np.isfinite(E_e) or E_e > env['Ec_far']:
            E_e = float(env['Ec_far'])
    except Exception:
        E_e = float(env['Ec_far'])

    return dict(AR=AR, x_as=x_as, T=T, height=height, h=h, E_e=E_e, E_h=E_h,
                top=holes['top'], loc=float(holes['loc'][0]),
                conf=(holes['top'] - E_h) * 1e3, Et=E_e - E_h,
                sites=int(env['mask'].size), vol_err=float(env['vol_err']))


def table(rows, targets, key, label, T):
    print(f"\n{label}  (T = {T:g} K)")
    print(f"{key:>7} {'height':>7} {'grid':>6} {'sites':>9} {'conf':>8} {'loc':>6} "
          f"{'E_trans':>9} {'lambda':>8} {'Yeap':>7} {'diff':>8}")
    print("-" * 88)
    errs = []
    for r in rows:
        k = r[key]
        tgt = targets.get(round(k, 3))
        d = r['Et'] - tgt if tgt is not None else float('nan')
        if tgt is not None:
            errs.append(d)
        lam = HC / r['Et'] if r['Et'] > 0 else float('nan')
        print(f"{k:7.2f} {r['height']:6.2f}n {r['h']:5.3f}n {r['sites']:>9,} "
              f"{r['conf']:7.1f}m {r['loc']*100:5.1f}% {r['Et']:+9.3f} {lam:7.2f}u "
              f"{(tgt if tgt is not None else float('nan')):6.2f} {d:+8.3f}")
    if errs:
        e = np.array(errs)
        print(f"{'':>7} mean signed error {e.mean():+.3f} eV   "
              f"RMS {np.sqrt((e**2).mean()):.3f} eV   max |err| {np.abs(e).max():.3f} eV")


if __name__ == '__main__':
    T = 300.0 if '--300K' in sys.argv else 80.0
    cells = 12.0 if '--fine' in sys.argv else CELLS_PER_HEIGHT
    t0 = time.time()

    print(f"Replicating Yeap et al. PRB 79, 075305 (2009), ellipsoidal InSb/InAs, "
          f"base {BASE} nm, piezo off, no wetting layer")
    rows6 = [run_point(AR, 0.0, T, cells) for AR in sorted(FIG6)]
    table(rows6, FIG6 if T == 80.0 else {}, 'AR', 'Fig 6(b): transition energy vs AR', T)

    rows7 = [run_point(2.0, x, T, cells) for x in sorted(FIG7)]
    table(rows7, FIG7 if T == 80.0 else {}, 'x_as',
          'Fig 7(b): transition energy vs arsenic fraction, AR = 2', T)
    print(f"\ntotal {time.time()-t0:.0f}s")
