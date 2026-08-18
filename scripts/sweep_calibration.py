"""What box does the production sweep need? The pad series that fixes PAD_XY and CROP_PAD.

    PYTHONIOENCODING=utf-8 python -u scripts/sweep_calibration.py [--skip N]

`scripts/transition_energies.py` carries three box constants and, until this script, evidence for
one of them. CROP_PAD = 4 nm was measured on Yeap's 2.5 nm island; PAD_XY = 8 / PAD_Z = 6 nm were
inherited. The sweep now runs over a 15x range in island size and down to x = 0.5 InGaSb, so both
need checking where they are worst, not where they were set.

SECTION A -- the strain pad, and why an ABSOLUTE pad is the wrong shape of constant
-----------------------------------------------------------------------------------
Continuum elasticity has no length scale. At fixed shape the strained band edges are therefore
size-independent, and a pad quoted in nanometres is 3.2x the base on a 2.5 nm island and 0.4x on a
20 nm one -- the same constant meaning two very different things. If 8 nm is generous at the small
end and thin at the large end, `v_top` drifts along the size axis of the sweep and reads as
physics: the island getting bigger, not the box getting relatively smaller.

So this sweeps pad/base rather than pad, at three sizes. Scale invariance then does double duty --
it calibrates the pad AND, once converged, the three sizes must agree with each other, which is a
free check on the whole strain chain. Cheap: no eigensolve, and because h scales with the island
too, every size costs the same.

WHAT A FOUND: the pad is not the problem, and the STAIRCASE is
---------------------------------------------------------------
Bases 2.5, 10 and 20 nm returned **bit-identical** numbers at every ratio -- which is the strongest
possible form of the invariance check and also means one series says everything, because scaling h
with the island makes the three geometrically similar and so literally the same discrete problem.
Read that as: nothing in the strain chain carries an absolute length. It is not three independent
confirmations.

The pad itself barely matters. Sorting the rows by `vol_err` splits them into two families, and
WITHIN the constant-mask family (vol_err = +0.044) v_top moves just 1.9 meV over pad/base 0.6 ->
4.8, most of it below 1.2:

    pad/base   0.6     0.8     1.0     1.2     1.6     2.4     3.2     4.8
    v_top   0.5980  0.5985  0.5991  0.5993  0.5997  0.5998  0.5999  0.5999

`Ec_far` is 0.4074 in every row to four decimals, so the electron reference does not care at all.
The production PAD_XY = 8 nm is pad/base 0.4 at base 20 -- thin by this table's standard, but worth
1-2 meV, not the tens of meV the mismatch in ratio suggested.

**The 18.9 meV step between ratio 0.4 and 0.6 is the MASK, not the box.** vol_err goes -0.120 ->
+0.044 there because changing the box changes the grid size and so the island's sub-cell
registration. A 12% volume error is worth 17 meV in v_top, which is far more than anything the
padding does, and it lands wherever the registration lottery puts it rather than trending with
size.

That is a caution about `conf`, not about the deliverable. `E_trans = Ec_far - E_hole` never
touches v_top, and E_hole is measurably robust to the same staircase -- flat to 3 meV over
h = 0.40 -> 0.20 on this island (0.1821 / 0.1788 / 0.1815 / 0.1813 eV) while v_top drifted 16 meV
across the same series. So: keep vol_tol where it is, quote E_trans, and treat the `conf` column
as indicative only.

SECTION B -- the k.p crop, where the LIGHT HOLE sets the requirement
---------------------------------------------------------------------
A six-band hole carries a light-hole admixture with decay length sqrt(G0/(m E)) -- 3.7 nm at
m ~ 0.015 against 0.73 nm for the heavy component. Note what that depends on: the BINDING ENERGY,
not the island size. So the required crop is absolute, and it is worst where the state is most
weakly bound -- which in this sweep is not the extreme of the size axis but the extreme of the
COMPOSITION axis, In(0.5)Ga(0.5)Sb, whose shallower well is the nearest thing here to the critical
size the sweep exists to locate.

Three geometries, chosen to be the corners rather than the middle:

    InSb    base  2.5  AR 1   Yeap's dot; the existing 4 nm measurement, re-run as a regression
    InSb    base 10    AR 4   large and flat -- the box/resolution corner, and the worst n_pw
    InGaSb  base  2.5  AR 1   x = 0.5, the weakest binding in the sweep and so the longest tail

The cutoff needs no such series: GMAX_FRACS tops out at frac = 1.0, i.e. gmax = pi/2h, which puts
|G - G'| at exactly the grid Nyquist pi/h. That is the largest cutoff the potential on this grid
can represent, so it is a ceiling rather than a choice, and `last_step` already reports what is
left underneath it. Measured, it is 1.0-1.2 meV at every geometry below -- so the cutoff is not
what limits this sweep, and CROP_PAD is.

WHAT B FOUND: 4 nm costs about 2 meV, and the composition worry did not bite
-----------------------------------------------------------------------------
E_trans (eV) against crop pad, all at 77 K:

    crop                  2.0      3.0      4.0      6.0     step 3->4  step 4->6
    InSb    b2.5 AR1   0.2260   0.2325   0.2349   0.2360      -2.33m     -1.15m
    InGaSb  b2.5 AR1   0.2481   0.2540   0.2562   0.2572      -2.11m     -1.07m
    InSb    b10  AR4  -0.0882  -0.0912  -0.0918  -0.0922      +0.67m     +0.35m

The steps halve each time, so extrapolating leaves ~1 meV beyond crop 6 and CROP_PAD = 4 is about
**2 meV short on a small island and under 0.5 meV on a large one**, always in the direction of
UNDERSTATING E_trans. That is inside STEP_TOL and an order of magnitude under the ~80 meV standing
disagreement with Yeap, so 4 nm stays -- crop 6 costs 3x the time (341 s vs 123 s at b2.5) to buy
1 meV.

**The hypothesis that picked InGaSb as a corner was wrong, and cheaply so.** The reasoning was that
x = 0.5 binds more weakly (151 vs 172 meV) and the light-hole decay length goes as sqrt(G0/(m E)),
so a shallower well should need a longer pad. It does not: its steps (-5.98, -2.11, -1.07) track
InSb's (-6.49, -2.33, -1.15) to within 0.5 meV. A 12% change in binding is simply too small to
move a square root, and the composition axis of the sweep therefore needs no separate pad. The
place that argument WILL bite is the critical size itself, where the binding goes to zero rather
than down by 12% -- which is why that is quoted as an extrapolated bracket and not solved directly.

Note b10 AR4's steps run the OTHER WAY (+2.97, +0.67, +0.35), and it is the one geometry whose
E_trans is negative -- deeply bound (88% localised) and in the no-emission regime, where the tight
box was lowering rather than raising the level, most likely through periodic images across its
thin z direction. It converges just as cleanly; only the sign of the approach differs.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import heterostructure as hs
import materials as mt
from transition_energies import CELLS_MIN, H_FLOOR, GMAX_FRACS, hole_ladder

T_K = 77.0
OUT = '_sweep_calibration.json'

#: (label, dot spec, base nm, AR) for section B.
GEOMETRIES = (('InSb b2.5 AR1', 'InSb', 2.5, 1.0),
              ('InSb b10 AR4', 'InSb', 10.0, 4.0),
              ('InGaSb0.5 b2.5 AR1', ('InGaSb', 0.5), 2.5, 1.0))


def build_at(dot, base, AR, pad, z_pad=None, h=None):
    """One `build`, with h from the usual CELLS_MIN rule unless pinned."""
    height = base / AR
    hh = float(h) if h else max(height / CELLS_MIN, H_FLOOR)
    mt.set_temperature(T_K)                     # MUST precede build: build stores material copies
    return hs.build(hs.ellipsoid(base, height), dot, matrix='InAs', h=hh, pad=pad,
                    z_pad=z_pad if z_pad is not None else pad * 0.75,
                    use_piezo=False, vol_tol=0.25, verbose=False)


def section_a(bases=(2.5, 10.0, 20.0), ratios=(0.4, 0.8, 1.6, 3.2), AR=1.0, store=None):
    """Strained band edges vs pad/base, at three sizes. No eigensolve."""
    print("=" * 92)
    print(f"A. STRAIN PAD. Band edges vs pad/base at AR {AR:g}; elasticity is scale-free, so the"
          f"\n   converged rows must agree ACROSS sizes as well as settle within one.")
    print(f"   {'base':>6} {'pad/base':>9} {'pad':>7} {'h':>6} {'grid':>14} {'vol err':>8} "
          f"{'v_top':>9} {'Ec_far':>8} {'d v_top':>9}")
    out = {}
    for base in bases:
        prev = None
        for ratio in ratios:
            pad = ratio * base
            t0 = time.time()
            env = build_at('InSb', base, AR, pad)
            tr = hs.valence_edge_top(env, reduce='mean', report=True)
            v = float(tr['top_mean'])
            d = '' if prev is None else f"{(v-prev)*1e3:+8.2f}m"
            print(f"   {base:6.1f} {ratio:9.2f} {pad:7.1f} {env['h']:6.3f} "
                  f"{str(list(env['mask'].shape)):>14} {env['vol_err']:+8.3f} {v:9.4f} "
                  f"{env['Ec_far']:8.4f} {d:>9}  {time.time()-t0:.0f}s", flush=True)
            prev = v
            out[(base, ratio)] = v
            if store is not None:
                store[f"A_b{base:g}_AR{AR:g}_r{ratio:g}"] = dict(
                    base=base, AR=AR, ratio=ratio, pad=pad, h=float(env['h']),
                    v_top=v, Ec_far=float(env['Ec_far']), Ev_far=float(env['Ev_far']),
                    vol_err=float(env['vol_err']), interior_std=float(tr['interior_std']))
                json.dump(store, open(OUT, 'w'), indent=1)

    print("\n   Scale invariance across sizes, at the largest pad/base:")
    r = ratios[-1]
    vs = [out[(b, r)] for b in bases]
    for b, v in zip(bases, vs):
        print(f"     base {b:5.1f} nm   v_top {v:.4f} eV")
    print(f"   -> spread {(max(vs)-min(vs))*1e3:.2f} meV. This is exactly zero in the continuum,"
          f"\n      so it is the residual of grid + box + staircase, and a floor on the sweep.")
    return out


def section_b(crops=(2.0, 3.0, 4.0, 6.0), store=None, only=None):
    """Hole level vs k.p crop pad, at the three corner geometries. The expensive one."""
    print("\n" + "=" * 92)
    print("B. K.P CROP PAD. The light-hole tail is set by the BINDING, so the worst case is the")
    print("   shallowest well (InGaSb x = 0.5), not the largest island.")
    for label, dot, base, AR in GEOMETRIES:
        if only is not None and only not in label:
            continue
        env = build_at(dot, base, AR, 8.0, z_pad=6.0)
        print(f"\n   {label}   h = {env['h']:.3f} nm, strain box {list(env['mask'].shape)}")
        print(f"   {'crop':>6} {'grid':>14} {'n_pw':>7} {'E_hole':>9} {'d':>9} {'E_trans':>9} "
              f"{'conf':>8} {'loc':>7} {'step':>8}")
        prev = None
        for crop in crops:
            t0 = time.time()
            try:
                r = hole_ladder(env, crop=crop, k=4)
            except Exception as exc:
                print(f"   {crop:6.1f}   {type(exc).__name__}: {exc}", flush=True)
                continue
            Eh = float(r['E'][0])
            d = '' if prev is None else f"{(Eh-prev)*1e3:+8.2f}m"
            row = dict(label=label, base=base, AR=AR, crop=crop, h=float(env['h']),
                       grid=list(r['sub']['mask'].shape), n_pw=r['ladder'][-1]['n_pw'],
                       E_hole=Eh, E_trans=float(env['Ec_far'] - Eh), v_top=r['v_top'],
                       conf=float((r['v_top'] - Eh) * 1e3), loc=r['loc'],
                       last_step=r['last_step'], n_above_bound=r['n_above_bound'],
                       seconds=time.time() - t0)
            print(f"   {crop:6.1f} {str(row['grid']):>14} {row['n_pw']:7,} {Eh:9.4f} {d:>9} "
                  f"{row['E_trans']:9.4f} {row['conf']:7.1f}m {row['loc']*100:6.1f}% "
                  f"{row['last_step']:+7.2f}m  {row['seconds']:.0f}s", flush=True)
            prev = Eh
            if store is not None:
                store[f"B_{label.replace(' ', '_')}_c{crop:g}"] = row
                json.dump(store, open(OUT, 'w'), indent=1)
    print("\n   The transition energy is what the sweep quotes, and unlike a confinement energy it")
    print("   never touches v_top -- so read the E_trans column, not conf.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', type=int, default=0, help="skip the first N sections")
    ap.add_argument('--crops', type=float, nargs='*', default=[2.0, 3.0, 4.0, 6.0])
    ap.add_argument('--only', default=None, help="substring of one GEOMETRIES label")
    a = ap.parse_args()

    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if a.skip < 1:
        section_a(store=store)
    section_b(crops=tuple(a.crops), store=store, only=a.only)
