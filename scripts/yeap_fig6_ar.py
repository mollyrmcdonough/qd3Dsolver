"""Yeap et al. Fig. 6: hole level and transition energy vs aspect ratio, at the real dot size.

    PYTHONIOENCODING=utf-8 python -u scripts/yeap_fig6_ar.py
    PYTHONIOENCODING=utf-8 python -u scripts/yeap_fig6_ar.py --ar 1 2 4     # a subset
    PYTHONIOENCODING=utf-8 python -u scripts/yeap_fig6_ar.py --h 0.25       # grid check

Source: G. H. Yeap, S. I. Rybchenko, I. E. Itskevich and S. K. Haywood, Phys. Rev. B 79, 075305
(2009). Their Fig. 1 fixes the geometry -- an oblate ellipsoid of base b = 2.5 nm with
**AR = b/h**, so raising AR flattens the dot at fixed base. Sec. III B: "we use a fixed value of
the dot base size of 2.5 nm; i.e., varying the AR implies varying the dot height."

Why this sweep is the test, and not the AR = 1 point alone
-----------------------------------------------------------
At AR = 1 we agree with them on the well top, agree with their electron to 17 meV, and differ on
the HOLE by 56. One point cannot say why; seven can, and did.

WHAT IT FOUND (h = 0.25 nm, crop 4 nm, cutoff-converged to <= 2.5 meV per point)
--------------------------------------------------------------------------------
1. **The band edges agree at EVERY aspect ratio.** Our strained dot heavy-hole edge against their
   Fig. 6(a) dashed line: -10, -12, +2, -5, -6 meV at AR 1 .. 3. The strain half of this
   calculation -- elasticity, deformation potentials, the whole Bir-Pikus chain -- tracks theirs
   across the entire shape range, not just at one point. Whatever the disagreement is, it is not
   the band structure.

2. **The disagreement is purely the size-quantization energy, and it is not a constant offset.**
   Hole level, ours minus theirs: -56, -37, -1, +45, +78 meV at AR 1 .. 3, crossing zero near
   AR 2. Their transition energy climbs twice as fast with aspect ratio as ours. That rules out
   the tidy explanation -- a systematic shift from the operator ordering -- which would have been
   flat.

3. **The first hole excitation at AR 1 is 147 meV**, against the "about 140 meV" their Sec. III B
   states. An independent check on the curvature of the well that does not involve the absolute
   level at all, and it passes.

IS THE FLAT-END DISAGREEMENT OURS? MEASURED: NO
-------------------------------------------------
The obvious worry was our own convergence, since the disagreement grows exactly where the state
gets harder to compute. Checked at AR 3, where the gap is +78 meV:

    baseline   h = 0.250, crop 4 nm   E = +0.0482
    box        h = 0.250, crop 6 nm   E = +0.0470    -1.2 meV
    grid       h = 0.175, crop 4 nm   E = +0.0440    -4.2 meV

A few meV against 78. The disagreement is real.

The same check at AR 4, where the state is weakest (19 meV of binding, so the light-hole tail is
sqrt(G0/(m E)) ~ 11 nm against a 4 nm crop and the box is the axis most likely to bite):

    baseline   h = 0.150, crop 4 nm   E = +0.0216
    box        h = 0.150, crop 6 nm   E = +0.0189    -2.7 meV

Bigger than at AR 3, as expected, and still forty times smaller than the 109 meV disagreement.
Both flat-end points are therefore converged in cutoff AND in box.

One point on OUR side genuinely was broken, and it is worth knowing how it presented. At h = 0.25
the AR 3.5 and AR 4 islands are both 2 cells thick -- the SAME discrete object -- and the solver
returned bit-identical energies for two different geometries, which read as a converged plateau in
the transition energy. `solve_point` now refuses below 4 cells across the height. Those two points
are excluded; reaching them needs h <= height/4, i.e. h <= 0.16 nm at AR 4.

WHERE THE FLAT-END DISAGREEMENT PROBABLY COMES FROM: THEIR HOLE STOPS BEING BOUND
----------------------------------------------------------------------------------
This package's energy zero IS the unstrained matrix valence edge, so `E_hole` is literally the
binding above the far field. Below zero the level sits inside the matrix valence continuum: not a
bound state, a resonance, whose energy in any finite domain is a property of that domain.

    AR          1     1.5       2     2.5       3     3.5       4
    ours     +174    +133     +99     +75     +48       -       -     meV, all BOUND
    theirs   +230    +170    +100     +30     -30     -70     -90     meV

Their levels cross into the continuum between AR 2.5 and 3 -- exactly where the disagreement takes
off. A finite FEM domain still returns a discrete eigenvalue there, and a larger domain would let
it relax back up toward the continuum edge, i.e. toward ours. That is a hypothesis about their
calculation which cannot be checked from here, but it is consistent in sign, in magnitude, and in
where it switches on. It also means the comparison is not well posed past AR ~2.5 for either of
us, and neither paper says so.

**Read directly off their Fig. 6(a), not inferred from ours.** The HH1 diamonds there sit at about
-0.34, -0.42, -0.49, -0.55, -0.60, -0.645, -0.665 eV for AR 1 .. 4 on their zero, so they pass
-0.590 -- the unstrained InAs valence edge, which IS the far-field continuum threshold -- between
AR 2.5 and AR 3. The same crossing, from their own numbers rather than from the subtraction.

**Why their figure hides it.** Fig. 6(a) does draw a "VB-InAs" reference line, but it runs from
about -0.79 at AR 1 to -0.72 at AR 4. The far-field InAs valence edge cannot vary with aspect
ratio -- it is -0.590 by the definition of their zero -- so that line is the BAND-BENT value at
the interface, which their own Sec. III A describes ("downward band bending, which increases the
heavy-hole confinement even further"). Their HH1 stays comfortably above that curve at every AR,
which makes the levels look bound throughout. Against the threshold that actually governs escape
to infinity, the last three are not. This repo's Ev_far = 0 convention is what made it visible.

**The strongest evidence is the SHAPE of the two curves past AR 3.** At h = 0.15 (so the mask is
4.8 and 4.2 cells thick) AR 3.5 and AR 4 converge to 0.11 and 0.12 meV on the last cutoff rung:

    AR            1     1.5       2     2.5       3     3.5       4
    ours       +174    +133     +99     +75     +48     +24     +22   meV above the far field
    theirs     +230    +170    +100     +30     -30     -70     -90

Ours SATURATES as it approaches zero -- +48, +24, +22 -- which is what a level being squeezed out
of a shrinking well must do, because zero is the continuum edge and there is nothing below it to
fall into. Theirs continues through it in a near-straight line, -30, -70, -90, with no feature at
the crossing whatsoever. A discrete level that does not notice the continuum edge is the signature
of a state that is being held discrete by a boundary condition. Localisation over the same range
falls 28.3% -> 17.5% -> 15.8% (random-vector baseline here is well under 1%).

**A third, independent sign of the same thing.** Their Sec. III B states "the separation between
the levels should be about 140 meV" over this AR range. Ours is 147 meV at AR 1 -- agreeing -- but
then collapses: 99, 53, 43, 32 meV at AR 1.5 .. 3. A level spacing that stays near 140 meV in a
well whose depth is fixed and whose height is shrinking to under a nanometre is what a
too-confined hole looks like, and it is the same disagreement seen in the excitation spectrum
rather than in the ground state.

Energy zeros differ and it matters. Yeap quote everything against the unstrained **InSb** valence
edge; this package uses the unstrained **matrix** (InAs) valence edge. The offset is the VBO,
+0.590 eV, applied here rather than left to the reader.

IS IT THE PIEZOELECTRIC POTENTIAL? NO -- 4 ueV, MEASURED
----------------------------------------------------------
A real model difference, and worth ruling out by measurement rather than by argument, because
`hs.build` defaults `use_piezo=True` while Yeap p. 1 states plainly: "The piezoelectric potential
is not included either because it is expected to be negligible for the dot sizes considered."

Same geometry, same grid, `--no-piezo` the only change:

    AR 1   ON 0.173744687   OFF 0.173740823   difference +3.9 ueV
    AR 3   ON 0.048153969   OFF 0.048152760   difference +1.2 ueV

Four THOUSANDTHS of a meV, against a 56-78 meV disagreement. Localisation is unchanged to 0.1%.
The reason it is so small is structural rather than accidental: phi is +-5 meV over the whole box
and only +-1 meV inside the island, and e14 is the single independent piezoelectric constant of a
zincblende crystal, so phi is exactly C4-ANTISYMMETRIC about [001] -- phi(y,-x,z) = -phi(x,y,z),
verified here to 1e-14. Its average over any C4-symmetric region is therefore identically zero,
which kills the first-order shift of a symmetric ground state; what survives is second order in a
1 meV perturbation. This also updates the "0.2 meV" quoted in README from an earlier, much larger
geometry.

IS THE DISAGREEMENT JUST THE ENERGY ZERO? NO -- CHECKED, AND WORTH KEEPING CHECKED
-----------------------------------------------------------------------------------
The natural suspicion, since the two zeros differ by 590 meV and the disagreement is ~10% of that.
Three independent reasons it is not:

1. **The transition energy never touches the zero.** Theirs is E1 - HH1, both read inside their own
   figure; ours is Ec(InAs, far) - E_hole, both on our zero. A difference of two differences: any
   common shift of either scale cancels identically. The disagreement lives there.
2. **`off` is pinned by the dot valence edge**, which is converted with the very same constant and
   agrees to -10 .. +2 meV at every AR. An offset wrong by d would put this column out by d.
3. **`off` is pinned again, independently, by their electron.** E1 = -0.20 on their zero maps to
   +0.3900 on ours, i.e. 17 meV below our Ec(InAs, far) = 0.4068 eV -- the weakly bound matrix
   state they describe. Wrong by 56 meV and their electron would be 73 meV deep or 39 meV unbound,
   both contradicting their own text.

And the shape is wrong for it regardless: a zero error is ONE CONSTANT, while the hole
disagreement runs -56 -> +78 meV and changes sign at AR 2. Removing the best constant (+6 meV)
leaves -62, -43, -6, +39, +72 -- the whole 134 meV spread survives.

What the zero DOES cost is ~10 meV: Yeap state the unstrained InSb/InAs valence offset is "about
600 meV" where our Vurgaftman tables give 590. That is the floor under item 2 above, and a fifth
of the AR 1 discrepancy.

Their electron is not at the band edge either: E1 in their Fig. 6(a) sits flat at -0.20 eV, i.e.
17 meV BELOW Ec(InAs, far) -- what they describe as a weakly bound matrix state in the tensile
shell. **We solve that state and find it unbound**, so the primary column here puts the electron
at Ec(far) and the `+e` column only shows what their 17 meV would do.

`scripts/electron_binding.py` runs the box series that decides it: over L = 20 -> 60 nm the level
tracks the empty-box quantum 3*G0*pi^2/(m L^2) down to 0.03-0.06 meV at every step -- pure 1/L^2,
no exponential settling -- and the shell contributes ~0.1 meV. Three dimensions have a finite
shallow-well threshold, and this well is nowhere near it: 150 meV deep at its deepest but only
~1 nm thick, V0*R^2 ~ 0.15 eV.nm^2 against the ~3.6 eV.nm^2 needed at m = 0.026. The island is
also a 810 meV BARRIER, so the net of barrier-up and shell-down stays repulsive at 2.5, 10 and
20 nm alike. Their 17 meV is then the same kind of artefact as their flat-AR hole levels: the
lowest state of a finite FEM domain, read as a bound state.

Their two panels disagree with each other by about 20 meV: Fig. 6(a) implies E1 - HH1 = 0.14 eV
at AR 1 where Fig. 6(b) plots 0.16. Fig. 6(b) is taken as primary here. That 20 meV is the floor
on any comparison with this paper.
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse.linalg as spla

import materials as mt
import heterostructure as hs
import kp_planewave as pw

warnings.filterwarnings('ignore', category=UserWarning)

BASE = 2.5                    # nm, their fixed base
T_K = 80.0
STRAIN_PAD = 6.0
CROP_PAD = 4.0
ARS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
OUT = '_yeap_fig6_ar.json'

#: Yeap Fig. 6(b), 80 K, read off the published figure. Uncertainty ~20 meV: their own Fig. 6(a)
#: implies E1 - HH1 = 0.14 eV at AR 1 where 6(b) plots 0.16, so the paper is internally
#: inconsistent by about that much.
YEAP_ETRANS = {1.0: 0.16, 1.5: 0.22, 2.0: 0.29, 2.5: 0.36, 3.0: 0.42, 3.5: 0.46, 4.0: 0.48}

#: Yeap Fig. 6(a): the electron level, flat across AR, on THEIR zero (unstrained InSb VB).
YEAP_E1 = -0.20

#: Yeap Fig. 6(a): the strained InSb heavy-hole edge (dashed), on their zero. Read off the figure.
YEAP_VB_INSB = {1.0: 0.02, 1.5: 0.05, 2.0: 0.07, 2.5: 0.09, 3.0: 0.11, 3.5: 0.12, 4.0: 0.13}


def vbo_offset():
    """Add this to a Yeap energy to put it on our zero. Their zero is the unstrained InSb valence
    edge, ours the unstrained InAs one."""
    return mt.material('InSb')['VBO'] - mt.material('InAs')['VBO']


def solve_point(AR, h, gmax_fracs=(0.6, 0.8, 1.0), k=6, verbose=True, crop=CROP_PAD,
                piezo=True):
    """One aspect ratio, with the cutoff ladder that makes the number quotable.

    ARPACK, not LOBPCG, and this was measured rather than assumed. Burt-Foreman makes the operator
    bounded, which removes the reason LOBPCG failed before, so it was retried here: on the 2.5 nm
    island at h = 0.35 (dim 5,370) `eigsh` took 29 s and preconditioned LOBPCG took 67-71 s for
    the identical eigenvalue (agreeing to 0.0 ueV), at both preconditioner shifts tried. 2.5x
    slower, so ARPACK stays. The inverse-diagonal preconditioner is evidently not strong enough
    against the spectral spread here; a Teter-Payne-Allan-style adaptive one might be, and has not
    been tried.

    `piezo` is exposed because `hs.build` defaults it ON while **Yeap carry no piezoelectric term
    at all**, so the default sweep compares two slightly different Hamiltonians. Whether that
    matters is measured, not assumed -- see the module docstring.
    """
    mt.set_temperature(T_K)
    height = BASE / AR
    shape = hs.ellipsoid(BASE, height)
    env = hs.build(shape, 'InSb', matrix='InAs', h=h, pad=STRAIN_PAD, vol_tol=0.30,
                   use_piezo=piezo, verbose=False)
    # THE GRID MUST RESOLVE THE HEIGHT, and this is not a formality at the flat end. At h = 0.25
    # the AR 3.5 and AR 4 masks are both 2 cells thick -- the SAME discrete object -- and the
    # solver duly returned bit-identical energies for two different geometries. That looked like a
    # converged plateau and was a resolution failure. Four cells is already thin; below that,
    # refuse rather than return a number.
    nz = int(env['mask'].any(axis=(0, 1)).sum())
    if nz < 4:
        raise ValueError(
            f"AR {AR:g} (height {height:.3f} nm) is only {nz} cells thick at h = {h}: the grid "
            f"cannot resolve this shape, and neighbouring aspect ratios will collapse onto the "
            f"same mask. Refine h to <= {height/4:.3f} nm.")

    sub = pw.crop_env(env, crop)
    tr = hs.valence_edge_top(sub, reduce='mean', report=True)

    Hp, vp, ladder = None, None, []
    for frac in gmax_fracs:
        gmax = frac * 0.5 * pw.nyquist(h)
        H = pw.hamiltonian_from_env(sub, gmax=gmax, ordering='burt-foreman')
        v0 = None if Hp is None else pw.prolong(Hp, vp, H)[:, 0]
        t0 = time.time()
        E, V = spla.eigsh(H.as_linear_operator(), k=k, which='LA', tol=1e-8, v0=v0,
                          ncv=min(H.dim - 1, 8 * k))
        order = np.argsort(-E)
        E, V = E[order], V[:, order]
        vp, Hp = V[:, 0], H
        ladder.append(dict(gmax=float(gmax), n_pw=int(H.n_pw), E=float(E[0]),
                           seconds=time.time() - t0))
        if verbose:
            d = '' if len(ladder) < 2 else f"{(E[0]-ladder[-2]['E'])*1e3:+7.2f}"
            print(f"      gmax {gmax:5.2f}  n_pw {H.n_pw:6,}  E {E[0]:+.4f}  step {d:>7} meV  "
                  f"{ladder[-1]['seconds']:5.0f}s", flush=True)

    loc = float(Hp.density(V[:, 0])[sub['mask']].sum())
    bound = pw.spectrum_bound(Hp)['bound']
    # Kramers pairs: every level is exactly twofold, so the distinct levels are every other one.
    levels = [float(E[i]) for i in range(0, len(E), 2)]
    kramers = float(max(abs(E[i] - E[i + 1]) for i in range(0, len(E) - 1, 2)))

    return dict(AR=AR, h=h, height=height, base=BASE, piezo=bool(piezo),
                phi_range=[float(env['phi'].min()), float(env['phi'].max())],
                phi_island=[float(env['phi'][env['mask']].min()),
                            float(env['phi'][env['mask']].max()),
                            float(env['phi'][env['mask']].mean())],
                vol_err=float(env['vol_err']), grid=list(sub['mask'].shape),
                crop_pad=crop, strain_pad=STRAIN_PAD,
                v_top_mean=float(tr['top_mean']), v_top_max=float(tr['top_max']),
                interior_std=float(tr['interior_std']),
                E_hole=float(E[0]), levels=levels, kramers=kramers,
                Ec_far=float(sub['Ec_far']), loc=loc, bound=bound,
                n_above_bound=int((E > bound + 1e-9).sum()),
                ladder=ladder, last_step=float((ladder[-1]['E'] - ladder[-2]['E']) * 1e3))


def report(rows):
    off = vbo_offset()
    print(f"\n{'='*100}", flush=True)
    print(f"Yeap Fig. 6 comparison. Their zero is the unstrained InSb valence edge; ours the "
          f"unstrained\nInAs one. Offset +{off:.3f} eV, applied to THEIR numbers below.",
          flush=True)
    print(f"\n{'AR':>4} {'height':>7} | {'v_top':>7} {'Yeap':>7} {'d':>6} | "
          f"{'E_hole':>8} {'Yeap':>8} {'d':>6} | {'E_tr':>6} {'+e':>6} {'Yeap':>6} {'d':>6} | "
          f"{'loc':>6} {'step':>7}", flush=True)
    for r in rows:
        AR = r['AR']
        y_tr = YEAP_ETRANS[AR]
        y_hh = YEAP_E1 - y_tr + off               # their hole level, on our zero
        y_vt = YEAP_VB_INSB[AR] + off             # their dot edge, on our zero
        e_tr = r['Ec_far'] - r['E_hole']          # electron at Ec(far)
        e_tr_e1 = e_tr - 0.017                    # with their 17 meV electron binding
        print(f"{AR:4g} {r['height']:7.3f} | {r['v_top_mean']:7.4f} {y_vt:7.4f} "
              f"{(r['v_top_mean']-y_vt)*1e3:+6.0f} | {r['E_hole']:+8.4f} {y_hh:+8.4f} "
              f"{(r['E_hole']-y_hh)*1e3:+6.0f} | {e_tr:6.3f} {e_tr_e1:6.3f} {y_tr:6.2f} "
              f"{(e_tr-y_tr)*1e3:+6.0f} | {r['loc']*100:5.1f}% {r['last_step']:+7.2f}",
              flush=True)
    print("\n  d columns are ours minus theirs, in meV. 'E_tr' is the physical one -- the electron"
          "\n  sits at Ec(InAs, far) because scripts/electron_binding.py finds NO bound state in"
          "\n  the tensile shell; the final d is against it. '+e' is kept only to show what their"
          "\n  17 meV would do. 'step' is the last rung of the cutoff ladder, in meV.", flush=True)

    d = [(r['E_hole'] - (YEAP_E1 - YEAP_ETRANS[r['AR']] + off)) * 1e3 for r in rows]
    print(f"\n  Hole-level offset across AR: {min(d):+.0f} to {max(d):+.0f} meV, "
          f"spread {max(d)-min(d):.0f} meV.", flush=True)
    print("  NOT a constant offset, so not the operator ordering. And not our convergence "
          "either:", flush=True)
    print("  at AR 3 the box (crop 4 -> 6 nm) moves the level 1.2 meV and the grid (h 0.25 -> "
          "0.175)", flush=True)
    print("  4.2 meV, against a 78 meV disagreement. See the module docstring for where it "
          "probably", flush=True)
    print("  does come from -- their hole level leaves the bound spectrum around AR 2.75.",
          flush=True)

    print(f"\n  Is the hole BOUND? Our zero is the unstrained matrix valence edge, so E_hole is"
          f"\n  the binding above the far field; below zero the level lies in the matrix valence"
          f"\n  continuum -- a resonance, whose energy in any finite domain is a property of that"
          f"\n  domain rather than of the dot.", flush=True)
    for r in rows:
        y = YEAP_E1 - YEAP_ETRANS[r["AR"]] + off
        print(f"    AR {r['AR']:4g}  ours {r['E_hole']*1e3:+6.0f} meV "
              f"{'BOUND' if r['E_hole'] > 0 else 'RESONANCE':<10} "
              f"theirs {y*1e3:+6.0f} meV {'BOUND' if y > 0 else 'RESONANCE'}", flush=True)

    print(f"\n  Hole level spacing (Kramers pairs). Yeap: \"the separation between the levels "
          f"should be\n  about 140 meV\" over this AR range.", flush=True)
    for r in rows:
        gaps = [(r['levels'][i] - r['levels'][i + 1]) * 1e3 for i in range(len(r['levels']) - 1)]
        print(f"    AR {r['AR']:4g}  " + '  '.join(f"{g:6.1f}" for g in gaps)
              + f"  meV   (Kramers splitting {r['kramers']*1e6:.2f} ueV)", flush=True)

    bad = [r for r in rows if r['n_above_bound']]
    if bad:
        print(f"\n  *** {len(bad)} point(s) have eigenvalues above the variational bound. Under "
              f"Burt-Foreman\n  that must not happen; something has gone wrong.", flush=True)
    else:
        print(f"\n  No eigenvalue above the variational bound at any point, as Burt-Foreman "
              f"requires.", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', type=float, nargs='*', default=None)
    ap.add_argument('--h', type=float, default=0.25)
    ap.add_argument('--crop', type=float, default=CROP_PAD)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--no-piezo', action='store_true',
                    help="drop the piezoelectric potential, which is what Yeap do")
    args = ap.parse_args()
    piezo = not args.no_piezo

    store = {}
    if os.path.exists(args.out):
        store = json.load(open(args.out))
        print(f"resuming: {len(store)} point(s) already in {args.out}", flush=True)

    want = args.ar if args.ar else list(ARS)
    if not args.report_only:
        print(f"InSb in InAs, base {BASE} nm, AR = b/h, T = {T_K} K, h = {args.h} nm, "
              f"strain pad {STRAIN_PAD} nm, k.p crop {CROP_PAD} nm, Burt-Foreman ordering",
              flush=True)
        for AR in want:
            key = (f"AR{AR:.2f}_h{args.h:.3f}"
                   + ('' if args.crop == CROP_PAD else f"_c{args.crop:.2f}")
                   + ('' if piezo else '_nopz'))
            if key in store:
                print(f"  AR {AR:g}: cached", flush=True)
                continue
            print(f"  AR {AR:g} (height {BASE/AR:.3f} nm)", flush=True)
            store[key] = solve_point(AR, args.h, crop=args.crop, piezo=piezo)
            json.dump(store, open(args.out, 'w'), indent=1)

    # Records written before `piezo` was recorded were all run with hs.build's default, i.e. ON.
    rows = [store[k] for k in sorted(store, key=lambda s: float(s[2:6]))
            if abs(store[k]['h'] - args.h) < 1e-9
            and abs(store[k]['crop_pad'] - args.crop) < 1e-9
            and store[k].get('piezo', True) == piezo]
    if rows:
        report(rows)
