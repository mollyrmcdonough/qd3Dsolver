"""Why the two largest strain solves were wrong, and that the fix is a fix.

The box-convergence study (`insb_box_convergence.py`) produced one catastrophic point --
conduction edge at -12.25 eV, negative gap -- and one quietly wrong point, where refining the
grid at FIXED box made the mean hydrostatic strain 4e-3 MORE compressive. A bigger or
better-resolved box must relax more, never less, so the second was as broken as the first; it
just looked like a number.

The first suspicion was the CG iteration cap. That was wrong: the solves converge in 8 iterations
to 9e-11. The cause is one line of the PRECONDITIONER.

    good = np.abs(det) > 1e-10 * np.abs(det).max()          # the old test

`_Operator` preconditions with the exact inverse of the homogeneous symbol S(k), and skips
wavevectors where S is singular. Only k = 0 is: rigid translation costs no energy. But S vanishes
as |k|^2 there, so det S ~ |k|^6, and on a grid whose longest side is N the smallest nonzero
wavevector 2 pi / N gives det / det_max ~ (2/N)^6. That falls through 1e-10 somewhere near
N ~ 100 and keeps falling. So the test discards a shell of long-wavelength modes that are
perfectly well conditioned and merely small -- and discards a wider shell the finer the grid.

Those are precisely the modes that carry an inclusion's long-range relaxation. Zeroing them makes
the preconditioner a low-pass filter, and CG cannot recover what the preconditioner never admits
to the Krylov space. The residual still reports converged, because it is measured in the filtered
space. Hence: an error that GROWS with resolution while every diagnostic looks clean.

The replacement (`_nonsingular`) tests det against (tr/3)^3, which is scale free -- both scale as
the cube of the block's own magnitude -- so it asks about conditioning instead of magnitude, and
excludes k = 0 separately on magnitude, which is the property k = 0 actually lacks.

Three things are checked here:

  PART 1  how many modes each criterion drops, at the grid sizes actually used. Cheap: the
          symbol needs no solve. This is the diagnosis.
  PART 2  that the fix does not move the answer where the old code was safe. If it did, the fix
          would be a change of physics rather than a repair.
  PART 3  the two broken points, re-run.

Run:  python scripts/elasticity_preconditioner_check.py            # all three parts
      python scripts/elasticity_preconditioner_check.py --diagnose # PART 1 only, seconds
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import qdsolver_core as qd
import elasticity_fd as ef
import materials_sb as ms
import ingasb_dot as ig

RADIUS, ASPECT = 10.0, 0.25

# (h, pad) -> the shape build() produces, and the gap the earlier run reported for it.
# The gridscale rows are pad = 7.5, which is what reproduces the grids in gridscale.log
# (35/0.5 = 70 by 20/0.5 = 40; 35/0.22 = 158 by 20/0.22 = 91).
CASES = [
    # h    pad    where it came from                        earlier gap
    (0.50,  7.5, 'gridscale coarse',                        0.6750),
    (0.40,  7.5, 'box_conv',                                None),
    (0.40, 14.0, 'box_conv',                                None),
    (0.40, 18.0, 'box_conv  <- the "converged" point',      None),
    (0.30,  7.5, 'box_conv fine / gridscale mid',           0.6676),
    (0.30, 14.0, 'box_conv fine  <- trend broke here',      0.6785),
    (0.22,  7.5, 'gridscale finest <- CB came out -12 eV', -19.7240),
]


def grid_shape(h, pad):
    """The grid build() will use -- laid out the same way, but with no solve behind it."""
    s = ig.spherical_lens(RADIUS, 2 * RADIUS * ASPECT)
    cx, cy, cz, *_ = qd.island_grid(s['extent'], h, pad)
    return (len(cx), len(cy), len(cz))


def dropped_counts(shape):
    """(old drops, new drops, det ratio of the smallest LEGITIMATE mode).

    The ratio is taken over the wavevectors the scale-free test accepts, i.e. everything but
    k = 0. Including k = 0 makes the column meaningless: S(0) is zero to roundoff, so its det is
    ~1e-50 on every grid and the ratio reports floating-point noise rather than the smallest mode
    the old test had to judge. That smallest legitimate det, against the maximum, is the number
    the old 1e-10 cutoff was unwittingly racing.
    """
    C_dot, C_mat = ms.elastic(ms.ingasb(1.0)), ms.elastic(ms.SB_MATERIALS['InAs'])
    f = 0.5                                    # only the scale matters for this test
    K_d, _ = ef._element_arrays(*C_dot, 1.0)
    K_m, _ = ef._element_arrays(*C_mat, 1.0)
    S = ef._symbol(K_m + f * (K_d - K_m), shape)
    M = np.moveaxis(S, [0, 1], [-2, -1])

    det = np.abs(np.linalg.det(M))
    old_good = det > 1e-10 * det.max()
    new_good = ef._nonsingular(M)

    legit = det[new_good]
    ratio = float(legit.min() / det.max()) if legit.size else 0.0
    out = (int(old_good.size - old_good.sum()), int(new_good.size - new_good.sum()), ratio)
    del S, M, det, old_good, new_good
    return out


def solve(h, pad):
    t0 = time.time()
    s = ig.spherical_lens(RADIUS, 2 * RADIUS * ASPECT)
    env = ig.build(s, 1.0, h=h, pad=pad, use_piezo=False, verbose=False)
    e, m = ig.valence_edge(env), env['mask']
    g = env['strain'].at(m)
    out = dict(shape=m.shape, sites=m.size, secs=time.time() - t0,
               iters=env['strain'].cg_iterations, resid=env['strain'].cg_residual,
               dropped=env['strain'].cg_modes_dropped,
               tr=float(g['exx'] + g['eyy'] + g['ezz']),
               cb=float(e['cb'][m].mean()), vb=float(e['v1'][m].mean()))
    out['gap'] = out['cb'] - out['vb']
    del env, e
    return out


if __name__ == '__main__':
    print(__doc__.split('Run:')[0])

    print('=' * 100)
    print('PART 1  modes the preconditioner refuses to invert.  Only k = 0 is legitimate, so the')
    print('        correct count is 1.  No solve is done here -- this is the symbol alone.')
    print('=' * 100)
    print(f"{'h':>6}{'pad':>6}{'grid':>16}{'Msites':>8}{'detmin/detmax':>15}"
          f"{'OLD drops':>11}{'NEW drops':>11}   provenance")
    shapes, drops = {}, {}
    for h, pad, note, _ in CASES:
        shapes[(h, pad)] = shape = grid_shape(h, pad)
        drops[(h, pad)] = old, new, ratio = dropped_counts(shape)
        flag = '' if old <= 1 else '   <-- FILTERED'
        print(f"{h:>6.2f}{pad:>6.1f}{'x'.join(map(str, shape)):>16}"
              f"{np.prod(shape)/1e6:>8.2f}{ratio:>15.1e}{old:>11,}{new:>11,}   {note}{flag}")

    if '--diagnose' in sys.argv:
        print('\n  --diagnose: stopping before the solves. Any row flagged FILTERED produced a')
        print('  wrong strain field, and every band edge derived from it is void.')
        sys.exit(0)

    print()
    print('PART 2  does the fix change the answer where the old test dropped only k = 0?')
    print('        A repair must be invisible there.')
    print('=' * 100)
    safe = [(h, pad, prev) for h, pad, _, prev in CASES
            if drops[(h, pad)][0] <= 1 and prev is not None]
    if not safe:
        print('  No case in the list was safe under the old test -- see PART 1. Falling back to a')
        print('  small grid the old code certainly handled.')
        safe = [(0.6, 4.0, None)]
    print(f"{'h':>6}{'pad':>6}{'grid':>16}{'CG':>5}{'resid':>10}{'drop':>6}"
          f"{'<Tr eps>':>11}{'gap now':>10}{'gap before':>12}{'delta':>9}")
    for h, pad, prev in safe:
        r = solve(h, pad)
        d = '' if prev is None else f"{(r['gap']-prev)*1e3:>+8.2f}m"
        p = '' if prev is None else f"{prev:>12.4f}"
        print(f"{h:>6.2f}{pad:>6.1f}{'x'.join(map(str, r['shape'])):>16}{r['iters']:>5}"
              f"{r['resid']:>10.1e}{r['dropped']:>6}{r['tr']:>+11.5f}{r['gap']:>10.4f}{p}{d}")

    print()
    print('PART 3  the points that were wrong, re-run.')
    print('=' * 100)
    print(f"{'h':>6}{'pad':>6}{'grid':>16}{'CG':>5}{'resid':>10}{'drop':>6}"
          f"{'<Tr eps>':>11}{'gap now':>10}{'gap before':>12}{'delta':>9}{'s':>7}")
    for h, pad, note, prev in CASES:
        if drops[(h, pad)][0] <= 1:
            continue
        try:
            r = solve(h, pad)
        except RuntimeError as exc:
            print(f"{h:>6.2f}{pad:>6.1f}   still refuses: {exc}")
            continue
        d = '' if prev is None else f"{(r['gap']-prev)*1e3:>+8.0f}m"
        p = '' if prev is None else f"{prev:>12.4f}"
        print(f"{h:>6.2f}{pad:>6.1f}{'x'.join(map(str, r['shape'])):>16}{r['iters']:>5}"
              f"{r['resid']:>10.1e}{r['dropped']:>6}{r['tr']:>+11.5f}{r['gap']:>10.4f}{p}{d}"
              f"{r['secs']:>7.0f}")

    print()
    print('  READING PART 3.  <Tr eps> must become monotonically LESS compressive as the box')
    print('  grows -- a larger box lets the island relax further, and its periodic images push')
    print('  back less. If the sign of that trend is now consistent across every row, the box')
    print('  study can be redone and the converged dot gap restated. The -27 meV figure quoted')
    print('  before rested on the pad = 18 row, which PART 1 shows was filtered.')
