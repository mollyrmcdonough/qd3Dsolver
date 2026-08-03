"""Is the residual InSb/InAs dot disagreement with Pryor & Pistol box contamination?

`insb_pryor_pistol_check.py` gets the analytic pseudomorphic WELL exactly right (0 meV on both
band edges) but leaves the DOT a few meV out, and `insb_band_edges_vs_size.py` shows that
few-meV number wobbling by ~7 meV between grid spacings. The well has no grid and no box, so
whatever is left is ours, not the parameter set's. This script asks which of our two numerical
knobs it is.

The two knobs are independent and were confounded in the earlier runs
--------------------------------------------------------------------
  RESOLUTION  cells across the island   -- fixes staircasing of the curved surface
  BOX         how much matrix surrounds it -- `elasticity_fd` is PERIODIC, so a tight box lets
                                             the island interact with its own images

Refining the spacing at fixed padding changes BOTH (the island grows in cells *and* fills more of
the box). The earlier size sweep showed exactly that pathology: with pad held at 6 nm, the mean
hydrostatic strain became monotonically more compressive as the island grew, and the conduction
edge drifted 28.6 meV, purely because lateral fill went 40% -> 73%.

So this script holds the island resolution FIXED and varies only the padding.

Why Pryor's grids suggest this is the answer
-------------------------------------------
Pryor specifies grids in SITES, not in a spacing -- which is the scale-invariant way, since the
elasticity equations have no length scale. C. Pryor, M.-E. Pistol and L. Samuelson, Phys. Rev. B
56, 10404 (1997) (arXiv:cond-mat/9705291) states a "periodic box of 130 x 130 x 120 sites with
the island contained within 65 x 65 x 20 region", i.e.

    lateral fill 50%,  vertical fill 17%

Our runs so far have been at 58% and 25% -- a TIGHTER box than his at comparable island
resolution, and tight in the direction that matters most for a flat island. If the residual is
box contamination, the gap should move toward their Table III value as the padding grows and
should flatten once the fill fraction drops past his.

NOTE ON `h`. Pryor's `h` is the DOT HEIGHT (his Fig. 1a, h/d = 1/4). The `spacing` column below
is the grid spacing, a different quantity entirely; it is reported only so the runs can be
reproduced, and the scale-invariant columns are the ones to read.

NOTE ON THE FIRST RUN OF THIS SCRIPT. Its `spacing = 0.3, pad = 14` row was WRONG: the
preconditioner in `elasticity_fd` used a magnitude-based singularity test that filtered
long-wavelength modes on grids past ~150 sites on a side, so that row reported a plausible gap
while making <Tr eps> more compressive in a LARGER box -- impossible. See
`elasticity_preconditioner_check.py` and the README section "A preconditioner cutoff that made the
strain error grow with resolution". The `pad = 18` row was NOT affected (it dropped only k = 0),
so the conclusion drawn from it survived; it was correct by a margin of 1.3x rather than by
design. With the preconditioner fixed those grids converge in 8 iterations instead of grinding to
the iteration cap, which is ~40x faster and is why the padding can now be pushed further.

Run:  python scripts/insb_box_convergence.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import qdsolver_core as qd
import ingasb_dot as ig

PP_DOT_GAP = 0.800 - 0.127        # Pryor & Pistol Table III, InAs substrate / InSb dot
PP_OFFSET = -0.590

RADIUS, ASPECT = 10.0, 0.25       # d = 20 nm, height = 5 nm
PRYOR_FILL = (0.50, 0.17)         # lateral, vertical, from the 1997 paper


def one(h, pad):
    s = ig.spherical_lens(RADIUS, 2 * RADIUS * ASPECT)
    env = ig.build(s, 1.0, h=h, pad=pad, use_piezo=False, verbose=False)
    e = ig.valence_edge(env)
    m = env['mask']
    g = env['strain'].at(m)
    nx, ny, nz = m.shape
    d_cells = int(round(2 * RADIUS / h))
    h_cells = int(round(2 * RADIUS * ASPECT / h))
    out = dict(h=h, pad=pad, shape=m.shape, sites=m.size,
               d_cells=d_cells, h_cells=h_cells,
               lat_fill=d_cells / nx, vert_fill=h_cells / nz,
               iters=env['strain'].cg_iterations, dropped=env['strain'].cg_modes_dropped,
               cb=float(e['cb'][m].mean()), vb=float(e['v1'][m].mean()),
               tr=float(g['exx'] + g['eyy'] + g['ezz']))
    out['gap'] = out['cb'] - out['vb']
    del env, e
    return out


def sweep(h, pads, label):
    print(f"\n{'='*112}\n{label}   spacing = {h} nm, island = "
          f"{int(round(2*RADIUS/h))}x{int(round(2*RADIUS/h))}x{int(round(2*RADIUS*ASPECT/h))} "
          f"cells (FIXED)\n{'='*112}")
    # `CG` and `drop` are printed rather than merely checked. `drop` must be 1 -- only k = 0 is
    # legitimately singular -- and a row where it is not had its long-wavelength modes filtered
    # out of the preconditioner, which is how two rows of the first version of this table came to
    # be wrong while looking fine. build() now raises on that, so these columns are a receipt.
    print(f"{'pad':>5}{'box sites':>16}{'Msites':>8}{'lat fill':>10}{'vert fill':>11}"
          f"{'CG':>4}{'drop':>6}{'<Tr eps>':>11}{'CB':>9}{'VB':>9}{'gap':>9}{'vs P&P':>9}{'s':>6}")
    rows = []
    for pad in pads:
        t0 = time.time()
        r = one(h, pad)
        rows.append(r)
        print(f"{pad:>5.1f}{'x'.join(str(n) for n in r['shape']):>16}"
              f"{r['sites']/1e6:>8.2f}{r['lat_fill']*100:>9.0f}%{r['vert_fill']*100:>10.0f}%"
              f"{r['iters']:>4}{r['dropped']:>6}"
              f"{r['tr']:>+11.5f}{r['cb']:>9.4f}{r['vb']:>9.4f}{r['gap']:>9.4f}"
              f"{(r['gap']-PP_DOT_GAP)*1e3:>+8.0f}m{time.time()-t0:>6.0f}", flush=True)
    return rows


if __name__ == '__main__':
    print(__doc__.split('Run:')[0])
    print(f"Pryor & Pistol Table III dot gap: {PP_DOT_GAP:.3f} eV")
    print(f"Pryor 1997 fill fractions: lateral {PRYOR_FILL[0]*100:.0f}%, "
          f"vertical {PRYOR_FILL[1]*100:.0f}%")

    # Coarser spacing so the padding can be pushed far without the site count exploding. The
    # padding runs well past Pryor's fill fractions, because the first version of this sweep
    # stopped at pad = 18 and was still moving ~6 meV per step there -- it never showed the
    # flattening that the whole argument depends on.
    coarse = sweep(0.4, (5.0, 7.5, 10.0, 14.0, 18.0, 24.0, 30.0),
                   'BOX CONVERGENCE at moderate island resolution')

    # Repeat the most informative paddings at higher island resolution, to show the conclusion is
    # about the BOX and not about staircasing.
    fine = sweep(0.3, (7.5, 14.0, 18.0), 'SAME TEST at higher island resolution')

    print(f"\n{'='*112}\nREADING IT")
    g = np.array([r['gap'] for r in coarse])
    tr = np.array([r['tr'] for r in coarse])
    print(f"  gap moves {(g[-1]-g[0])*1e3:+.1f} meV from pad {coarse[0]['pad']:.0f} -> "
          f"{coarse[-1]['pad']:.0f} nm; last two points differ by {(g[-1]-g[-2])*1e3:+.1f} meV")
    print(f"  <Tr eps> moves {(tr[-1]-tr[0])*1e3:+.2f} m-strain over the same range")
    print(f"  converged (largest box) gap {g[-1]:.4f} eV vs Pryor & Pistol {PP_DOT_GAP:.3f} "
          f"-> {(g[-1]-PP_DOT_GAP)*1e3:+.0f} meV")
    print("\n  If the gap flattens as the padding grows and lands near their value, the residual")
    print("  was periodic-image contamination in OUR box, and the parameter set is clean -- which")
    print("  the analytic well already says at 0 meV. If it flattens somewhere else, the dot")
    print("  disagreement is real and lives in the shape or the 3D strain field.")
    print("\n  A tight box makes the island stiffer (its images resist relaxation), so it should")
    print("  overstate |Tr eps| and therefore overstate the strain-opened gap. Watch whether the")
    print("  two move together in that direction.")
