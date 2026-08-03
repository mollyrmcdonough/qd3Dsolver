"""Which material systems confine a carrier, and how big the island has to be.

One converged strain solve per (dot, matrix) pair, turned into the two numbers that decide
whether a nanostructure is useful: does a single-particle well exist for each carrier, and if so
what is the smallest island of that shape that can hold it.

Why one solve per system is enough
----------------------------------
Continuum elasticity has no length scale, so at fixed SHAPE the strain field and every band edge
are independent of the island's size. That is verified to 0.0 meV over a 4x range in
`insb_band_edges_vs_size.py`, and exactly rather than approximately -- scaling the grid and box
with the island makes every size the same discrete problem, so the columns agree bit for bit.

The well depth V0 is therefore size-independent and the well radius R is exactly proportional to
the island's linear size L. The binding condition for a finite spherical well,

    V0 R^2 > pi^2 hbar^2 / (8 m*)

is then a condition on L alone, and one solve gives the critical L by extrapolation. Without
scale invariance this table would be a size sweep per material pair instead of a single point.

What the critical size does and does not mean
---------------------------------------------
It is the size below which NO bound single-particle state exists, from a necessary condition
applied to a sphere of the same volume as the real well. Failing is decisive -- a real well binds
no better than that sphere, and a thin shell binds far worse. Passing is only suggestive, so a
system that clears the threshold deserves a real solve (`heterostructure.electron_states` or
`eight_band_states`) before anything is claimed about level spacings.

It also ignores Coulomb attraction entirely. In a broken-gap system the electron is bound to the
hole as an exciton whether or not any single-particle pocket holds it, so "does not confine the
electron" here means "no single-particle well", not "no bound electron".

Accuracy, measured rather than assumed
--------------------------------------
Two numerical knobs, both checked on InSb/InAs (see the README):

  BOX. `elasticity_fd` is periodic, so a tight box lets the island feel its own images. The
       electron critical size runs 45.3 -> 41.9 -> 39.2 -> 38.1 -> 37.6 nm over paddings
       7.5 -> 30 nm at fixed resolution: converging, ~1 nm left at pad = 3R.

  RESOLUTION. Staircasing of the curved cap. At pad = 2R the critical size runs
       31.8 / 35.3 / 37.4 / 38.8 nm for 8 / 12 / 16 / 20 cells across the island radius --
       still moving 1.4 nm at the finest, so it is NOT resolution converged and roughly linear
       in h, as staircasing should be.

So treat the electron critical size as good to about +/-3 nm, i.e. two significant figures. The
VERDICT is far more robust than the number: for InSb/InAs the pocket falls short by a factor of
3.5 at every box and every resolution tried. The hole numbers are stable to 0.1 nm because that
well is deep and compact.

Run:  python scripts/confinement_map.py            # default resolution, a few minutes
      python scripts/confinement_map.py --fine     # 20 cells/R, slower
      python scripts/confinement_map.py --quick    # 8 cells/R, seconds, indicative only
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import numpy as np

import materials as mt
import heterostructure as hs

RADIUS, ASPECT = 10.0, 0.25        # spherical cap, h/d = 1/4, Pryor & Pistol's dot geometry

#: (dot spec, matrix). The four binaries and both alloys this project carries, in the
#: combinations that are lattice-plausible. Ordered matrix-by-matrix so the table reads as a
#: comparison within each host.
CASES = [
    (('InAsSb', 0.25), 'InAs'),
    (('InAsSb', 0.50), 'InAs'),
    (('InAsSb', 0.75), 'InAs'),
    ('InSb', 'InAs'),
    ('GaSb', 'InAs'),
    (('InGaSb', 0.50), 'InAs'),
    ('InAs', 'GaAs'),
    ('GaSb', 'GaAs'),
    ('InAs', 'GaSb'),
]

#: Misfit past which linear elasticity is being pushed hard enough to say so. Pryor's own
#: InAs/GaAs benchmark sits at 6.7%, so that is the calibration point rather than a limit.
MISFIT_WARN = 0.075


def one(dot, matrix, cells, pad_frac=2.0):
    """Solve one (dot, matrix) pair and reduce it to a row."""
    h = RADIUS / cells
    shape = hs.spherical_lens(RADIUS, 2 * RADIUS * ASPECT)
    env = hs.build(shape, dot, matrix=matrix, h=h, pad=pad_frac * RADIUS,
                   use_piezo=False, verbose=False)
    edges = hs.valence_edge(env)
    row = dict(dot=env['dot']['name'], matrix=env['matrix']['name'],
               misfit=env['eps_star'], align=env['alignment'],
               sites=env['mask'].size, h=h)
    for carrier in ('electron', 'hole'):
        c = hs.critical_size(env, carrier, edges=edges)
        best = max((r for r in hs.well_metrics(env, carrier, edges=edges)
                    if r['volume'] > 0 and not r['box_limited']),
                   key=lambda r: r['VR2'], default=None)
        core = hs.well_core(env, carrier, edges=edges)
        row[carrier] = dict(critical=c['critical'], factor=c['factor'],
                            VR2=0.0 if best is None else best['VR2'],
                            threshold=float('nan') if best is None else best['threshold'],
                            # The WELL CORE, not the selected contour: the contour that maximises
                            # V0R^2 is the shallowest surviving one and is halo-contaminated, and
                            # reading location off it puts the InAs/GaAs hole in the matrix.
                            core_in_dot=core['frac_in_dot'], depth_max=core['depth_max'],
                            box_limited=c.get('box_limited', False))
    return row


def table(rows, cells):
    L = 2 * RADIUS
    print(f"\n{'='*118}")
    print(f"spherical cap, d = {L:g} nm, h/d = {ASPECT}, {cells} cells across the radius, "
          f"pad = 2R.  Island is {L:g} nm across.")
    print(f"{'='*118}")
    print(f"{'dot':>15}{'in':>7}{'misfit':>9}{'alignment':>19}"
          f"{'e V0R2/thr':>13}{'e core':>9}{'e crit':>9}"
          f"{'h V0R2/thr':>13}{'h core':>9}{'h crit':>9}")
    for r in rows:
        cells_ = []
        for carrier in ('electron', 'hole'):
            c = r[carrier]
            ratio = '--' if not np.isfinite(c['threshold']) else f"{c['VR2']/c['threshold']:.2f}"
            crit = ('--' if not np.isfinite(c['critical'])
                    else 'binds' if c['factor'] <= 1.0 else f"{c['critical']:.0f} nm")
            cells_.append((ratio, crit, c['core_in_dot']))
        (er, ec, ef_), (hr, hc, hf_) = cells_
        warn = ' !' if abs(r['misfit']) > MISFIT_WARN else '  '
        print(f"{r['dot']:>15}{r['matrix']:>7}{r['misfit']*100:>8.2f}%{warn[1]}"
              f"{r['align']['type']:>18}"
              f"{er:>13}{ef_*100:>8.0f}%{ec:>9}"
              f"{hr:>13}{hf_*100:>8.0f}%{hc:>9}")
    if any(abs(r['misfit']) > MISFIT_WARN for r in rows):
        print(f"\n  ! misfit above {MISFIT_WARN*100:.1f}%: linear elasticity is being pushed "
              f"harder than Pryor's own 6.7% benchmark.")


def reading_it(rows):
    print(f"\n{'='*118}\nREADING IT")
    print("  'V0R2/thr' is the binding criterion as a RATIO: >= 1 means a single-particle well")
    print("  could hold that carrier at this size. 'crit' is the smallest island of this shape")
    print("  that could, or 'binds' if the current 20 nm island already does.")
    print("  'core' is the fraction of the well's DEEPEST region inside the island: ~100% is a")
    print("  dot-confined carrier, ~0% means it is EXPELLED and what holds it (if anything) is")
    print("  a strain pocket in the matrix around the island.")

    binds_e = [r for r in rows if r['electron']['factor'] <= 1.0]
    binds_h = [r for r in rows if r['hole']['factor'] <= 1.0]
    both = [r for r in rows if r in binds_e and r in binds_h]
    print(f"\n  {len(binds_h)}/{len(rows)} systems confine a hole at 20 nm; "
          f"{len(binds_e)}/{len(rows)} confine an electron; {len(both)} confine both.")

    # Binding both carriers is NOT the same as being a type-I dot. What makes a dot type I is
    # that both wells are in the SAME place -- the island -- which is what gives strong
    # electron-hole overlap and therefore bright recombination. A staggered system can bind both
    # and still hold them on opposite sides of the interface, with the overlap that implies.
    if both:
        print("\n  Of those, where the two carriers actually sit (>50% of each well CORE inside")
        print("  the island counts as dot-confined). Both-in-the-dot is the type-I case, with")
        print("  strong overlap; SPLIT means spatially indirect, and a weak transition.")
        for r in both:
            e_in = r['electron']['core_in_dot'] > 0.5
            h_in = r['hole']['core_in_dot'] > 0.5
            where = ('both in the dot -- type I, strong overlap' if e_in and h_in else
                     'both in the matrix' if not e_in and not h_in else
                     'SPLIT: ' + ('electron in dot, hole in matrix' if e_in else
                                  'hole in dot, electron in matrix'))
            print(f"    {r['dot']:>15} in {r['matrix']:<6} {r['align']['type']:>18}  -> {where}")
    print("\n  Failing the criterion is decisive; passing it is only suggestive -- confirm with")
    print("  a real solve. Coulomb binding is not included, so a system that fails for the")
    print("  electron can still hold a bound exciton, which is the broken-gap case exactly.")
    print("  The critical size is good to ~2 significant figures; see the docstring.")


if __name__ == '__main__':
    cells = 12
    if '--fine' in sys.argv:
        cells = 20
    elif '--quick' in sys.argv:
        cells = 8

    print(__doc__.split('Run:')[0])
    print(f"materials at T = {mt.TEMPERATURE:g} K; {cells} cells across the island radius")

    rows = []
    for dot, matrix in CASES:
        t0 = time.time()
        try:
            r = one(dot, matrix, cells)
        except Exception as exc:                       # a mask or convergence failure, not a bug
            name = dot if isinstance(dot, str) else f"{dot[0]}({dot[1]})"
            print(f"  SKIPPED {name} in {matrix}: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print(f"  {r['dot']:>15} in {r['matrix']:<6} {r['sites']/1e6:>5.2f} Msites  "
              f"{time.time()-t0:>5.0f}s", flush=True)

    table(rows, cells)
    reading_it(rows)

    try:
        import matplotlib
        matplotlib.use('Agg')
        fig = mt.plot_alignment([d for d, _ in CASES if _ == 'InAs'], matrix='InAs')
        fig.savefig('_confinement_alignment.png', dpi=140)
        print("\n  band lineup written to _confinement_alignment.png")
    except Exception as exc:
        print(f"\n  (no figure: {type(exc).__name__}: {exc})")
