"""Yeap et al. Fig. 6(a) in layout: single-particle levels vs aspect ratio, with the InSb and
InAs valence-band edges as reference lines.

SUPERSEDED -- use `scripts/plot_yeap_fig6.py`, which draws their actual figure at their actual
geometry, both panels. This one is kept only because the large-island curves it plots are a
different (and still valid) dataset.

Their figure is a 2.5 nm-base InSb/InAs ellipsoid at 80 K over AR 1-4. **This is not that figure.**
Our islands here are 20-40 nm across at 77 K over AR 4-16, because at the time the finite-
difference hole solver had a hard floor above their whole size range. That floor is GONE: the
interface runaway was diagnosed as an operator-ordering defect and fixed with Burt-Foreman
ordering in `kp_planewave`, so their 2.5 nm dots are now computable and `scripts/yeap_fig6_ar.py`
computes them. What this script reproduces is the STRUCTURE of their figure and the reference
lines in it, which are band edges and therefore size-independent -- see
`scripts/yeap_band_profile.py`, where the dot and matrix heavy-hole edges are checked against
their published values at AR 1, 2 and 4.

Plotted on THEIR zero (unstrained InSb valence edge = 0) so the axis can be read against the
paper. The package stores energies on the matrix zero; the two differ by VBO(matrix) = -0.59 eV
for InAs.

The electron
------------
Yeap's triangles are a solved E1 lying ~12 meV below E_c(InAs, far) and flat in aspect ratio.
**We do not solve it.** In these systems the only electron well is a thin tensile shell outside
the island, far too shallow for our box: measured, an 8.1 nm box put the lowest electron state
645 meV ABOVE E_c(far), which is pure box quantisation and not a bound state. So the electron is
drawn as the matrix conduction edge with a 12 meV band beneath it marking where a resolved level
would sit. Using E_c(far) for the electron is the same approximation the transition energies
already make, and it is worth about 12 meV on a 100-250 meV answer.

Run:  python scripts/plot_yeap_fig6a.py [--out _yeap_fig6a.png] [--base 30]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import materials as mt

SWEEPS = ('_transitions_77K.json', '_transitions_77K_arfix.json')
E1_BINDING = 0.012          # eV, Yeap's electron binding below E_c(far). NOT computed here.
CELLS_MIN, RATIO_MAX, LOC_MIN = 6.0, 0.30, 0.75


def load(base):
    rows = []
    for p in SWEEPS:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        assert not any('aspect' in r for r in d.values()), \
            f"{p}: pre-2026-08-05 h/d records; run scripts/migrate_aspect_to_AR.py"
        for r in d.values():
            if r.get('status') == 'ok' and r['base'] == base:
                rows.append(r)
    return rows


def graded(r):
    return (r['cells'] >= CELLS_MIN and r['artifact_ratio'] <= RATIO_MAX
            and r['loc'] >= LOC_MIN)


def series(rows, x, prefer_h=0.5):
    """One point per AR. Where both a fixed-h and a default-grid solve exist, take the fixed-h
    one -- the default rule refines the grid as AR grows and confounds shape with resolution."""
    by_ar = {}
    for r in rows:
        if r['x'] != x:
            continue
        cur = by_ar.get(r['AR'])
        if cur is None or (abs(r['h'] - prefer_h) < abs(cur['h'] - prefer_h)):
            by_ar[r['AR']] = r
    return [by_ar[a] for a in sorted(by_ar)]


def panel(ax, rows, x, name, colour, shift):
    s = series(rows, x)
    if not s:
        ax.text(0.5, 0.5, 'no data', ha='center', transform=ax.transAxes)
        return
    AR = [r['AR'] for r in s]
    Ec = s[0]['Ec_far'] + shift
    hole = [r['E_hole'] + shift for r in s]
    dot_vb = [r['v_top'] + shift for r in s]
    mat_vb = 0.0 + shift                      # package zero IS the unstrained matrix valence edge
    ok = [graded(r) for r in s]

    # Reference lines, in Yeap's styles.
    #
    # CAUTION on the dash-dot line. Yeap's "VB-InAs" is the matrix heavy-hole edge AT THE
    # INTERFACE, which is strained (the tensile matrix bends the heavy hole down) and runs
    # -0.79 -> -0.72 eV across AR 1-4. The line drawn here is the FAR-FIELD InAs valence edge,
    # flat at -0.59 eV by definition of their zero. They are ~200 meV apart and it would be easy
    # to read that as an error in the calculation rather than a difference in what is plotted.
    # The interface edge is not in the sweep records -- it needs one band-edge solve per AR
    # (`scripts/yeap_band_profile.py`, no eigensolver, ~1 min each).
    ax.plot(AR, dot_vb, '--', color='k', lw=1.6, label='InSb valence-band edge (in dot, strained)')
    ax.axhline(mat_vb, color='#777777', lw=1.4, ls='-.',
               label='InAs valence-band edge (far field)\n(Yeap plot the INTERFACE edge, '
                     '$\\approx$200 meV lower)')

    # Electron: the matrix conduction edge, with a band marking where a resolved E1 would sit.
    ax.axhline(Ec, color='#2b6cb0', lw=1.5, ls='-')
    ax.axhspan(Ec - E1_BINDING, Ec, color='#2b6cb0', alpha=0.25, lw=0)
    ax.plot([], [], '^', color='#2b6cb0', ms=7,
            label='electron: $E_c$(InAs, far)\n(band = Yeap\'s 12 meV binding, not solved here)')
    ax.plot(AR, [Ec] * len(AR), '^', color='#2b6cb0', ms=7, zorder=6)

    # Hole levels, Yeap's diamonds.
    ax.plot(AR, hole, '-', color=colour, alpha=0.45, lw=1.2)
    ax.plot([a for a, g in zip(AR, ok) if g], [e for e, g in zip(hole, ok) if g],
            'D', color=colour, ms=7, zorder=6, label='hole level')
    ax.plot([a for a, g in zip(AR, ok) if not g], [e for e, g in zip(hole, ok) if not g],
            'D', mfc='none', mec=colour, ms=7, mew=1.4, zorder=6)

    ax.fill_between(AR, hole, [Ec] * len(AR), color='#b0b0b0', alpha=0.28, lw=0)
    ax.set_xlabel('aspect ratio  AR = d/h   (flatter $\\rightarrow$)')
    ax.set_title(name, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    return s


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = float(sys.argv[sys.argv.index('--base') + 1]) if '--base' in sys.argv else 30.0
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else '_yeap_fig6a.png'

    rows = load(base)
    if not rows:
        sys.exit(f"no completed points at base {base:g} nm in {SWEEPS}")
    T = rows[0]['T']
    shift = mt.material('InAs')['VBO']         # repo zero -> papers' zero (unstrained InSb VB = 0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=True)
    s1 = panel(axes[0], rows, 1.00, 'InSb / InAs', '#c0392b', shift)
    s2 = panel(axes[1], rows, 0.50, 'In$_{0.5}$Ga$_{0.5}$Sb / InAs', '#8e44ad', shift)
    axes[0].set_ylabel('energy (eV), zero = unstrained InSb valence edge')
    fig.suptitle(
        f"Yeap et al. Fig. 6(a) layout, applied to our islands: d = {base:g} nm, T = {T:g} K.\n"
        "NOT a replication -- their dots are 2.5 nm across at AR 1-4, below this solver's floor. "
        "Grey shading = how far the hole level is from opening a gap.\n"
        "filled diamonds meet every reliability criterion; open ones are indicative only",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    print(f"base {base:g} nm, T = {T:g} K, zero = unstrained InSb VB "
          f"(shifted by VBO(InAs) = {shift:+.3f} eV)\n")
    for s, name in ((s1, 'InSb'), (s2, 'In0.5Ga0.5Sb')):
        if not s:
            continue
        print(f"{name}:")
        print(f"  {'AR':>6} {'h':>5} {'dot VB':>8} {'E_hole':>8} {'E_c':>8} "
              f"{'conf':>8} {'E_trans':>9}  grid")
        for r in s:
            print(f"  {r['AR']:6.2f} {r['h']:5.2f} {r['v_top']+shift:+8.4f} "
                  f"{r['E_hole']+shift:+8.4f} {r['Ec_far']+shift:+8.4f} "
                  f"{r['conf']:7.1f}m {r['E_trans']:+9.3f}  "
                  f"{'fixed-h' if abs(r['h']-0.5) < 1e-9 else 'default'}"
                  f"{'' if graded(r) else '  (indicative)'}")
        print()
    print(f"wrote {out}")
