"""Yeap et al. Fig. 6, both panels, at their geometry -- ours against theirs.

    PYTHONIOENCODING=utf-8 python -u scripts/plot_yeap_fig6.py [--out _yeap_fig6.png]

Supersedes `plot_yeap_fig6a.py`, which drew the STRUCTURE of their figure using 20-40 nm islands
because the finite-difference hole solver had a hard floor above their whole size range. That
floor is gone: the symmetrized operator's interface runaway was diagnosed as an ordering defect
and fixed with Burt-Foreman ordering in `kp_planewave`, so their actual 2.5 nm dots are now
computable. Data comes from `scripts/yeap_fig6_ar.py`.

Plotted on THEIR zero, the unstrained InSb valence edge, so the axes can be read straight against
the paper. This package stores energies on the unstrained matrix (InAs) valence edge; the two
differ by the VBO, 0.590 eV.

The electron: solved, and NOT bound
------------------------------------
Yeap's triangles are a resolved E1 sitting flat at -0.20 eV, i.e. 17 meV below Ec(InAs, far) -- a
state in the tensile shell OUTSIDE the island, not a dot state. `scripts/electron_binding.py`
solves for it directly and finds **no bound state at all** on their 2.5 nm dot: over boxes L =
20 -> 60 nm the level tracks the empty-box quantum 3*G0*pi^2/(m L^2) down to 0.03-0.06 meV at
every step, pure 1/L^2 with no exponential settling, and the well contributes ~0.1 meV. The reason
is the 3D shallow-well threshold -- the shell is 150 meV deep at its deepest but only ~1 nm thick,
so V0*R^2 ~ 0.15 eV.nm^2 against the ~3.6 eV.nm^2 a bound state needs at m = 0.026.

So the SOLID curve in panel (b), electron at Ec(InAs, far), is the physical one, and the dashed
one is kept only to show what folding in their 17 meV would do. On this reading their 17 meV is a
property of their finite FEM domain -- the same conclusion already reached for their flat-aspect-
ratio hole levels, and consistent: both are states that are not actually bound.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import materials as mt
from yeap_fig6_ar import YEAP_ETRANS, YEAP_E1, YEAP_VB_INSB, vbo_offset, OUT


def load(path, h=None, crop=None, piezo=True):
    """One row per AR: the FINEST grid available for that aspect ratio.

    The sweep is deliberately not on a single spacing. AR 1..3 are converged at h = 0.25, but AR
    3.5 and 4 need h = 0.15 to put four cells across the height -- below that, neighbouring
    aspect ratios collapse onto the same discrete mask and return the same energy. Filtering on
    one h therefore drops the flat end entirely, which is what it used to do.

    Picking the finest h per AR is the best available estimate at every point; the residual grid
    dependence is a few meV (AR 3: 48.2 meV at h = 0.25, 44.0 at h = 0.175) against the 50-110 meV
    the figure is about. Pass `h` to force a single spacing for a controlled series instead.

    The store also holds convergence checks at other (h, crop), so filter -- plotting those
    unfiltered draws two points at one AR, which showed up only as a doubled annotation label
    rather than as a visibly wrong curve.
    """
    if not os.path.exists(path):
        raise SystemExit(f"no data at {path}; run scripts/yeap_fig6_ar.py first")
    store = json.load(open(path))
    rows = [r for r in store.values()
            if (h is None or abs(r['h'] - h) < 1e-9)
            and (crop is None or abs(r['crop_pad'] - crop) < 1e-9)
            and r.get('piezo', True) == piezo]
    best = {}
    for r in rows:                       # finest h wins; ties broken by the larger k.p box
        cur = best.get(r['AR'])
        if cur is None or (r['h'], -r['crop_pad']) < (cur['h'], -cur['crop_pad']):
            best[r['AR']] = r
    if not best:
        raise SystemExit(f"no rows matching h={h}, crop={crop}, piezo={piezo}")
    return sorted(best.values(), key=lambda r: r['AR'])


def main(path, out, h, crop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mt.set_temperature(80.0)
    off = vbo_offset()
    rows = load(path, h, crop)
    ar = np.array([r['AR'] for r in rows])
    # -off puts OUR energies on THEIR zero.
    hole = np.array([r['E_hole'] for r in rows]) - off
    vtop = np.array([r['v_top_mean'] for r in rows]) - off
    ec = np.array([r['Ec_far'] for r in rows]) - off
    etr = np.array([r['Ec_far'] - r['E_hole'] for r in rows])

    y_ar = np.array(sorted(YEAP_ETRANS))
    y_tr = np.array([YEAP_ETRANS[a] for a in y_ar])
    y_hh = YEAP_E1 - y_tr
    y_vt = np.array([YEAP_VB_INSB[a] for a in y_ar])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 8.0), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1.0]))

    # The unstrained InAs valence edge -- our energy zero, and the edge of the matrix valence
    # CONTINUUM. A hole level below this line is not a bound state at all but a resonance, whose
    # energy in a finite domain is a property of the domain. Their HH1 crosses it near AR 2.75,
    # which is exactly where the two curves in panel (b) separate; ours never does. This line is
    # the point of the figure, so draw it even though the paper does not.
    ax.axhline(-off, color='#2a6a2a', lw=1.1, ls=':',
               label='InAs VB (far); below = valence continuum')
    ax.plot(y_ar, y_vt, '--', color='0.35', lw=1.2, label='VB-InSb (Yeap)')
    ax.plot(ar, vtop, '-', color='#b03030', lw=1.6, marker='s', ms=4,
            label='dot HH edge (ours)')
    ax.plot(y_ar, y_hh, 'D', color='0.35', ms=6, mfc='none', mew=1.3,
            label='HH1 (Yeap Fig. 6b)')
    ax.plot(ar, hole, '-D', color='#b03030', lw=1.6, ms=5, label='HH1 (ours)')
    ax.plot(y_ar, np.full_like(y_ar, YEAP_E1), '^', color='0.35', ms=6, mfc='none', mew=1.3,
            label='E1 (Yeap)')
    ax.plot(ar, ec, '-', color='k', lw=1.2, label='$E_c$(InAs, far) (ours)')
    # Their E1 sits 17 meV below our Ec(far). We solve that state and find it unbound (box series
    # in scripts/electron_binding.py: pure 1/L^2, no settling), so this band marks a difference
    # to be explained, not a binding we are missing.
    ax.fill_between(ar, ec - 0.017, ec, color='k', alpha=0.12, lw=0,
                    label='17 meV: their E1 (we find no bound state)')
    hs_used = sorted({r['h'] for r in rows})
    hlab = (f"h = {hs_used[0]:g} nm" if len(hs_used) == 1
            else f"h = {hs_used[0]:g}–{hs_used[-1]:g} nm, finest per AR")
    ax.set_ylabel('Energy (eV), zero = unstrained InSb VB')
    ax.set_title(f"Yeap et al. PRB 79, 075305 Fig. 6 — InSb/InAs ellipsoid, "
                 f"base {rows[0]['base']:g} nm, 80 K\n"
                 f"plane-wave six-band k·p, Burt–Foreman ordering, {hlab}",
                 fontsize=8.5)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc='center left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

    # Shade the continuum, after plotting so the limits are settled. A level in here is not a
    # bound state: it is degenerate with the matrix valence band at infinity, so its energy is a
    # property of whatever box the solver used.
    ylo = ax.get_ylim()[0]
    ax.fill_between([y_ar[0], y_ar[-1]], ylo, -off, color='#2a6a2a', alpha=0.07, lw=0, zorder=0)
    ax.set_ylim(bottom=ylo)
    # y_hh DECREASES with AR, so reverse both arrays to give np.interp the increasing xp it
    # requires. Negating instead (which is what this line did first) reverses the sense and
    # silently returns the wrong end of the curve -- it read AR 4.00 for a crossing at 2.75.
    cross = float(np.interp(-off, y_hh[::-1], y_ar[::-1]))
    ax.annotate(f"their HH1 enters the\ncontinuum near AR {cross:.2f}",
                xy=(cross, -off), xytext=(cross + 0.05, -off - 0.085), fontsize=6.5,
                color='#2a6a2a', ha='left',
                arrowprops=dict(arrowstyle='->', color='#2a6a2a', lw=0.8))

    bx.plot(y_ar, y_tr, '-D', color='0.35', lw=1.4, ms=6, mfc='none', mew=1.3,
            label='Yeap, 80 K')
    # Solid is the physical curve: the electron is NOT bound (scripts/electron_binding.py), so
    # E_c(InAs, far) is the reference and there is no binding to subtract. The dashed curve keeps
    # their 17 meV visible only to show how much of the comparison rides on that one number.
    bx.plot(ar, etr, '-o', color='#1f60a8', lw=1.8, ms=5,
            label='ours, electron at $E_c$(far) — no bound state')
    bx.plot(ar, etr - 0.017, '--o', color='#1f60a8', lw=1.0, ms=3.5, alpha=0.45,
            label='if their 17 meV electron binding were real')
    for a, o, t in zip(ar, etr, np.interp(ar, y_ar, y_tr)):
        bx.annotate(f"{(o-t)*1e3:+.0f}", (a, (o + t) / 2), fontsize=6.5, ha='center',
                    color='0.4')
    bx.set_xlabel('Aspect ratio  AR = b/h   (larger = flatter)')
    bx.set_ylabel('Transition energy (eV)')
    bx.legend(fontsize=7.5, frameon=False, loc='upper left')
    bx.spines[['top', 'right']].set_visible(False)
    bx.tick_params(labelsize=8)
    bx.text(0.99, 0.03, 'labels: ours − theirs, meV', transform=bx.transAxes,
            ha='right', fontsize=6.5, color='0.4')

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}", flush=True)
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=OUT)
    ap.add_argument('--out', default='_yeap_fig6.png')
    ap.add_argument('--h', type=float, default=None,
                    help="force one grid spacing; default takes the finest available per AR")
    ap.add_argument('--crop', type=float, default=4.0)
    a = ap.parse_args()
    main(a.data, a.out, a.h, a.crop)
