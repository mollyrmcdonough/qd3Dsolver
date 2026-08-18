"""Band alignment for the Stage 5 structures, with the computed HH1 level drawn on it.

    PYTHONIOENCODING=utf-8 python -u scripts/band_alignment.py [--T 77] [--out FILE]

One panel per (dot, aspect ratio). Each shows the strained conduction and HEAVY-HOLE valence
edges through the island, plus every HH1 level from `scripts/transition_energies.py` at that
shape -- one per diameter.

WHY ONE PROFILE CAN CARRY SEVEN LEVELS
---------------------------------------
Continuum elasticity has no length scale, so at fixed SHAPE the strained band edges are
size-independent and only the confinement changes with size. Verified rather than assumed, at
AR 2 over a 5x size range, sampling the profile at fixed reduced position x/R:

    base  4 nm   HH at x/R = 0, 0.5, 0.9, 1.2, 2.0:  -0.1525 -0.1354 -0.0827 +0.0507 +0.0107
    base 10 nm                                        -0.1525 -0.1354 -0.0827 +0.0507 +0.0107
    base 20 nm                                        -0.1525 -0.1354 -0.0827 +0.0507 +0.0107

Identical to four decimals. So the horizontal axis is x/R, the profile is drawn once per shape,
and the seven HH1 levels are all read against that one profile legitimately.

HOW TO READ IT: THE EMISSION CONDITION IS A CROSSING ON THIS PLOT
------------------------------------------------------------------
These are broken-gap type II. The island's valence edge sits ABOVE the matrix conduction edge, so
the electron is expelled -- and `scripts/electron_binding.py` shows it is not bound anywhere, at
any size in this range, so the electron reference is E_c(InAs, far), drawn as the heavy black
line. The transition is spatially indirect and

    E_trans = E_c(matrix, far) - E_HH1

which means a level BELOW the black line emits and a level above it does not. The gap closing as
the island grows is visible directly as the HH1 ladder walking up through that line.

The y-limits span the FULL conduction edge, including the island's, so the broken-gap statement is
visible rather than implied: the dot's valence edge sits above the matrix's conduction edge, and
the dot's conduction edge is ~810 meV above the matrix's, which is why the electron is expelled.
That costs some vertical resolution in the valence detail, which is the trade being made.

No electron level is drawn, because there is no bound one to draw.

The heavy-hole branch is selected by CHARACTER, not by taking v1. `local_band_edges` numbers its
branches rather than labelling them, and the ordering inverts in the tensile matrix around the
island -- the heavy hole bends down and the light hole up -- so v1 there is the light hole and
plotting it misreports the barrier by several hundred meV. `yeap_band_profile.heavy_hole` does
the selection; this reuses it rather than repeating it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import heterostructure as hs
import materials as mt
from yeap_band_profile import edges, heavy_hole

PROFILE_BASE = 10.0      # nm; any size gives the same normalised profile (see docstring)
CELLS = 12
ARS = (1.0, 2.0, 4.0)
DOTS = (('InSb', 'InSb'), ('In0.50Ga0.50Sb', ('InGaSb', 0.5)))


def profile(dot_spec, AR, T, base=PROFILE_BASE, cells=CELLS):
    """In-plane and growth-direction cuts of the HH and conduction edges, vs reduced position.

    **Cut through the MASK's centre, never the array's.** `island_grid` puts the island base at
    z = 0 while `ellipsoid_mask` centres the ellipsoid on z = 0, so the vertical padding is
    asymmetric and the middle index of the z axis is not the middle of the island. Measured on a
    10 nm sphere: the z axis runs -9.58 .. +19.58 nm, so `nz//2` sits at z = +5.42 nm -- 5.4 nm
    ABOVE an island that spans -4.58 .. +4.58, i.e. in pure matrix.

    Indexing at `nz//2` therefore drew an "in-plane cut" that never entered the dot: it showed the
    HH edge at -0.2 eV where the interior is +0.55, and the two cuts disagreed by 0.75 eV at the
    one point they must agree, the centre. The x and y axes ARE symmetric, which is why only the
    in-plane cut was wrong and the growth-direction one looked fine.
    """
    height = base / AR
    h = max(height / cells, 0.15)
    mt.set_temperature(T)                    # MUST precede build: build stores material copies
    env = hs.build(hs.ellipsoid(base, height), dot_spec, matrix='InAs', h=h,
                   pad=base, z_pad=base, use_piezo=False, vol_tol=0.35, verbose=False)
    e = edges(env)
    hh = heavy_hole(env, e)
    m = env['mask']
    ix, iy, iz = np.argwhere(m).mean(axis=0).round().astype(int)
    assert m[ix, iy, iz], "mask centroid is not inside the mask"
    # Reduced position measured from the island's own centre, for the same reason.
    x0, z0 = float(env['cx'][ix]), float(env['cz'][iz])
    return dict(
        x=(env['cx'] - x0) / (base / 2.0), hh_x=hh[:, iy, iz], ec_x=e['cb'][:, iy, iz],
        z=(env['cz'] - z0) / (height / 2.0), hh_z=hh[ix, iy, :], ec_z=e['cb'][ix, iy, :],
        centre=(float(hh[ix, iy, iz]), float(e['cb'][ix, iy, iz])),
        Ec_far=float(env['Ec_far']), v_top=float(hs.valence_edge_top(env, reduce='mean')))


def load_levels(path, T):
    """HH1 per (dot, AR, diameter) from the sweep, plane-wave records only."""
    if not os.path.exists(path):
        raise SystemExit(f"no sweep data at {path}; run scripts/transition_energies.py first")
    out = {}
    for r in json.load(open(path)).values():
        if r.get('status') != 'ok' or 'last_step' not in r or r['matrix'] != 'InAs':
            continue
        out.setdefault((r['dot'], r['AR']), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r['base'])
    return out


def main(T, data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    levels = load_levels(data, T)
    fig, axes = plt.subplots(len(DOTS), len(ARS), figsize=(13.2, 9.2),
                             sharex=True, sharey=True)
    cmap = plt.get_cmap('viridis')

    # Compute every profile first so the shared y-limits can span the full conduction edge --
    # including the island's, which is the broken-gap statement and must be in frame.
    profs = {(label, AR): profile(spec, AR, T)
             for label, spec in DOTS for AR in ARS}
    lo = min(min(p['hh_x'].min(), p['hh_z'].min()) for p in profs.values())
    hi = max(max(p['ec_x'].max(), p['ec_z'].max()) for p in profs.values())
    span = hi - lo

    for row, (label, spec) in enumerate(DOTS):
        for col, AR in enumerate(ARS):
            ax = axes[row, col]
            p = profs[(label, AR)]
            rows = levels.get((label, AR), [])

            # The two cuts share exactly one point, the island centre, so they must agree there.
            # This is what caught the nz//2 indexing bug: the cuts differed by 0.75 eV. It costs
            # nothing and it is the only free check on the sampling being right.
            i0 = int(np.argmin(np.abs(p['x'])))
            j0 = int(np.argmin(np.abs(p['z'])))
            d = abs(p['hh_x'][i0] - p['hh_z'][j0])
            assert d < 1e-9, (f"{label} AR {AR:g}: cuts disagree at the centre by {d*1e3:.1f} meV "
                              f"-- they sample the same voxel and cannot")

            # The wells, as the carrier sees them.
            ax.plot(p['x'], p['hh_x'], '-', color='#b03030', lw=1.9, label='HH edge, in-plane cut')
            ax.plot(p['z'], p['hh_z'], ':', color='#b03030', lw=1.5,
                    label='HH edge, growth-direction cut')
            ax.plot(p['x'], p['ec_x'], '-', color='#1f60a8', lw=1.6, label='CB edge, in-plane')
            ax.plot(p['z'], p['ec_z'], ':', color='#1f60a8', lw=1.3,
                    label='CB edge, growth-direction cut')

            # The electron reference. Not a bound level -- there is none; this is the matrix
            # conduction edge at infinity, which is what E_trans is measured down from.
            ax.axhline(p['Ec_far'], color='k', lw=2.0,
                       label='$E_c$(InAs, far) — electron reference')
            ax.axhline(0.0, color='0.6', lw=0.8, ls='--', label='unstrained InAs VB (zero)')

            for i, r in enumerate(rows):
                emits = r['E_trans'] > 0
                c = cmap(i / max(len(rows) - 1, 1))
                ax.plot([-2.45, 2.45], [r['E_hole']] * 2, '-', color=c, lw=1.3,
                        alpha=0.95 if emits else 0.45,
                        dashes=(None, None) if emits else (4, 2))
                ax.annotate(f"{r['base']:g}", (2.5, r['E_hole']), fontsize=6.2, color=c,
                            va='center', ha='left')

            # Shade only as far as the levels actually reach. Spanning to the top of the axes
            # floods the dot's own bands once the conduction edge is in frame, which reads as if
            # the whole region meant something.
            top = max([r['E_hole'] for r in rows], default=p['Ec_far'])
            ax.axhspan(p['Ec_far'], top + 0.02, color='#b03030', alpha=0.07, lw=0)
            if row == 0 and col == 0:
                ax.annotate('dot CB edge — ~810 meV above the\nmatrix, so the electron is expelled',
                            xy=(-1.0, p['ec_x'].max()), xytext=(-2.5, p['ec_x'].max() - 0.30),
                            fontsize=6.6, color='#1f60a8', ha='left',
                            arrowprops=dict(arrowstyle='->', color='#1f60a8', lw=0.7))
                ax.annotate('dot VB edge sits ABOVE the matrix CB\n= BROKEN GAP',
                            xy=(0.0, p['centre'][0]), xytext=(-2.5, p['centre'][0] + 0.22),
                            fontsize=6.6, color='#b03030', ha='left',
                            arrowprops=dict(arrowstyle='->', color='#b03030', lw=0.7))
            if row == 0:
                ax.set_title(f"AR = {AR:g}   ({'sphere' if AR == 1 else 'flattened'})",
                             fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{label}\nEnergy (eV), zero = unstrained InAs VB", fontsize=8.5)
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlim(-2.6, 2.9)

    for ax in axes[-1]:
        ax.set_xlabel('reduced position   x/R  (in-plane) or z/(h/2)  (growth)', fontsize=8.5)
    axes[0, 0].set_ylim(lo - 0.06 * span, hi + 0.06 * span)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7.8, frameon=False, ncol=5, loc='lower center',
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Band alignment and HH1 levels, InSb / In$_{{0.5}}$Ga$_{{0.5}}$Sb islands in InAs, "
        f"{T:g} K\n"
        f"coloured lines are HH1 at diameters 2.5–20 nm (labelled, nm); solid = emits, "
        f"faded dashed = no gap.  Band edges are size-independent at fixed shape.\n"
        f"$E_{{trans}} = E_c$(InAs, far) $- E_{{HH1}}$, so a level below the black line emits — "
        f"the gap closes as the ladder walks up through it.", fontsize=9.2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.99))
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f"wrote {out}", flush=True)

    for (label, AR), rows in sorted(levels.items()):
        emit = [r['base'] for r in rows if r['E_trans'] > 0]
        print(f"  {label:>16} AR {AR:g}: emits at d = "
              f"{', '.join(f'{b:g}' for b in emit) if emit else 'none'} nm")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=77.0)
    ap.add_argument('--data', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    data = a.data or f"_transitions_{a.T:g}K.json"
    out = a.out or f"_band_alignment_{a.T:g}K.png"
    main(a.T, data, out)
