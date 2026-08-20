"""HH1 and the electron reference against ASPECT RATIO, one figure per material combination.

    PYTHONIOENCODING=utf-8 python -u scripts/hh1_vs_ar.py

The companion to `scripts/summarise_transitions.py --plot`, which puts E_trans on the y-axis. Here
the two levels that E_trans is the DIFFERENCE of are plotted separately, so the gap closing is a
visible crossing rather than a curve passing through zero:

    E_trans = E_c(InAs, far) - E_HH1

One panel per temperature, one figure per (dot, matrix) pair. The x-axis is log-spaced because the
computed aspect ratios are 1, 2, 4 -- geometric, so a linear axis would bunch them.

WHAT EACH CURVE IS, AND WHICH ONE MOVES
----------------------------------------
`E_c(InAs, far)` is FLAT. It is the matrix conduction edge infinitely far from the island, so it
depends on temperature and on nothing else -- not on the dot material, not on size, not on shape.
Drawn heavy and black. It is also the correct electron reference here rather than a convenience:
`scripts/electron_binding.py` finds no bound electron state at any size in this range, so there is
no electron level to put in its place.

`E_HH1` is the whole story. It rises with AR (flatter binds the hole harder, because biaxial shear
raises the heavy hole) and rises with diameter (a bigger island confines less, so the level sits
closer to the dot's own valence edge). Where it crosses the black line, emission stops.

The dot's own strained heavy-hole edge is drawn faintly as the ceiling HH1 is climbing towards --
it bounds every level in the panel and, unlike HH1, is size-independent at fixed shape.

READ THE CROSSINGS, NOT THE INTERPOLATION
-------------------------------------------
Only three aspect ratios were computed, so the lines between them are guides to the eye and
nothing more. A crossing that falls between two computed AR values is bracketed, not located;
`summarise_transitions.emission_window` is the place that states those brackets honestly.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from summarise_transitions import grade

#: (file label, dot name as stored in the records, pretty name)
COMBOS = (('InSb', 'InSb', 'InSb / InAs'),
          ('InGaSb0.5', 'In0.50Ga0.50Sb', 'In$_{0.5}$Ga$_{0.5}$Sb / InAs'))
TEMPS = (77.0, 300.0)


def load(T):
    path = f"_transitions_{T:g}K.json"
    if not os.path.exists(path):
        raise SystemExit(f"no sweep data at {path}; run scripts/transition_energies.py first")
    out = {}
    for r in json.load(open(path)).values():
        if r.get('status') != 'ok' or 'last_step' not in r or r['matrix'] != 'InAs':
            continue
        out.setdefault(r['dot'], []).append(r)
    return out


def main(out_tmpl):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {T: load(T) for T in TEMPS}
    written = []

    for label, dot, pretty in COMBOS:
        fig, axes = plt.subplots(1, len(TEMPS), figsize=(11.4, 5.6), sharey=True)
        for ax, T in zip(axes, TEMPS):
            rows = data[T].get(dot, [])
            if not rows:
                continue
            bases = sorted({r['base'] for r in rows})
            cmap = plt.get_cmap('viridis')
            Ec = rows[0]['Ec_far']

            # Flat by construction: the matrix conduction edge at infinity depends on T alone.
            assert len({round(r['Ec_far'], 9) for r in rows}) == 1, "Ec_far varies within a panel"
            ax.axhline(Ec, color='k', lw=2.2, zorder=5,
                       label=f"$E_c$(InAs, far) = {Ec:.4f} eV")

            # The ceiling: the dot's own strained HH edge, size-independent at fixed shape.
            vt = {}
            for r in rows:
                vt.setdefault(r['AR'], []).append(r['v_top'])
            ars_v = sorted(vt)
            ax.plot(ars_v, [np.mean(vt[a]) for a in ars_v], '-', color='#b03030', lw=1.3,
                    alpha=0.55, label='dot HH edge (well top)')

            for i, b in enumerate(bases):
                sel = sorted([r for r in rows if r['base'] == b], key=lambda r: r['AR'])
                if len(sel) < 2:
                    continue
                x = [r['AR'] for r in sel]
                y = [r['E_hole'] for r in sel]
                good = [grade(r)[0] == 'ok' for r in sel]
                c = cmap(i / max(len(bases) - 1, 1))
                ax.plot(x, y, '-', color=c, lw=1.5, zorder=3)
                ax.scatter([u for u, g in zip(x, good) if g], [v for v, g in zip(y, good) if g],
                           color=c, s=42, zorder=4, label=f"d = {b:g} nm")
                ax.scatter([u for u, g in zip(x, good) if not g],
                           [v for v, g in zip(y, good) if not g],
                           facecolors='none', edgecolors=c, s=42, zorder=4)

            ax.axhspan(Ec, 0.75, color='#b03030', alpha=0.06, lw=0, zorder=0)
            ax.text(0.98, 0.965, 'HH1 above this line → no gap', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.2, color='#8a2020')
            ax.axhline(0.0, color='0.6', lw=0.8, ls='--', zorder=1)
            ax.set_xscale('log', base=2)
            ax.set_xticks([1, 2, 4])
            ax.set_xticklabels(['1', '2', '4'])
            ax.set_xlabel('aspect ratio  AR = d/h   (flatter $\\rightarrow$)')
            ax.set_title(f"{T:g} K", fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8.5)

        axes[0].set_ylabel('Energy (eV), zero = unstrained InAs VB')
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h_, l_, fontsize=7.6, frameon=False, ncol=5, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"$E_{{HH1}}$ and the electron reference vs aspect ratio — {pretty}\n"
            f"$E_{{trans}} = E_c$(InAs, far) $- E_{{HH1}}$, so the gap closes where a coloured "
            f"curve crosses the black line.  Filled = passes every criterion; open = indicative.",
            fontsize=9.6)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        path = out_tmpl.format(label=label)
        fig.savefig(path, dpi=160, bbox_inches='tight')
        written.append(path)
        print(f"wrote {path}", flush=True)

        # Where each diameter's HH1 sits relative to the reference, in words.
        for T in TEMPS:
            rows = data[T].get(dot, [])
            Ec = rows[0]['Ec_far'] if rows else float('nan')
            emit = sorted({r['base'] for r in rows if r['E_hole'] < Ec})
            print(f"    {T:g} K: E_c(far) = {Ec:.4f} eV; HH1 below it (gap open) at d = "
                  f"{', '.join(f'{b:g}' for b in emit) if emit else 'none'} nm", flush=True)
    return written


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='_hh1_vs_AR_{label}.png')
    a = ap.parse_args()
    main(a.out)
