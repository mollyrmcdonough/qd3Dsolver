"""Summarise a transition-energy sweep, applying the reliability criteria the sweep cannot.

Run:  python scripts/summarise_transitions.py [_transitions_77K.json] [--plot]

BOTH ORIGINAL CRITERIA HERE WERE WRONG FOR THE CURRENT SWEEP, IN OPPOSITE DIRECTIONS
--------------------------------------------------------------------------------------
This file used to grade on `artifact_ratio` and on a flat `loc >= 0.75`. Neither survives the move
to the plane-wave Burt-Foreman solver, and the failures are worth keeping written down because one
of them would have thrown away every interesting point in silence.

**`artifact_ratio` no longer exists, and should not.** It was (0.046/h^2)/V0, the size of the
symmetrized six-band operator's interface runaway against the well depth -- a contamination
estimate for a solver whose spectrum was unbounded at an abrupt interface. Burt-Foreman ordering
removed the cause (`kp_planewave`), so there is nothing to take a ratio of. Grading now uses real
per-point convergence evidence instead: `last_step`, the final rung of the monotone-from-below
cutoff ladder, which BOUNDS what is left rather than estimating it, and `n_above_bound`, which must
be zero and is the acceptance test for the ordering itself.

**`loc >= 0.75` is a LARGE-DOT criterion and this sweep is mostly small dots.** A 2.5 nm island
genuinely holds its hole at 55%, because a small well spills: measured here, InSb at base 2.5 nm
gives loc 54.8% while base 10 nm at AR 4 gives 88.2%. Both are correct. Applying 0.75 would have
marked the entire small-island end -- the end where the gap actually opens, and the whole point of
the sweep -- as unreliable, while the large islands it passes are the ones with no gap at all.

The honest test is the RANDOM-VECTOR BASELINE: a state that ignores the island entirely scores
island volume / box volume. Below that, the state is actively avoiding the island; far above it,
the state is bound however modest the percentage looks. On the 2.5 nm dot the baseline is ~1%, so
55% is emphatically not a marginal state. `grade` therefore measures loc against each row's own
baseline instead of against a constant.

The FORM of that comparison took two attempts. A plain ratio, loc >= 5 x baseline, works at the
small end and breaks at the large one: at base 20 nm the island is 26.7% of the k.p box, so the
rule demands 133% and no state can satisfy it -- it flagged two 98%-localised points as suspect.
What replaced it is the ENHANCEMENT (loc - baseline)/(1 - baseline), which is 0 for a state that
ignores the island and 1 for one entirely inside it, and so keeps the baseline-relative virtue
without acquiring a ceiling.

WHAT STAYS: CELLS ACROSS THE HEIGHT
-------------------------------------
Thin islands are squeezed from both sides, and below ~6 cells the confining direction is not
resolved -- neighbouring aspect ratios start collapsing onto the same discrete mask and return
near-identical energies, which reads as a converged plateau. **This bites hardest exactly where the
physics is most favourable**: flatter is better for opening a gap, and flatter is where the grid
runs out, so the best-looking points are the least trustworthy. The plot marks that rather than
smoothing it over: filled markers pass every criterion, open markers are indicative only.

Aspect ratio is AR = d/h -- larger is flatter. Files written before 2026-08-05 store h/d under
`aspect`; run `scripts/migrate_aspect_to_AR.py` on those first.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CELLS_MIN = 6.0          # below this the confining direction is not resolved
STEP_MAX = 5.0           # meV; last rung of the cutoff ladder, which bounds the remainder
#: Localisation ENHANCEMENT, (loc - baseline)/(1 - baseline): 0 for a state that ignores the
#: island, 1 for one entirely inside it. A plain ratio loc/baseline was tried first and is wrong at
#: the large-island end -- at base 20 the island is 26.7% of the k.p box, so "5x baseline" demands
#: 133% and can never be met however perfectly bound the state is. Two genuinely well-bound points
#: were flagged that way before this replaced it. The enhancement has the ratio's virtue (it is
#: measured against each row's own random-vector baseline, not a constant) without the ceiling.
LOC_ENHANCEMENT = 0.40
G0 = 0.0381              # hbar^2/2m0, eV.nm^2
M_LH = 0.015             # light-hole admixture mass; the long tail, per kp_planewave
TAIL_OVER_PAD = 1.5      # flag when the lh decay length exceeds the crop by this factor


def tail_length(r):
    """Light-hole decay length sqrt(G0/(m E)) for this row's binding, in nm.

    The six-band hole's heavy component decays in under a nanometre, but its light-hole admixture
    goes as sqrt(G0/(m E)) with m ~ 0.015 -- and that depends on the BINDING, so it diverges
    exactly where the binding goes to zero. That is the critical-size regime this sweep exists to
    locate, which is the awkward part: the answer is hardest to box precisely where it matters.
    """
    E = max(r['binding'] * 1e-3, 1e-6)          # `binding` is stored in meV
    return float(np.sqrt(G0 / (M_LH * E)))


def baseline(r):
    """Random-vector localisation for this row: island volume / k.p box volume.

    A state that ignores the island scores exactly this. It is the only fixed point available,
    and unlike a constant threshold it moves with the geometry -- which is the whole reason a flat
    75% was wrong here (see the module docstring).
    """
    vol_island = (4.0 / 3.0) * np.pi * (r['base'] / 2.0) ** 2 * (r['height'] / 2.0)
    vol_box = float(np.prod(r['grid'])) * r['h'] ** 3
    return vol_island / vol_box


def grade(r):
    """Return (verdict, reasons). 'ok' only if every criterion passes."""
    bad = []
    if r['cells'] < CELLS_MIN:
        bad.append(f"{r['cells']:.1f} cells")
    if r.get('last_step', 0.0) > STEP_MAX:
        bad.append(f"step {r['last_step']:+.1f}m")
    if r.get('n_above_bound'):
        bad.append(f"{r['n_above_bound']} above bound")
    if r.get('unbound'):
        bad.append("unbound (resonance)")
    tail, pad = tail_length(r), r.get('crop_pad', 4.0)
    if tail > TAIL_OVER_PAD * pad:
        bad.append(f"lh tail {tail:.1f} nm > {pad:g} nm crop")
    b = baseline(r)
    enh = (r['loc'] - b) / max(1.0 - b, 1e-9)
    if enh < LOC_ENHANCEMENT:
        bad.append(f"loc {r['loc']*100:.0f}% vs baseline {b*100:.1f}% (enh {enh:.2f})")
    if abs(r['vol_err']) > 0.10:
        bad.append(f"vol {r['vol_err']:+.2f}")
    return ('ok' if not bad else 'indicative'), bad


def emission_window(rows):
    """Bracket the two size limits that decide whether an island emits at all.

    These are broken-gap type II, so a gap exists only in a WINDOW and both edges are size limits
    running in opposite directions:

        TOO LARGE   E_trans = Ec_far - E_hole falls through zero as the island grows, because the
                    hole stops being confined enough to clear the broken-gap offset. Above that
                    size the hole level sits above the matrix conduction edge and there is no
                    transition to quote.
        TOO SMALL   the hole stops being bound at all. `binding` -> 0, the level merges into the
                    matrix valence continuum, and what the solver returns is a resonance whose
                    energy belongs to the box.

    Both are reported as BRACKETS between adjacent computed points, not as fitted crossings. A fit
    would imply a precision the sweep does not have: the light-hole tail diverges as the binding
    goes to zero (see `tail_length`), so the box error grows without bound on approach to the
    small-size edge -- the one place a smooth extrapolation looks most convincing and is least
    justified. Rows that fail `grade` are excluded from the bracket and counted separately, since
    those are exactly the rows nearest both edges.
    """
    print("\nEMISSION WINDOW (brackets between adjacent points, graded rows only)")
    groups = {}
    for r in rows:
        groups.setdefault((r['dot'], r['AR']), []).append(r)
    any_bracket = False
    for (dot, AR), g in sorted(groups.items()):
        good = sorted([r for r in g if grade(r)[0] == 'ok'], key=lambda r: r['base'])
        dropped = len(g) - len(good)
        if len(good) < 2:
            print(f"   {dot:>16} AR {AR:5.2f}: only {len(good)} graded point(s), no bracket"
                  f"{f' ({dropped} excluded)' if dropped else ''}")
            continue
        notes = []
        for a, b in zip(good, good[1:]):
            if a['E_trans'] > 0 >= b['E_trans']:
                notes.append(f"gap closes between d = {a['base']:g} and {b['base']:g} nm")
        # The unbinding edge has to be read over ALL rows, not the graded ones: `grade` rejects
        # `unbound`, so a sign change in `binding` can never appear among `good` and testing for
        # one there would be dead code that reads like a live check. These rows are flagged by
        # construction -- near the edge the light-hole tail exceeds any affordable crop -- so the
        # bracket is labelled indicative rather than silently mixed in with the gap-closing one.
        allr = sorted(g, key=lambda r: r['base'])
        for a, b in zip(allr, allr[1:]):
            if a['binding'] > 0 >= b['binding']:
                notes.append(f"hole unbinds between d = {a['base']:g} and {b['base']:g} nm "
                             f"[indicative]")
        lo, hi = good[0], good[-1]
        if not notes:
            notes.append(f"no crossing in d = {lo['base']:g}-{hi['base']:g} nm "
                         f"(E_trans {lo['E_trans']*1e3:+.0f} -> {hi['E_trans']*1e3:+.0f} meV)")
        any_bracket = True
        print(f"   {dot:>16} AR {AR:5.2f}: {'; '.join(notes)}"
              f"{f'   [{dropped} excluded]' if dropped else ''}")
    if not any_bracket:
        print("   (nothing bracketable yet -- needs >= 2 graded points at one AR)")

    if '--plot' in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Two panels because the sweep has two independent knobs and one x-axis cannot show both.
        # Left is the answer to "what aspect ratio do I need"; right to "what size".
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
        MARKERS = {20.0: 'o', 30.0: 's', 40.0: '^'}
        for x, colour in ((1.00, '#1f77b4'), (0.50, '#d62728')):
            name = 'InSb' if x >= 1 else 'In$_{0.5}$Ga$_{0.5}$Sb'
            for group, xkey, ax in ((sorted({r['base'] for r in rows}), 'AR', axes[0]),
                                    (sorted({r['AR'] for r in rows}), 'base', axes[1])):
                for g, marker in zip(group, ('o', 's', '^', 'v', 'D', 'P', '*')):
                    fixed = 'base' if xkey == 'AR' else 'AR'
                    sel = sorted([r for r in rows if r['x'] == x and r[fixed] == g],
                                 key=lambda r: r[xkey])
                    if len(sel) < 2:
                        continue
                    u = [r[xkey] for r in sel]
                    e = [r['E_trans'] * 1e3 for r in sel]
                    solid = [grade(r)[0] == 'ok' for r in sel]
                    lbl = (f"{name}, d={g:g} nm" if xkey == 'AR'
                           else f"{name}, AR={g:.3g}")
                    ax.plot(u, e, '-', color=colour, alpha=0.35, lw=1.2)
                    ax.scatter([q for q, s in zip(u, solid) if s],
                               [q for q, s in zip(e, solid) if s],
                               marker=marker, color=colour, s=55, label=lbl, zorder=5)
                    ax.scatter([q for q, s in zip(u, solid) if not s],
                               [q for q, s in zip(e, solid) if not s],
                               marker=marker, facecolors='none', edgecolors=colour, s=55, zorder=5)
        for ax, xlabel in zip(axes, ('aspect ratio  AR = d/h   (flatter $\\rightarrow$)',
                                     'island diameter d (nm)')):
            ax.axhline(0, color='k', lw=1.2)
            ax.axhspan(0, 400, color='#2e7d32', alpha=0.06)
            ax.set_xlabel(xlabel)
            ax.legend(fontsize=7.5, frameon=False)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('$E_c$(InAs, far) $-$ $E_{hole}$   (meV)')
        fig.suptitle(f'Transition energy at {T:g} K  (above zero = gap open)   |   '
                     'filled = meets every reliability criterion; open = indicative only',
                     fontsize=10)
        fig.tight_layout()
        p = path.replace('.json', '.png')
        fig.savefig(p, dpi=150)
        print(f"\nwrote {p}")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else '_transitions_77K.json'
    data = json.load(open(path))
    if any('aspect' in r for r in data.values()):
        sys.exit(f"{path} holds pre-2026-08-05 h/d records; run "
                 f"scripts/migrate_aspect_to_AR.py {path} first")
    rows = [r for r in data.values() if r.get('status') == 'ok']
    if not rows:
        sys.exit(f"no completed points in {path}")

    # A sweep file accumulates -- the resume key carries base/AR/T but NOT the solver, so a file
    # written by the old finite-difference driver keeps its records when the plane-wave driver is
    # run over the same output. Measured: _transitions_77K.json held 26 symmetrized-FD points at
    # bases 20-40 alongside the new ones, with no key collision to reveal it. Those two solvers do
    # not belong on one axis (the symmetrized operator has an interface runaway; see kp_planewave),
    # so segregate rather than grade them together. `last_step` is the discriminator because only
    # the plane-wave path runs a cutoff ladder; `method` tags records written since 2026-08-14.
    legacy = [r for r in rows if 'last_step' not in r and r.get('method') is None]
    rows = [r for r in rows if r not in legacy]
    if legacy and '--include-legacy' not in sys.argv:
        print(f"NOTE: {len(legacy)} pre-plane-wave record(s) in this file are EXCLUDED "
              f"(bases {sorted({r['base'] for r in legacy})}).\n"
              f"      They come from the symmetrized six-band FD solver and carry its interface\n"
              f"      runaway; grading them with the criteria below would score them as converged\n"
              f"      because they have no cutoff ladder at all. Pass --include-legacy to see "
              f"them.\n")
    if not rows:
        sys.exit(f"no plane-wave points in {path}")
    rows.sort(key=lambda r: (r['x'], -r['AR'], r['base']))
    T = rows[0]['T']

    print(f"{path}: {len(rows)} completed point(s), T = {T:g} K\n")
    print(f"{'dot':>16} {'base':>5} {'AR=d/h':>7} {'height':>7} {'cells':>6} {'bind':>8} "
          f"{'loc':>6} {'base%':>6} {'tail':>6} {'step':>7} {'E_trans':>9} {'lambda':>8}  verdict")
    print("-" * 124)
    n_ok = 0
    for r in rows:
        verdict, why = grade(r)
        n_ok += verdict == 'ok'
        lam = f"{r['lam']:7.2f}u" if r['lam'] else " no gap"
        print(f"{r['dot']:>16} {r['base']:5.1f} {r['AR']:7.2f} {r['height']:7.2f} "
              f"{r['cells']:6.1f} {r['binding']:+7.1f}m {r['loc']*100:5.1f}% "
              f"{baseline(r)*100:5.1f}% {tail_length(r):5.1f}n {r['last_step']:+6.2f}m "
              f"{r['E_trans']:+9.3f} {lam:>8}  "
              f"{verdict}{'  (' + ', '.join(why) + ')' if why else ''}")

    print(f"\n{n_ok} of {len(rows)} meet every criterion (cells >= {CELLS_MIN:g}, "
          f"step <= {STEP_MAX:g} meV, nothing above the spectrum bound, bound not resonant,"
          f"\nloc enhancement >= {LOC_ENHANCEMENT:g}, |vol err| <= 0.10, "
          f"lh tail <= {TAIL_OVER_PAD:g}x the crop pad).")

    open_gap = [r for r in rows if r['E_trans'] > 0]
    print(f"{len(open_gap)} point(s) with an open gap.")
    if not open_gap:
        best = max(rows, key=lambda r: r['E_trans'])
        print(f"   closest: {best['dot']} base {best['base']:g} nm, AR {best['AR']:.2f} "
              f"-> {best['E_trans']*1e3:+.0f} meV ({grade(best)[0]})")
    emission_window(rows)
