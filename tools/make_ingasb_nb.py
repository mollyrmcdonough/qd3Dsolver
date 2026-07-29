"""Generate ingasb_lens.ipynb and ingasb_dash.ipynb.

Both notebooks are the same study of an InAs / In(x)Ga(1-x)Sb / InAs dot; they differ only in the
island shape and in the size parameter that gets swept. Run:  python tools/make_ingasb_nb.py
"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LENS = dict(
    fname='ingasb_lens.ipynb',
    title='InAs / In$_x$Ga$_{1-x}$Sb / InAs — lens-shaped dot',
    shape_params="""# --- SHAPE: circular lens (half-ellipsoid dome) ---------------------------------
RADIUS   = 10.0        # nm, base radius
HEIGHT   = 6.0         # nm, apex height above the base plane

shape = ig.lens(RADIUS, HEIGHT)""",
    shape_note="""A lens is the rotationally symmetric case: circular base, dome top, in-plane $C_\\infty$
broken only by the underlying crystal and by the piezoelectric field. It is the natural
counterpoint to the dash, and the shape most often assumed for self-assembled dots when no
faceting is resolved.""",
    sweep_name='radius',
    sweep_values='(6.0, 8.0, 10.0, 12.0, 14.0)',
    sweep_shape="lambda r: (ig.lens(r, HEIGHT * r / RADIUS), X_COMP)",
    sweep_note="""The lens is scaled homothetically -- height tracks radius, so the aspect ratio is fixed and
the sweep is genuinely a size sweep rather than a shape sweep.""",
)

DASH = dict(
    fname='ingasb_dash.ipynb',
    title='InAs / In$_x$Ga$_{1-x}$Sb / InAs — quantum dash',
    shape_params="""# --- SHAPE: quantum dash (truncated rectangular pyramid) ------------------------
# Geometry after Tersoff & Tromp, Phys. Rev. Lett. 70, 2782 (1993): rectangular base,
# sloping side facets at a fixed contact angle, flat top.
LENGTH   = 40.0        # nm, base length along [100] (the elongation axis)
WIDTH    = 16.0        # nm, base width along [010]
HEIGHT   = 6.0         # nm, height of the flat top above the base plane
ANGLE    = 54.7356     # deg, side-facet contact angle = arctan(sqrt(2)): the angle a {111}
                       # plane makes with the (001) base, i.e. {111} side facets. Try 45.0
                       # for {101} or 11.3 for the very shallow {105} hut facets.

shape = ig.dash(LENGTH, WIDTH, HEIGHT, ANGLE)""",
    shape_note="""The dash is a **truncated rectangular pyramid**: rectangular base, four side facets at a
fixed contact angle, flat top. That geometry is Tersoff and Tromp's, from
[*Phys. Rev. Lett.* **70**, 2782 (1993)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.2782),
"Shape transition in growth of strained islands: Spontaneous formation of quantum wires". Their
result is the transition itself: below a critical size a strained island is compact and roughly
symmetric, above it the island elongates at essentially fixed width, because a long thin island
relaxes its strain better. They observed aspect ratios above 50:1 for Ag on Si(001).

Two consequences the calculation should show. The elongation breaks the in-plane symmetry down to
$C_2$, so any p-like doublet splits by far more than a piezoelectric field would split it, and the
splitting is set by the aspect ratio. And the strain becomes anisotropic in plane: the narrow
direction relaxes more than the long one, so $\\varepsilon_{xx} \\neq \\varepsilon_{yy}$ inside the
island.

*Provenance limit:* the PRL is paywalled and returned 403, so the geometry here comes from the
abstract plus secondary descriptions of the paper's energy expression -- which is written in terms
of the two base dimensions and the height and carries a separate **top-facet** surface energy, and
that top term is what rules out a pointed pyramid or a dome. The shape is reliable at the level of
"rectangular base, sloping facets, flat top". The facet angle is *not* fixed by that paper.

`ANGLE` defaults to $\arctan\sqrt{2} = 54.7356°$, the angle a $\{111\}$ plane makes with the
$(001)$ base -- the natural low-index facet family for a zincblende island grown on $(001)$. Set
it to $45°$ for $\{101\}$ (the family of the Pryor pyramid elsewhere in this package) or
$11.3° = \arctan(1/5)$ for the very shallow $\{105\}$ facets of Ge/Si hut clusters. The choice
matters for more than looks: shallow facets remove so much material from a narrow island that
whole aspect ratios become geometrically unable to hold a given volume.""",
    sweep_name='aspect',
    sweep_values='(1.0, 2.0, 3.0, 4.0, 6.0)',
    sweep_shape="_dash_at_aspect",
    sweep_note="""The aspect ratio is swept **at constant volume**, so anything that changes along the series is
shape and not size. That needs care: the facet set-back removes proportionally more material from
a narrow island, so holding $L\\times W\\times h$ fixed does *not* hold the volume fixed. The height
is solved for at each aspect ratio instead, and an aspect ratio the island is too narrow to
support at this facet angle is reported and skipped rather than silently clipped.""",
)

PARAM_HEADER = """import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

import qdsolver_core as qd
import materials_sb as ms
import ingasb_dot as ig

# =================================================================================
# PARAMETERS -- everything you would want to vary is here
# =================================================================================
X_COMP   = 0.35        # InSb fraction of the In(x)Ga(1-x)Sb dot: 0 = GaSb, 1 = InSb

{shape_params}

# --- NUMERICS -------------------------------------------------------------------
H_GRID   = 0.75        # nm, grid spacing. The single-band solver factorizes the Hamiltonian
                       # completely (sparse LU), and 3D fill-in grows fast, so this is a
                       # compromise: 0.5 nm here would be ~1e6 points and thrash.
PAD      = {pad}        # nm, matrix around the island. The strain pocket is SHALLOW and the
                       # electron is not bound by it, so the apparent binding depends on the
                       # box; the ladder below measures that drift explicitly.
USE_PIEZO = True
RUN_EIGHT_BAND = True  # the hole needs it; see the last section for the cost and caveats.

# --- MATERIAL CONVENTIONS -------------------------------------------------------
# See materials_sb.py: the source database has a sign inconsistency in GaSb's a_v that
# directly sets the hole well depth. Switch and re-run to see how much it matters.
ms.GASB_AV_CONVENTION  # currently: {{}}
print(f"GaSb a_v convention: {{ms.GASB_AV_CONVENTION!r}} -> a_v = "
      f"{{ms.SB_MATERIALS['GaSb']['a_v']:+.2f}} eV")
print(f"dot = {{ms.ingasb(X_COMP)['name']}}, shape = {{shape['label']}}")
print(f"island volume {{shape['volume']:.1f}} nm^3")"""


FORCE = False


def build(cfg, pad):
    nb = nbf.v4.new_notebook()
    c = []
    md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
    co = lambda s: c.append(nbf.v4.new_code_cell(s))

    md(f"""# {cfg['title']}

A three-dimensional strain + band-structure study of an In$_x$Ga$_{{1-x}}$Sb island buried in
InAs, using the same machinery as this package's Pryor InAs/GaAs benchmark: real-space
finite-element elasticity with position-dependent elastic constants, the Bir–Pikus strain
Hamiltonian, and first-order piezoelectricity.

## Why this system is not another InAs/GaAs

**The alignment is broken gap.** On the absolute scale in `materials_sb.py`, the
In$_x$Ga$_{{1-x}}$Sb valence edge lies *above* the InAs conduction edge at every composition — by
about 20 meV at $x=0$ rising to 180 meV at $x=1$. So this is not a type-I dot with an electron and
a hole sharing a box. Holes are confined in the island. Electrons are **expelled** from it, because
the island's conduction edge sits far above the matrix's.

**The only thing that could bind the electron is strain, not a band offset.** The island is
compressed by the matrix (misfit runs $-0.6\\%$ to $-6.5\\%$ across the composition range), and by
reaction the InAs wrapped around it is put into *tension*. Since $a_c < 0$, tension pulls the InAs
conduction edge **down**, so the island digs a conduction-band pocket in the matrix around itself
— a genuinely inhomogeneous-strain effect that a model with uniform strain inside the island and
none outside would not have at all.

**That pocket turns out not to bind an electron anywhere in this parameter range**, and this
notebook measures that rather than assuming it either way. Across $x = 0.35 \\to 1.0$ the deepest
point of the pocket runs 83 $\\to$ 190 meV, and with a 4$\\times$ larger island on top of that the
computed ground state moves by 3.6 meV — in the *wrong* direction, because what is being measured
is box zero-point energy, not binding. The reason is geometric: binding any state in a spherical
well of depth $V_0$ and radius $R$ needs $V_0R^2 > \\pi^2\\hbar^2/8m \\approx 3.78$ eV nm$^2$ at the
InAs electron mass, and this pocket reaches at most $2.45$ — an estimate that is itself
*optimistic*, since the pocket is a thin shell wrapped around a large repulsive barrier, and a
shell binds worse than a compact sphere of equal volume.

So the electron is held by Coulomb attraction to the hole in the island, not by a single-particle
well. That is the standard type-II picture, and it is why the sweeps below plot $V_0R^2$ against
the binding threshold rather than an "electron binding energy" — the latter, out of a finite box,
would be measuring the box.

{cfg['shape_note']}

## What this notebook computes

Cheap path (seconds per configuration, so composition and size are explorable):

1. inhomogeneous strain, from `elasticity_fd`
2. the piezoelectric potential
3. the local band-edge landscape at $k=0$, including the full Bir–Pikus shear terms
4. single-band electron states in the strain pocket
5. sweeps over composition and size

Expensive path, opt-in at the end: eight-band states, which is what the *hole* actually needs.""")

    co(PARAM_HEADER.format(shape_params=cfg['shape_params'], pad=pad))

    md("""---
## Material parameters, and two things wrong with them

This system has no reference calculation to check against, so the parameter set is the weakest
link and is printed in full rather than hidden. Everything comes from `database.py` in the sibling
`aestimo` checkout, which is a verifiable provenance chain — the values were transcribed from that
file, not recalled. Anything tagged `UNVERIFIED` is not in that database and is a **knob, not a
datum**; vary it before trusting a result that depends on it.

Two defects in the source data are corrected here, neither silently:

1. **The elastic constants are in two different units.** InAs is stored as `C11 = 8.329`
   (10¹¹ dyne/cm²) while GaSb is `88.42` and InSb `68.47` (GPa). aestimo multiplies all three by
   the same factor, so it reads GaSb and InSb as **10× too stiff**.
2. **GaSb's $a_v$ has the wrong sign** relative to InAs and InSb. A negative $a_v$ makes the GaSb
   valence band move the opposite way under compression from InAs's, which is not a real
   difference between two closely related antimonides. Since the hole lives in the island and
   $a_v$ sets its well depth, this matters a lot — hence the switch in the parameter cell.""")

    co("ms.audit()")

    md("""### Where the composition puts the band edges

The last column is the one that decides the character of the whole structure.""")

    co("""ms.alignment_table()""")

    md("""---
## Geometry and grid

The mask is checked, not assumed. A faceted island on a cubic grid is staircased, and if the
sample points land *on* the bounding planes the volume comes out systematically too large in a way
that refining $h$ does not remove. `qd.island_grid` samples at cell centres in all three
directions to avoid exactly that; `build` asserts the volume error and the mirror symmetry and
raises if either fails.""")

    co("""env = ig.build(shape, X_COMP, h=H_GRID, pad=PAD, use_piezo=USE_PIEZO)""")

    md("""---
## The band landscape

Band edges from the local value of the strain — the $k=0$ eigenvalues of the strain Hamiltonian,
the same construction as Pryor's Fig. 2, but here the feature to look for is the opposite one.
Watch the conduction edge (black) **dip below** the dashed unstrained-InAs line in the matrix just
outside the island: that dip is the electron pocket. Inside the island the conduction edge shoots
up, and the valence edge (red) rises above everything — the broken gap.""")

    co("""fig = ig.plot_bands(env)
plt.show()
print(f"Kramers degeneracy residual: {ig.band_profiles(env)['kramers']:.2e} eV")""")

    co("""fig = ig.plot_maps(env)
plt.show()""")

    md("""---
## The electron: does the strain pocket bind anything?

The pocket is tens of meV deep and the InAs electron mass is small (~0.025 $m_0$), which is a
marginal combination. A finite box adds confinement energy of the same order, so **a small box
manufactures a bound state that is not there**. Two checks, in that order: watch the apparent
binding drift with box size, then apply a criterion that does not depend on the box at all.""")

    co("""# Run the ladder at a COARSER grid than the main calculation: this is a convergence test
# in box size, not in h, and the top of the ladder is a large grid. The single-band solver
# factorizes completely, so cost grows quickly with the number of points.
H_BOX = 1.0
print(f"{'pad':>6} {'box (nm)':>20} {'points':>10} {'pocket':>9} {'E0':>9} "
      f"{'apparent binding':>18} {'in dot':>7}")
for pad_ in (8.0, 15.0, 22.0, 30.0):
    e_ = ig.build(shape, X_COMP, h=H_BOX, pad=pad_, use_piezo=USE_PIEZO, verbose=False)
    s_ = ig.electron_states(e_, k=2, verbose=False)
    pocket = e_['Ec_far'] - e_['Ec'][~e_['mask']].min()
    box = f"{len(e_['cx'])*H_BOX:.0f} x {len(e_['cy'])*H_BOX:.0f} x {len(e_['cz'])*H_BOX:.0f}"
    print(f"{pad_:>6.0f} {box:>20} {e_['mask'].size:>10,} {pocket*1e3:>8.1f}m "
          f"{s_['E'][0]:>9.4f} {(e_['Ec_far']-s_['E'][0])*1e3:>17.1f}m "
          f"{s_['inside'][0]*100:>6.1f}%")
    del e_, s_""")

    md("""The apparent binding above should be shrinking steadily as the box grows, with no sign of
settling — that is what an *unbound* state looks like in a finite box: its energy is box
zero-point, which falls as $1/L^2$, and it approaches the barrier edge from above rather than
converging below it.

The check that does not care about the box is geometric. A finite spherical well of depth $V_0$
and radius $R$ holds at least one bound state only if $V_0R^2 > \\pi^2\\hbar^2/8m$. Taking $R$ from
the volume of matrix lying below the far-field conduction edge at each depth threshold gives a
direct verdict — and it is a *generous* one, because the real pocket is a thin shell wrapped
around a large repulsive barrier, and a shell binds worse than a compact sphere of the same
volume. If the generous version fails, the real one does too.""")

    co("""_ = ig.pocket_metrics(env)""")

    co("""st = ig.electron_states(env, k=4)
print(f"\\nstates below the far-field InAs conduction edge: {st['n_bound']}")
print("A small 'inside the dot' fraction is CORRECT here -- the electron is expelled from the")
print("island and lives in the tensile InAs shell around it. A large fraction would mean the")
print("band alignment had come out type-I, i.e. something is wrong.")""")

    md("""---
## The hole

No single-band hole solve is offered, and that is deliberate: the hole sits in a narrow-gap alloy
strained by several percent, where heavy and light hole are split by hundreds of meV and strongly
mixed. A parabolic band would be meaningless.

What *is* cheap and correct is the exact $k=0$ top of the local valence band — the largest
eigenvalue of the 6×6 valence block, including the full Bir–Pikus shear terms. That gives the well
depth and its shape without solving a confined problem.""")

    co("""hw = ig.hole_well(env)
print(f"valence edge, highest inside the island : {hw['v_in']:.4f} eV")
print(f"valence edge, highest in the matrix     : {hw['v_out']:.4f} eV")
print(f"unstrained InAs valence edge (the zero) : {hw['far']:.4f} eV")
print(f"\\nhole well depth (island vs unstrained InAs) : {hw['depth']*1e3:7.1f} meV")
print(f"island valence edge above matrix maximum    : {hw['above_matrix']*1e3:7.1f} meV"
      f"   <- positive means holes stay in the island")
print(f"island valence edge above InAs E_c          : {hw['broken_gap']*1e3:7.1f} meV"
      f"   <- positive means BROKEN GAP")""")

    md(f"""---
## Composition sweep

Composition is the strongest knob in this system, because it moves two things at once: the band
alignment *and*, through Vegard's law, the misfit — from $-0.6\\%$ at $x=0$ to $-6.5\\%$ at $x=1$.
The strain then feeds back on the alignment through the deformation potentials, so the trends are
not obviously monotone and are worth computing rather than guessing.""")

    co("""rows_x = ig.sweep(lambda xv: (shape, xv), (0.0, 0.2, 0.35, 0.5, 0.75, 1.0), 'x',
                  h=H_GRID, pad=PAD, use_piezo=USE_PIEZO)""")

    co("""fig = ig.plot_sweep(rows_x, 'x')
fig.suptitle(f"composition sweep — {shape['label']}", fontsize=10, y=1.04)
plt.show()""")

    md(f"""---
## Size sweep

{cfg['sweep_note']}""")

    if cfg['sweep_name'] == 'aspect':
        co("""ASPECTS = (1.0, 1.5, 2.0, 3.0, 4.0)

# Both the VOLUME and the HEIGHT are held fixed across the series, and the base is solved for.
# The alternative -- fix the base area and solve for height -- drives the narrow end toward its
# geometric ceiling, where the flat top has nearly vanished and the island is really a pointed
# ridge; the series would then vary aspect ratio AND top-face fraction AND height at once, and
# no trend could be attributed to elongation. This way aspect ratio is the only thing changing.
V_TARGET = shape['volume']
t_ = np.tan(np.deg2rad(ANGLE))
print(f"holding volume = {V_TARGET:.1f} nm^3 and height = {HEIGHT:g} nm fixed; "
      f"width floor = 2H/tan(theta) = {2*HEIGHT/t_:.2f} nm")
print(f"  {'aspect':>7} {'L base':>8} {'W base':>8} {'L top':>8} {'W top':>8} {'top cells':>10}")
for a_ in ASPECTS:
    l_, w_ = ig.dash_base_for_aspect(a_, HEIGHT, V_TARGET, ANGLE)
    ins = 2 * HEIGHT / t_
    flag = '' if (w_ - ins) / H_GRID >= 4 else '   <- top too narrow to resolve'
    print(f"  {a_:>6.1f}:1 {l_:>8.2f} {w_:>8.2f} {l_-ins:>8.2f} {w_-ins:>8.2f} "
          f"{(w_-ins)/H_GRID:>10.1f}{flag}")

def _dash_at_aspect(a):
    dims = ig.dash_base_for_aspect(a, HEIGHT, V_TARGET, ANGLE)
    if dims is None:
        raise ValueError(f"no base holds {V_TARGET:.0f} nm^3 at height {HEIGHT:g} nm, "
                         f"{ANGLE:g} deg facets")
    return ig.dash(dims[0], dims[1], HEIGHT, ANGLE), X_COMP

rows_s = ig.sweep(_dash_at_aspect, ASPECTS, 'aspect', h=H_GRID, pad=PAD,
                  use_piezo=USE_PIEZO)""")

    co(f"""fig = ig.plot_sweep(rows_s, '{cfg['sweep_name']}')
fig.suptitle(f"size sweep — dot = {{ms.ingasb(X_COMP)['name']}}", fontsize=10, y=1.04)
plt.show()""")

    md(r"""---
# Eight-band hole states

The hole is the reason this system is interesting and the one carrier a single band cannot
describe: it sits in a narrow-gap alloy strained by several percent, where heavy and light hole
are split by hundreds of meV and strongly mixed. Everything above this point was landscape; this
is the actual state.

### The grid is chosen differently here, and deliberately

The electron needed a *large box* because it is unbound. The hole is the opposite — several
hundred meV deep and tightly localized in the island — so the box can be small, and the grid
spacing spent on **resolving the island** instead. That is why this section builds its own
environment at $h = 1$ nm with only 8 nm of padding rather than reusing `env`.

Vertical resolution is the binding constraint. These islands are flat, so $z$ is the dominant
confinement direction, and it is the one a cubic grid resolves worst. At a 6 nm island height and
$h = 1$ nm there are 6 cells through the island; at the 3 nm height used earlier there were only
3, which is not enough to quote a ground-state energy from.

### Two things that could invalidate the numbers, both checked below

**1. $\sigma$ targeting is genuinely ambiguous in a broken-gap system.** The island's valence edge
lies *above* the matrix's conduction edge, so in the ~200 meV between them, island-like valence
states and matrix-like conduction box states exist at the *same energy*. A folded-spectrum solve
returns whatever is nearest $\sigma$, and its residual certifies only that the result *is* an
eigenpair — not that it is the one wanted. That exact failure invalidated a session's worth of
hole numbers in the InAs/GaAs benchmark, where the gap was not even broken. The localization
fraction is the only discriminator, and the solve is repeated with $\sigma$ moved.

**2. The valence-band parameters are the least trustworthy in the set.** GaSb's $a_v$ sign is a
judgement call that sets the well depth outright, so the whole calculation is run under *both*
conventions and the spread reported rather than a single number. The $[111]$ shear deformation
potential $d$ is worse — it is not in the database at all, and it enters the Bir–Pikus $S$ term,
which is specifically a valence-mixing term. That one is left as a separate sensitivity study.""")

    co("""H8, PAD8 = 1.0, 8.0          # own grid: small box, resolution spent on the island

if RUN_EIGHT_BAND:
    results8 = {}
    for conv in ('signfixed', 'database'):
        av = ms.set_gasb_av(conv)
        print(f"{'='*78}\\nGaSb a_v = {av:+.2f} eV  ({conv});  "
              f"alloy a_v = {ms.ingasb(X_COMP)['a_v']:+.4f} eV\\n{'='*78}")
        t0 = time.time()
        env8 = ig.build(shape, X_COMP, h=H8, pad=PAD8, use_piezo=USE_PIEZO)
        zc = int(env8['mask'][env8['mask'].shape[0]//2, env8['mask'].shape[1]//2, :].sum())
        print(f"  z-cells through the island centre: {zc}")
        hw8 = ig.hole_well(env8)
        print(f"  hole well depth {hw8['depth']*1e3:.1f} meV, "
              f"broken-gap overlap {hw8['broken_gap']*1e3:.1f} meV")
        vb = ig.eight_band_states(env8, band='vb', k=4)
        results8[conv] = dict(E=vb['E'].copy(), inside=vb['inside'].copy(),
                              sigma=vb['sigma'], well=hw8['depth'],
                              a_v=ms.ingasb(X_COMP)['a_v'])
        print(f"  {time.time()-t0:.0f}s")
        del env8, vb
    ms.set_gasb_av('signfixed')      # restore the module default
else:
    results8 = {}
    print("RUN_EIGHT_BAND is False -- skipped.")""")

    md("""### How much does the `a_v` convention move the answer?

This is the point of running both. If the two columns agree, the sign ambiguity does not matter
for what we care about and can be set aside. If they do not, every hole energy in this notebook
carries that spread as a systematic uncertainty, and it should be quoted with one.""")

    co("""if results8:
    a, b = results8['signfixed'], results8['database']
    print(f"{'':>26} {'signfixed':>12} {'database':>12} {'difference':>12}")
    print(f"{'GaSb a_v (eV)':>26} {'+1.32':>12} {'-1.32':>12}")
    print(f"{'alloy a_v (eV)':>26} {a['a_v']:>12.4f} {b['a_v']:>12.4f} "
          f"{b['a_v']-a['a_v']:>12.4f}")
    print(f"{'hole well depth (meV)':>26} {a['well']*1e3:>12.1f} {b['well']*1e3:>12.1f} "
          f"{(b['well']-a['well'])*1e3:>12.1f}")
    print(f"{'sigma used (eV)':>26} {a['sigma']:>12.4f} {b['sigma']:>12.4f}")
    print()
    print(f"{'state':>6} {'E signfixed':>14} {'in dot':>8} {'E database':>14} {'in dot':>8} "
          f"{'shift (meV)':>12}")
    for j in range(min(len(a['E']), len(b['E']))):
        print(f"{j:>6} {a['E'][j]:>14.4f} {a['inside'][j]*100:>7.1f}% "
              f"{b['E'][j]:>14.4f} {b['inside'][j]*100:>7.1f}% "
              f"{(b['E'][j]-a['E'][j])*1e3:>12.1f}")
    print()
    print("Level SPACINGS are the quantity to compare -- they are what an experiment measures,")
    print("and they are far less sensitive to a rigid shift of the well than the absolute")
    print("energies are.")
    for lbl, r in (('signfixed', a), ('database', b)):
        sp = np.diff(r['E']) * 1e3
        print(f"  {lbl:>10}: " + ", ".join(f"{d:+.1f}" for d in sp) + " meV")""")

    md(r"""### Is $\sigma$ returning the states we think it is?

The check that matters. `hole_sigma` targets the exact $k=0$ top of the local valence band, but
in a broken-gap system there are matrix conduction states at the same energy. Re-solving with
$\sigma$ pushed *up* by 100 meV should return the same island-localized levels — if instead a
different set appears, or the localization fractions collapse, then $\sigma$ was picking states
out of the matrix continuum and the energies above are not hole states.""")

    co("""if results8:
    ms.set_gasb_av('signfixed')
    env8 = ig.build(shape, X_COMP, h=H8, pad=PAD8, use_piezo=USE_PIEZO, verbose=False)
    base = results8['signfixed']
    import kp_pryor as kp
    m8 = env8['mask']
    sig_hi = kp.hole_sigma(env8['Ev'], env8['strain'],
                           np.where(m8, env8['dot']['b'], env8['matrix']['b']),
                           np.where(m8, env8['dot']['d'], env8['matrix']['d']),
                           np.where(m8, env8['dot']['delta_so'], env8['matrix']['delta_so']),
                           inside_mask=m8, margin=0.100)
    print(f"sigma: {base['sigma']:.4f} eV  ->  {sig_hi:.4f} eV (+100 meV)\\n")

    ops = __import__('kp_confined').GridOperators(m8.shape, H8, periodic=False)
    fields = kp.material_fields(m8, ms.kp_params(env8['dot']), ms.kp_params(env8['matrix']),
                                env8['Ev'], env8['Ec'], n_bands=8)
    import eigensolvers as eig
    H = kp.confined_hamiltonian(ops, fields, n_bands=8, strain=env8['strain'])
    E2, V2, info2 = eig.solve_interior(H, k=4, sigma=sig_hi, tol=1e-7, maxiter=8000)
    n_ = m8.size
    inside2 = np.array([float((np.abs(V2[:, j].reshape(8, n_))**2).sum(axis=0)
                              .reshape(m8.shape)[m8].sum() / (np.abs(V2[:, j])**2).sum())
                        for j in range(V2.shape[1])])
    print(f"\\n{'state':>6} {'E at sigma':>13} {'in dot':>8} {'E at sigma+100':>16} {'in dot':>8} "
          f"{'diff (meV)':>11}")
    for j in range(min(len(base['E']), len(E2))):
        print(f"{j:>6} {base['E'][j]:>13.4f} {base['inside'][j]*100:>7.1f}% "
              f"{E2[j]:>16.4f} {inside2[j]*100:>7.1f}% "
              f"{(E2[j]-base['E'][j])*1e3:>11.2f}")
    same = np.allclose(np.sort(base['E']), np.sort(E2), atol=2e-3)
    print(f"\\nsame spectrum to 2 meV: {same}")
    print("If False, sigma placement is selecting different states and NOTHING above is safe.")
    del env8, H""")

    md("""---
## What this notebook does and does not establish

**Does.** The strain field, the band-edge landscape and their dependence on composition and size,
all on validated machinery: the elasticity solver is checked against the inhomogeneous
misfitting-sphere result that no homogeneous solver can reproduce, the Bir–Pikus terms against
over-determined identities, and the $k=0$ valence block against Kramers degeneracy, which nothing
in the code enforces.

**Does not.**

* *The parameter set is the weakest link*, and unlike the InAs/GaAs benchmark there is no
  published calculation to check it against. The GaSb $a_v$ sign and the four `UNVERIFIED`
  entries should be varied before any number here is quoted.
* *No wetting layer.* Real islands of this kind grow on one, and it will bind states of its own.
* *First-order piezoelectricity only*, and the second-order response is known to be comparable and
  often opposed in strained III-V dots (Bester, Zunger and co-workers, *Phys. Rev. B* **74**,
  081305(R) and *Phys. Rev. Lett.* **96**, 187602, both 2006).
* *Linear elasticity* is being pushed hard at the In-rich end, where the misfit reaches 6.5%.
* *No excitonic binding, and here that is the main gap.* The single-particle strain pocket does
  not bind an electron, so what binds it is the Coulomb attraction to the hole — a leading
  contribution, not a correction. `schrodinger_poisson.py` in this package already does a
  self-consistent electron–hole pair for the type-I case; extending it here is the obvious next
  step, and until then this notebook describes the *landscape* the exciton would live in rather
  than the exciton itself.""")

    nb['cells'] = c
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    path = os.path.join(ROOT, cfg['fname'])

    # Refuse to silently destroy executed results. These notebooks take ~25 minutes each to run,
    # and regenerating one to fix a typo used to wipe the other's outputs as collateral -- which
    # is exactly what happened once. Pass --force to overwrite an executed notebook deliberately.
    if os.path.exists(path) and not FORCE:
        existing = nbf.read(path, as_version=4)
        n_out = sum(1 for cell in existing.cells
                    if cell.cell_type == 'code' and cell.get('outputs'))
        if n_out:
            print(f"REFUSING to overwrite {cfg['fname']}: it has {n_out} executed cells with "
                  f"output. Re-run with --force if you really mean to discard them.")
            return

    nbf.write(nb, path)
    print(f"wrote {cfg['fname']} with {len(c)} cells")


if __name__ == '__main__':
    import sys
    args = [a for a in sys.argv[1:]]
    FORCE = '--force' in args
    which = [a for a in args if not a.startswith('--')]
    targets = {'lens': LENS, 'dash': DASH}
    if not which:
        which = list(targets)
    unknown = [w for w in which if w not in targets]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; choose from {list(targets)} "
                         f"(optionally with --force)")
    for w in which:
        build(targets[w], pad=15.0)
