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
HEIGHT   = 4.0         # nm, apex height above the base plane

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
HEIGHT   = 3.0         # nm, height of the flat top above the base plane
ANGLE    = 25.0        # deg, side-facet contact angle

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
"rectangular base, sloping facets, flat top". The facet angle is *not* fixed by that paper, so
`ANGLE` is yours to set.""",
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
H_GRID   = 0.5         # nm, grid spacing
PAD      = {pad}        # nm, matrix around the island. The electron pocket here is SHALLOW,
                       # so this has to be large; a box-convergence check is run below.
USE_PIEZO = True
RUN_EIGHT_BAND = False # opt-in: minutes, not seconds. See the last section.

# --- MATERIAL CONVENTIONS -------------------------------------------------------
# See materials_sb.py: the source database has a sign inconsistency in GaSb's a_v that
# directly sets the hole well depth. Switch and re-run to see how much it matters.
ms.GASB_AV_CONVENTION  # currently: {{}}
print(f"GaSb a_v convention: {{ms.GASB_AV_CONVENTION!r}} -> a_v = "
      f"{{ms.SB_MATERIALS['GaSb']['a_v']:+.2f}} eV")
print(f"dot = {{ms.ingasb(X_COMP)['name']}}, shape = {{shape['label']}}")
print(f"island volume {{shape['volume']:.1f}} nm^3")"""


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

**What binds the electron is strain, not a band offset.** The island is compressed by the matrix
(misfit runs $-0.6\\%$ to $-6.5\\%$ across the composition range), and by reaction the InAs wrapped
around it is put into *tension*. Since $a_c < 0$, tension pulls the InAs conduction edge **down**,
so the island digs a shallow conduction-band pocket in the matrix around itself. That pocket is
the only thing holding the electron, and it is a genuinely inhomogeneous-strain effect — a model
with uniform strain inside the island and zero outside would not have it at all.

That is also why the electron and hole end up spatially separated, which is the defining feature
of a type-II dot and the reason these structures are interesting for long-lived carriers.

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
## The electron: a shallow pocket, and whether the box is deciding it

The pocket is tens of meV deep and the electron mass in InAs is small (~0.025 $m_0$), which is a
marginal combination for binding a state at all. A finite box adds its own confinement energy of
the same order, so **the box can manufacture or destroy the bound state**. That has to be measured
before any statement about binding means anything.""")

    co("""for pad_ in (PAD, PAD * 1.5, PAD * 2.0):
    e_ = ig.build(shape, X_COMP, h=H_GRID, pad=pad_, use_piezo=USE_PIEZO, verbose=False)
    s_ = ig.electron_states(e_, k=2, verbose=False)
    pocket = e_['Ec_far'] - e_['Ec'][~e_['mask']].min()
    print(f"pad {pad_:5.1f} nm  box {len(e_['cx'])*H_GRID:5.1f} x {len(e_['cy'])*H_GRID:5.1f} x "
          f"{len(e_['cz'])*H_GRID:5.1f} nm   pocket {pocket*1e3:6.1f} meV   "
          f"E0 {s_['E'][0]:.4f} eV   binding {(e_['Ec_far']-s_['E'][0])*1e3:+7.1f} meV   "
          f"{s_['inside'][0]*100:4.1f}% in dot")
    del e_, s_""")

    co("""st = ig.electron_states(env, k=4)
print(f"\\nbound states below the far-field InAs conduction edge: {st['n_bound']}")
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
        co("""V_TARGET = shape['volume']

def _dash_at_aspect(a):
    \"\"\"Dash of aspect ratio `a` holding the volume fixed at the reference island's.\"\"\"
    w = np.sqrt(LENGTH * WIDTH / a)
    l = w * a
    hgt = ig.dash_height_for_volume(l, w, V_TARGET, ANGLE)
    if hgt is None:
        raise ValueError(f"aspect {a}: width {w:.1f} nm too narrow to hold {V_TARGET:.0f} nm^3 "
                         f"at {ANGLE} deg facets")
    return ig.dash(l, w, hgt, ANGLE), X_COMP

print(f"target volume {V_TARGET:.1f} nm^3, facets {ANGLE} deg\\n")
rows_s = []
for a_ in (1.0, 2.0, 3.0, 4.0, 6.0):
    try:
        rows_s += ig.sweep(_dash_at_aspect, (a_,), 'aspect', h=H_GRID, pad=PAD,
                           use_piezo=USE_PIEZO)
    except ValueError as exc:
        print(f"  aspect = {a_}: skipped -- {exc}")""")
    else:
        co("""rows_s = ig.sweep(lambda r: (ig.lens(r, HEIGHT * r / RADIUS), X_COMP),
                  (6.0, 8.0, 10.0, 12.0, 14.0), 'radius',
                  h=H_GRID, pad=PAD, use_piezo=USE_PIEZO)""")

    co(f"""fig = ig.plot_sweep(rows_s, '{cfg['sweep_name']}')
fig.suptitle(f"size sweep — dot = {{ms.ingasb(X_COMP)['name']}}", fontsize=10, y=1.04)
plt.show()""")

    md("""---
## Eight-band states (opt-in)

Set `RUN_EIGHT_BAND = True` in the parameter cell to run this. It costs minutes rather than
seconds, and it is what the hole actually requires.

**Read this before believing the numbers.** In a broken-gap system "the states near the valence
edge" is not a clean idea: the island's valence edge lies *above* the matrix's conduction edge, so
at one and the same energy there are island-like valence states and matrix-like conduction states,
and they hybridize. A folded-spectrum solve returns the eigenvalues nearest $\\sigma$, and its
residual certifies only that they *are* eigenpairs — not that they are the ones wanted. That
failure mode has already cost this project a session's worth of invalidated numbers in the
InAs/GaAs benchmark, where the gap was not even broken.

So the localization fraction printed beside each state is not decoration; it is the only thing
separating a hole state from a matrix electron state at the same energy. Re-run with $\\sigma$
moved and confirm the spectrum is unchanged before quoting anything.""")

    co("""if RUN_EIGHT_BAND:
    t0 = time.time()
    vb = ig.eight_band_states(env, band='vb', k=4)
    print(f"\\n{time.time()-t0:.0f}s")
    print("re-run with sigma raised to confirm these are the intended states:")
    print("  ig.eight_band_states(env, band='vb', k=4)  after editing kp.hole_sigma's margin")
else:
    print("RUN_EIGHT_BAND is False -- skipped. Set it True in the parameter cell to run.")""")

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
* *No excitonic binding.* In a type-II structure the electron and hole are spatially separated and
  the Coulomb attraction is a leading, not a small, contribution to whether the electron is bound
  at all. The single-particle pocket studied here is only part of the story.""")

    nb['cells'] = c
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    path = os.path.join(ROOT, cfg['fname'])
    nbf.write(nb, path)
    print(f"wrote {cfg['fname']} with {len(c)} cells")


if __name__ == '__main__':
    build(LENS, pad=18.0)
    build(DASH, pad=18.0)
