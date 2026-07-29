"""Generate insb_lens_inas.ipynb -- Pryor's Figs. 2 and 4 for an InSb lens in InAs.

This is `tools/make_inhom_nb.py` with the material system and the island shape changed. The
machinery is identical and is the machinery benchmarked against Pryor 1998; only the input
differs. Run:  python tools/make_insb_lens_nb.py  [--force]
"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FNAME = 'insb_lens_inas.ipynb'

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))


# ======================================================================================
# Framing
# ======================================================================================

md(r"""# InSb lens dots in InAs — Pryor's method, Figs. 2 and 4

This is `pryor_inhomogeneous.ipynb` with two things changed and nothing else: the material
system is **InSb in InAs** rather than InAs in GaAs, and the island is a **lens** rather than a
$\{101\}$ pyramid. The machinery underneath is the same machinery that was benchmarked against

> C. Pryor, *Phys. Rev. B* **57**, 7190 (1998); preprint
> [arXiv:cond-mat/9710304](https://arxiv.org/abs/cond-mat/9710304)

— real-space finite-element elasticity carrying position-dependent $C_{ijkl}$ (`elasticity_fd`,
Pryor's own method), the Bir–Pikus strain Hamiltonian and the eight-band $k\cdot p$ matrix
(`kp_pryor`), and first-order piezoelectricity (`piezoelectric`). What changes is the input to
it.

## Read this before reading any number below

**There is no reference calculation.** Pryor's paper is what made the InAs/GaAs notebook
checkable: his Table I fixes the parameters and his Figs. 2, 4 and 6 fix the answers, so every
number there could be put beside a published one. None of that exists here. The parameters come
from `materials_sb.py`, which assembles them from a database with two demonstrated defects in it
and tags four entries `UNVERIFIED`. Those are knobs, not data. The *machinery* is validated; the
*parameter set* is not, and nothing in this notebook can tell the difference. Vary them before
quoting anything.

## Three ways this system is not Pryor's

**1. The alignment is broken gap, and that removes one of his two Fig. 4 panels.** Unstrained,
the InSb valence edge sits **180 meV above** the InAs conduction edge. So this is not a type-I
dot with an electron and a hole sharing a box: holes are confined in the island, and electrons
are *expelled* from it because the island's conduction edge is far above the matrix's. The only
single-particle thing that could hold an electron is the pocket the island's compression digs in
the InAs wrapped around it — and that pocket does not bind, which is measured below rather than
assumed. So Fig. 4(a) has no counterpart as an energy-versus-size plot; what is swept in its
place is the box-independent binding criterion, which is a statement about the pocket rather
than about the computational box.

Fig. 4(b), the hole ladder, is the substance of this notebook — but it does **not** port
directly either. The broken gap puts a dense matrix-electron continuum right on top of the
valence-band edge, so the $\sigma$ that `pryor_fig4.py` targets returns box states rather than
holes, converged and certified and wrong. That is measured in the Fig. 4(b) section against a
dense diagonalization, and it is why `ingasb_dot.hole_ladder` scans $\sigma$ instead of seeding
it once.

**2. The misfit is comparable; the stiffness contrast is smaller.** InSb in InAs has
$\varepsilon_T = -6.50\%$ against $-6.69\%$ for InAs in GaAs, so linear elasticity is being
pushed just as hard. But the bulk-modulus ratio $K_{\rm dot}/K_{\rm matrix}$ is $0.824$ here
against $0.753$ there, so the *inhomogeneity* correction that motivated Pryor's method is about
a third smaller in relative terms. It is still why `elasticity_fd` is used and not
`strain_fourier`, for a reason specific to this system: the feature of interest on the
conduction side lives in the **matrix**, where a homogeneous solver has to borrow the dot's
stiffness or the matrix's and gets one of them wrong outright.

**3. The shape is smooth, which changes the grid problem completely.** The $\{101\}$ pyramid was
the hard case — its facets fall exactly on grid points, so the mask volume error swings by over
12% between spacings with no smooth trend in $h$, and the parity that is right at one spacing is
wrong at the next. A half-ellipsoid dome has no flat facet to land on a grid plane. The grid
section below measures the volume error rather than assuming this helps, but it does help, and
it is the one respect in which this calculation is easier than Pryor's.

It also removes a story. The near-degenerate excited pair in Pryor's Fig. 4(a) is the $C_4$
symmetry of a **square** island split by the piezoelectric field. A circular lens has no square
base to be $C_4$ about, so there is no such pair to look for and no such splitting to explain.

## Fig. 6 is deliberately not attempted

Pryor's Fig. 6 is a Hartree electron–hole pair. In a type-I dot both particles are bound as
single particles and the Hartree loop is a correction to energies that already exist. Here the
electron is not bound at all without the hole, so the same loop would be doing the *binding*,
and targeting an electron state with a folded-spectrum solve in a broken-gap spectrum — where
matrix-like conduction states and island-like valence states sit at the same energy — is not a
solved problem in this package. `ingasb_dot.eight_band_states` documents that hazard, and the
σ-placement failure it describes has already invalidated a session's worth of hole energies once
in the easier type-I case. So the exciton is left out rather than done badly. It is the largest
missing piece and is named again at the end.""")


# ======================================================================================
# Setup
# ======================================================================================

co("""import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

import qdsolver_core as qd
import strain_fourier as sf
import elasticity_fd as ef
import piezoelectric as pz
import kp_confined as kpc
import kp_pryor as kp
import eigensolvers as eig
import materials_sb as ms
import ingasb_dot as ig

# =====================================================================================
# PARAMETERS -- everything worth varying is here
# =====================================================================================
X_COMP  = 1.0          # InSb fraction of the In(x)Ga(1-x)Sb island. 1.0 IS InSb: every
                       # linearly interpolated key collapses to the binary and the gap
                       # bowing term x(1-x) vanishes. Lower it to walk back toward GaSb.
MATRIX  = 'InAs'

# --- SHAPE: circular lens (half-ellipsoid dome) --------------------------------------
RADIUS, HEIGHT = 10.0, 5.0      # nm. h/d = 1/4, the aspect ratio usually assumed for a
                                # self-assembled lens when no faceting is resolved.
shape = ig.lens(RADIUS, HEIGHT)

# --- GRIDS ---------------------------------------------------------------------------
# Two, for two different jobs. The band landscape is strain plus a 6x6 diagonalization per
# grid point, so it can afford to be fine. The size sweep is an eight-band folded-spectrum
# solve at 8 unknowns per point and cannot.
H_FIELD, PAD_FIELD = 0.5, 6.0
H_SWEEP, PAD_SWEEP = 0.75, 5.0
K_SWEEP = 4                     # LEVELS kept per size (each already a Kramers pair), which is
                                # Pryor's own cut-off: "only the first four states"

# The hole solve scans sigma rather than seeding it once -- see the Fig. 4(b) section for the
# measurement that forced this. Cost is N_SIGMA folded solves per island size, so these two
# are the dial between runtime and the risk of missing the top of the ladder.
N_SIGMA = 5                     # sigma values between the k=0 valence top and the InAs E_v
PROBE   = 24                    # states requested at each sigma. Well above the ~8 needed for
                                # four levels, because PROBE sets how WIDE each sigma's energy
                                # window is: at 10 the windows left uncovered gaps and two real
                                # levels fell into one. Whether 24 is enough on your grids is
                                # not assumed -- the coverage check reports it per size.

RADII  = (6.0, 8.0, 10.0, 12.0)           # base radius, nm
ASPECT = HEIGHT / (2 * RADIUS)            # h/d, held fixed so the sweep is size, not shape
lens_of = lambda r: ig.lens(r, 2 * r * ASPECT)

USE_PIEZO = True
CACHE = 'insb_lens_levels.json'           # the sweep checkpoints here after every size

dot, matrix = ms.ingasb(X_COMP), ms.SB_MATERIALS[MATRIX]
eps_star = ms.misfit(dot, matrix)
C_dot, C_mat = ms.elastic(dot), ms.elastic(matrix)
K_i, mu_i = ef.voigt_moduli(*C_dot)
K_m, mu_m = ef.voigt_moduli(*C_mat)

print(f"dot = {dot['name']} in {MATRIX};  {shape['label']}, volume {shape['volume']:.0f} nm^3")
print(f"misfit  eps_T = {eps_star*100:+.2f}%   (Pryor's InAs/GaAs: -6.69%)")
print(f"Voigt moduli (GPa):  {dot['name']} K={K_i:.1f} mu={mu_i:.1f} | "
      f"{MATRIX} K={K_m:.1f} mu={mu_m:.1f}")
print(f"stiffness contrast K_dot/K_matrix = {K_i/K_m:.3f}   (Pryor's InAs/GaAs: 0.753)")""")


# ======================================================================================
# Materials
# ======================================================================================

md(r"""---
## The parameter set, and what is wrong with it

Printed in full rather than hidden, because with no reference calculation this is the weakest
link in the notebook. Everything comes from `database.py` in the sibling `aestimo` checkout,
which is a verifiable provenance chain — the values were transcribed from that file, not
recalled — and that file attributes its III–V entries to Vurgaftman, Meyer and Ram-Mohan,
*J. Appl. Phys.* **89**, 5815 (2001).

Two defects in the source are corrected here, neither silently, and `audit()` re-derives both:

1. **The elastic constants are in two different units.** InAs is stored as $C_{11} = 8.329$
   (10¹¹ dyne/cm²) while InSb is $68.47$ (GPa). aestimo scales both identically, so it reads
   InSb **10× too stiff**. Since the whole point of `elasticity_fd` is the contrast between the
   two, this one would not be a small error here.
2. **GaSb's $a_v$ has the wrong sign** relative to InAs and InSb. That defect does not touch this
   notebook — at $x = 1$ the island is InSb and GaSb drops out of every interpolation — but the
   switch is left visible because lowering `X_COMP` brings it straight back in.

Note also that the masses are **derived** from the Kane expression rather than tabulated:
`database.py` lists InAs $m_e = 0.4$, an order of magnitude off. The derived InSb mass, 0.010
against an accepted 0.0135, is the expected weak point — at a 0.174 eV gap the expression is
very sensitive to $E_p$ and $F$ — and it feeds the pocket-binding criterion below.""")

co("ms.audit()")

md(r"""### Where the band edges actually sit

The last column decides the character of the whole structure. The $x = 1$ row is this notebook;
the others are there to show that the broken gap is not an artefact of sitting at the end of the
composition range.""")

co("""ms.alignment_table(compositions=(0.0, 0.5, 1.0), matrix=MATRIX)

print(f"\\nunstrained, on the zero used throughout (the unstrained {MATRIX} valence edge):")
print(f"  {MATRIX:>5}: E_v = {0.0:+.3f}   E_c = {matrix['Eg']:+.3f} eV")
Ev0 = dot['VBO'] - matrix['VBO']
print(f"  {dot['name']:>5}: E_v = {Ev0:+.3f}   E_c = {Ev0 + dot['Eg']:+.3f} eV")
print(f"\\nE_v(InSb) - E_c(InAs) = {Ev0 - matrix['Eg']:+.3f} eV  ->  BROKEN GAP")
print(f"electron mass (Kane-derived): {MATRIX} {ms.electron_mass(matrix):.4f}, "
      f"{dot['name']} {ms.electron_mass(dot):.4f} m0")""")


# ======================================================================================
# Elasticity validation
# ======================================================================================

md(r"""---
## Validation: the test a homogeneous solver cannot pass

Carried over from `pryor_inhomogeneous.ipynb` unchanged in structure, and re-run with *these*
elastic constants — the check is only worth anything at the contrast actually in use. An
isotropic surrogate is used so that the misfitting-sphere formula

$$\mathrm{Tr}\,\varepsilon = -3\,\varepsilon_T\,\frac{4\mu_m}{3K_i + 4\mu_m}$$

is exact. It involves the **inclusion's** bulk modulus and the **matrix's** shear modulus, so a
homogeneous solver — which must take both from one material — cannot reach the true value for
any choice of constants. The homogeneous case must reproduce what `strain_fourier` already
validates against; the inhomogeneous case is the new capability, and the point is that its
answer lies *outside* the interval a homogeneous solver can reach.

The interval is narrower here than in the InAs/GaAs case, because the stiffness contrast is
smaller. How much narrower is printed rather than asserted.""")

co("""h = 0.5
nu_m, nu_d = qd.voigt_poisson_ratio(*C_mat), qd.voigt_poisson_ratio(*C_dot)
Ci_m = sf.isotropic_constants(nu_m)
Ci_d = sf.isotropic_constants(nu_d, C11=Ci_m[0] * (C_dot[0] / C_mat[0]))
Ki, mui_ = ef.voigt_moduli(*Ci_d)
Km, Mm = ef.voigt_moduli(*Ci_m)

cc = qd.centered_axis(96, h)
Xs, Ys, Zs = np.meshgrid(cc, cc, cc, indexing='ij')
sph = qd.sphere_mask(Xs, Ys, Zs, 6.0)
core = qd.sphere_mask(Xs, Ys, Zs, 3.6)      # sample away from the staircased surface

print(f"{'case':34} {'analytic':>11} {'numeric':>11} {'rel err':>9}")
for label, Cd, Kx in ((f'homogeneous   (K_i = K_{MATRIX})', Ci_m, Km),
                      (f"INHOMOGENEOUS (K_i = K_{dot['name']})", Ci_d, Ki)):
    e = ef.solve_strain_fd(sph, eps_star, Cd, Ci_m, h, tol=1e-11)
    g = e.at(core)
    num = g['exx'] + g['eyy'] + g['ezz']
    exact = ef.inhomogeneous_sphere_trace(eps_star, Kx, Mm)
    print(f"{label:34} {exact:>11.6f} {num:>11.6f} {abs(num/exact-1):>9.2e}")

lo = ef.inhomogeneous_sphere_trace(eps_star, Ki, mui_)
hi = ef.inhomogeneous_sphere_trace(eps_star, Km, Mm)
true = ef.inhomogeneous_sphere_trace(eps_star, Ki, Mm)
print(f"\\nany homogeneous solve is confined to [{min(lo,hi):+.6f}, {max(lo,hi):+.6f}]")
print(f"the true inhomogeneous value {true:+.6f} is outside it, by "
      f"{min(abs(true-lo), abs(true-hi))/abs(true)*100:.1f}% of itself")
del Xs, Ys, Zs, sph, core""")


# ======================================================================================
# Geometry and grid
# ======================================================================================

md(r"""---
## Geometry and the grid

The pyramid's facets were the hard case: they are flat planes at 45°, they fall exactly on grid
points, and whether the samples land just inside or just outside them swings the mask volume by
over 12% with **no smooth trend in $h$**. `pryor_inhomogeneous.ipynb` had to tabulate that and
pick a spacing and a parity by measurement.

A half-ellipsoid dome has no flat facet to land on. The claim is that this makes the volume
error small and monotone in $h$, which is exactly the kind of claim that should be measured
rather than believed — so it is, below. `qd.island_grid` samples at cell centres in all three
directions and puts the island base at $z = 0$; `ig.build` asserts both the volume error and the
mirror symmetry and raises rather than warns.

Watch the **z-cells** column as well. These islands are flat, so $z$ is the dominant confinement
direction and it is the one a cubic grid resolves worst. That column, not the volume error, is
what limits how coarse the sweep grid can be.""")

co("""print(f"{'h (nm)':>7} {'grid':>16} {'points':>10} {'8-band unknowns':>16} "
      f"{'z-cells':>8} {'vol err':>9} {'asym':>6}")
for hh in (1.0, 0.75, 0.5, 0.4, 0.25):
    cxg, cyg, czg, Xg, Yg, Zg = qd.island_grid(shape['extent'], hh, PAD_FIELD)
    m = shape['mask'](Xg, Yg, Zg)
    zc = int(m[len(cxg)//2, len(cyg)//2, :].sum())
    asym = max(qd.mirror_asymmetry(m, 0), qd.mirror_asymmetry(m, 1))
    print(f"{hh:>7.2f} {str(m.shape):>16} {m.size:>10,} {8*m.size:>16,} {zc:>8} "
          f"{qd.mask_volume_error(m, hh, shape['volume']):>+9.4f} {asym:>6}")
    del Xg, Yg, Zg, m
print("\\nCompare the {101} pyramid in pryor_inhomogeneous.ipynb: -14.5% / -0.16% / -7.4% at")
print("h = 0.5 / 0.4 / 0.25, i.e. no trend at all. A smooth shape is the easy case.")""")


# ======================================================================================
# Fig. 2 analogue
# ======================================================================================

md(r"""---
# Fig. 2 — band structure from the local value of the strain

> **FIG. 2.** Band structure based on the local value of the strain. (a) Bands along the 001
> direction, through the center of the island. (b) Bands Along the 100 direction, through the
> base of the island.

The same construction: the $k = 0$ eigenvalues of the strain Hamiltonian, which at $k=0$
factorizes into a closed-form conduction band plus a $6\times6$ valence block, because the
interband terms $u, v$ carry a derivative and vanish. `kp_pryor.local_band_edges` diagonalizes
it, and `ingasb_dot.valence_edge` wires this material system into it.

**What to look for is not what Pryor's figure shows.** His is a type-I well: the conduction band
drops inside the island and the valence band rises, both carriers into the same box. Here:

* inside the island the conduction edge shoots **up**, far above the matrix — the electron is
  expelled;
* the valence edge inside rises above *everything*, including the InAs conduction edge — the
  broken gap, now several hundred meV wide because compression drives it further;
* just **outside** the island the conduction edge dips **below** the far-field InAs value. That
  dip is the electron pocket, and it exists only because the island's compression puts the
  surrounding matrix into tension. A model with uniform strain inside the dot and none outside
  would not have it at all. It is the reason this notebook needs a real elasticity solve rather
  than an eigenstrain estimate.""")

co("""env = ig.build(shape, X_COMP, h=H_FIELD, pad=PAD_FIELD, use_piezo=USE_PIEZO,
               matrix=MATRIX)""")

co("""edges = ig.valence_edge(env)
cx, cy, cz, m = env['cx'], env['cy'], env['cz'], env['mask']
ix, iy = len(cx) // 2, len(cy) // 2

idx = np.where(m[ix, iy, :])[0]
i_base = int(idx[0])                       # lowest layer of island cells
cut001 = (ix, iy, slice(None))             # [001] through the centre column
cut100 = (slice(None), iy, i_base)         # [100] through the base layer

print(f"island spans z = {cz[idx[0]]:.2f} .. {cz[idx[-1]]:.2f} nm on this grid "
      f"({len(idx)} cells)")
print(f"Kramers degeneracy residual in the 6x6 valence block: {edges['kramers']:.2e} eV")
print("  (nothing in the code enforces that degeneracy -- it is a free correctness check)")""")

co("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
panels = [(cut001, cz, 'distance along 001  (nm)',
           '(a) [001], through the centre of the island'),
          (cut100, cx, 'distance along 100  (nm)',
           f'(b) [100], through the base of the island (z = {cz[i_base]:.2f} nm)')]

for ax, (cut, r, xlabel, title) in zip(axes, panels):
    ax.plot(r, edges['cb'][cut], 'k-', lw=1.4, label='$E_c$')
    ax.plot(r, edges['v1'][cut], '-', color='#b03030', lw=1.4, label='$E_v$ (top)')
    ax.plot(r, edges['v2'][cut], '-', color='#b03030', lw=0.8, alpha=0.55)
    ax.plot(r, edges['v3'][cut], '-', color='#b03030', lw=0.8, alpha=0.55)
    ax.axhline(env['Ec_far'], color='0.55', ls='--', lw=0.9)
    ax.axhline(env['Ev_far'], color='0.55', ls=':', lw=0.9)
    for e in np.flatnonzero(np.diff(m[cut].astype(int))):
        ax.axvline((r[e] + r[e+1]) / 2, color='0.85', lw=0.8, zorder=0)
    ax.set(xlabel=xlabel)
    ax.set_title(title, fontsize=9)

axes[0].set_ylabel('E  (eV)')
axes[0].legend(fontsize=8, loc='upper left')
axes[1].text(0.99, 0.03, f"dashed: unstrained {MATRIX} $E_c$\\ndotted: unstrained {MATRIX} $E_v$",
             transform=axes[1].transAxes, ha='right', fontsize=7, color='0.4')
fig.suptitle(f"Band structure from the local value of the strain — {shape['label']}, "
             f"{dot['name']} in {MATRIX}, inhomogeneous elasticity", fontsize=10)
fig.tight_layout()
plt.show()""")

co("fig = ig.plot_maps(env)\nplt.show()")

md(r"""### The numbers this figure carries

Pryor attached two quantities to his Fig. 2 — the conduction well *"0.4 eV deep at the base of
the island, tapering to 0.27 eV at the tip"* — and those were targets, because he published
them. Nothing here is a target. What follows is the same set of quantities measured for this
system, with no published value to sit beside them, and they should be read as *what this
parameter set gives* rather than as a result.

Three of them are worth the space anyway:

* **the electron pocket**, deepest just outside the island where the tension peaks. It is the
  only single-particle attraction the electron has, and the next section shows it is not enough;
* **the hole well**, which is large — the compression drives the InSb valence edge up hard,
  because $a_v$ is positive in this convention and $\mathrm{Tr}\,\varepsilon$ is strongly
  negative. This is the confinement the eight-band ladder below resolves;
* **the strained broken-gap overlap**, which is not the unstrained 180 meV. Strain moves both
  edges by a lot at −6.5% misfit, and the sign of that motion is what makes the system more
  broken-gap rather than less.""")

co("""hw = ig.hole_well(env)
Ec_out = env['Ec'][~m]
pocket = env['Ec_far'] - Ec_out.min()

print(f"{'energy zero = unstrained ' + MATRIX + ' valence edge':<52}\\n")
print(f"{'conduction edge, mean inside the island':<52} {env['Ec'][m].mean():>8.3f} eV")
print(f"{'conduction edge, far field (= Eg of ' + MATRIX + ')':<52} {env['Ec_far']:>8.3f} eV")
print(f"{'conduction edge, deepest point in the matrix':<52} {Ec_out.min():>8.3f} eV")
print(f"{'  -> electron pocket depth':<52} {pocket*1e3:>8.1f} meV\\n")
print(f"{'valence edge, highest inside the island':<52} {hw['v_in']:>8.3f} eV")
print(f"{'valence edge, highest anywhere in the matrix':<52} {hw['v_out']:>8.3f} eV")
print(f"{'  -> hole well depth vs unstrained ' + MATRIX + ' E_v':<52} {hw['depth']*1e3:>8.1f} meV")
print(f"{'  -> island E_v above the matrix maximum':<52} {hw['above_matrix']*1e3:>8.1f} meV")
print(f"{'     (positive: holes stay in the island)':<52}\\n")
print(f"{'broken-gap overlap, UNSTRAINED':<52} "
      f"{(dot['VBO']-matrix['VBO']-matrix['Eg'])*1e3:>8.1f} meV")
print(f"{'broken-gap overlap, STRAINED':<52} {hw['broken_gap']*1e3:>8.1f} meV")
print(f"{'     (positive: island E_v above matrix E_c)':<52}")

print(f"\\npiezoelectric potential: {env['phi'].min()*1e3:+.1f} .. {env['phi'].max()*1e3:+.1f} meV")
print(f"C4 antisymmetry of that potential: {pz.c4_antisymmetry_error(env['phi']):.1e}")
print("  phi(C4 r) = -phi(r) is fixed by the tensor structure of the zincblende response and")
print("  holds for ANY dot shape, so a nonzero value here is a bug -- in the grid, usually --")
print("  rather than physics. It is a free check, and it has caught one here before.")""")


# ======================================================================================
# Fig. 4 analogue
# ======================================================================================

md(r"""---
# Fig. 4 — level energies as a function of island size

> **FIG. 4.** Bound state energies as a function of island size. [...] (a) Conduction band.
> (b) Valence band.

The lens is scaled homothetically — height tracks radius, so $h/d$ is fixed and the sweep is
genuinely size rather than shape. As in Pryor's figure, and for the same reason, the grid
spacing is held **fixed across the sweep** rather than refined with the island: refining it
would make the discretization error vary along the $x$-axis, and a trend contaminated by a
changing numerical error is not the trend.

Four levels per band is Pryor's own cut-off ("only the first four states"), so `K_SWEEP = 4`.
Each of those is a Kramers pair — time reversal makes every eigenvalue here exactly twofold
degenerate — and `hole_ladder` folds the partner in rather than reporting it as a second level.

## Panel (a) does not exist here, and that is the result

Pryor's conduction panel plots bound-state energies against size. That presupposes bound states.
In this system the electron has no band offset to bind it — it is expelled from the island — and
the only candidate is the strain pocket. A finite-box solve will always return *something*, and
the two sections below are the two independent ways of showing that what it returns is the box:

1. **the apparent binding drifts with the box** and never settles, which is what an unbound
   state looks like when it is quantized by walls;
2. **the box-independent criterion fails.** A finite spherical well of depth $V_0$ and radius $R$
   holds at least one bound state only if $V_0R^2 > \pi^2\hbar^2/8m$. Taking $R$ from the volume
   of matrix lying below the far-field conduction edge gives a verdict that does not mention the
   box — and it is a *generous* verdict, because the real pocket is a thin shell wrapped around a
   large repulsive barrier and a shell binds worse than a compact sphere of equal volume. If the
   generous version fails, the real one does too.

So what is swept in place of panel (a) is $V_0R^2$ against that threshold.""")

md(r"""### First: what the sweep will cost

Printed before it runs, and worth reading, because **this is a multi-hour cell**. The hole solve
scans $\sigma$ (see below for why it has to), so the cost is `N_SIGMA` folded-spectrum solves per
island size, not one. The anchor measurement: a 2 560-unknown problem took 62 s for a 5-point
scan. The grids below are 40–130× that, and iteration counts do not fall with size, so budget
hours rather than minutes and let the checkpoint file do its job — `hole_size_sweep` writes
`insb_lens_levels.json` after **every** size, so an interrupted run resumes where it stopped.

Trim `RADII` first if that is too much; it is the linear knob. Dropping `N_SIGMA` is the
tempting second one and is the riskier of the two, because a scan too coarse to bracket the
ladder returns an empty result that looks like a physical statement.

The z-cells column is the other thing to check. At $h/d = 1/4$ the smallest island in `RADII` is
the thinnest object in the sweep, and a level quoted from three or four cells of vertical
resolution is indicative, not converged.""")

co("""print(f"{'radius':>7} {'height':>7} {'grid':>16} {'points':>9} {'unknowns':>10} "
      f"{'z-cells':>8} {'vol err':>9}")
for R in RADII:
    s_ = lens_of(R)
    cxg, cyg, czg, Xg, Yg, Zg = qd.island_grid(s_['extent'], H_SWEEP, PAD_SWEEP)
    mm = s_['mask'](Xg, Yg, Zg)
    zc = int(mm[len(cxg)//2, len(cyg)//2, :].sum())
    print(f"{R:>7.1f} {2*R*ASPECT:>7.1f} {str(mm.shape):>16} {mm.size:>9,} "
          f"{8*mm.size:>10,} {zc:>8} "
          f"{qd.mask_volume_error(mm, H_SWEEP, s_['volume']):>+9.4f}")
    del Xg, Yg, Zg, mm""")

md(r"""### (a) The electron: the box, and then the criterion

The ladder below deliberately runs at a **coarser grid** than the band landscape and a much
larger box. This is a convergence test in box size, not in $h$; the single-band solver
factorizes the Hamiltonian completely (sparse LU) and 3D fill-in grows fast, so the points have
to be spent on the box.

Read the "apparent binding" column with its sign. It comes out **negative** — the ground state
sits *above* the far-field InAs conduction edge, not below it — and it climbs toward zero from
below as the box grows, without crossing. That is what an unbound state looks like when walls
quantize it: the energy is box zero-point energy, falling as $1/L^2$, so the state approaches the
barrier edge from above and never converges beneath it. A bound state would do the opposite,
settling at a fixed positive binding once the box exceeded its decay length.""")

co("""H_BOX = 1.0
print(f"{'pad':>5} {'box (nm)':>18} {'points':>9} {'pocket':>9} {'E0':>9} "
      f"{'apparent binding':>18} {'in dot':>8}")
for pad_ in (6.0, 12.0, 18.0, 24.0):
    e_ = ig.build(shape, X_COMP, h=H_BOX, pad=pad_, use_piezo=USE_PIEZO, matrix=MATRIX,
                  verbose=False)
    s_ = ig.electron_states(e_, k=2, verbose=False)
    pk = e_['Ec_far'] - e_['Ec'][~e_['mask']].min()
    box = f"{len(e_['cx'])*H_BOX:.0f} x {len(e_['cy'])*H_BOX:.0f} x {len(e_['cz'])*H_BOX:.0f}"
    print(f"{pad_:>5.0f} {box:>18} {e_['mask'].size:>9,} {pk*1e3:>8.1f}m "
          f"{s_['E'][0]:>9.4f} {(e_['Ec_far']-s_['E'][0])*1e3:>17.1f}m "
          f"{s_['inside'][0]*100:>7.1f}%")
    del e_, s_
print(f"\\nA small 'in dot' fraction is CORRECT: the electron is expelled from the island and")
print(f"sits in the tensile {MATRIX} shell around it. A large fraction would mean the alignment")
print("had come out type-I, i.e. that something is wrong.")""")

co("_ = ig.pocket_metrics(env)")

md(r"""### (b) The hole: the eight-band ladder, and the $\sigma$ problem in front of it

This is the expensive part of the notebook and the part that carries the physics. No single-band
hole solve is offered anywhere in this package for this system, deliberately: the hole sits in a
narrow-gap material strained by several percent, where heavy and light hole are split by hundreds
of meV and strongly mixed, and a parabolic band would be meaningless. So the ladder is the full
eight-band Hamiltonian with the inhomogeneous FD strain and the piezoelectric potential — the
same machinery as `pryor_fig4.py`, driven through `ingasb_dot`.

#### The obvious way to do this is wrong, and it fails quietly

In Pryor's InAs/GaAs benchmark, `kp_pryor.hole_sigma` — the exact $k=0$ top of the local valence
band — is the right target for a folded-spectrum solve: the hole states sit a few tens of meV
below it, and asking for the eigenvalues nearest it returns them. `pryor_fig4.bound_states` does
exactly that, and it is correct there.

It is not correct here, and the failure does not announce itself. Measured on this system by
**dense diagonalization** of the full eight-band matrix on a small grid — so this is not an
artefact of an iterative solver:

| | |
|---|---|
| `hole_sigma` | 0.806 eV |
| spectrum *at* `hole_sigma` | levels ~1 meV apart, each ~1% localized in the island |
| island-localized states (40–84% inside) | −0.04 to +0.44 eV |
| gap between the two | **370–850 meV** |

The states at $\sigma$ are InAs electron box states. They are there because the alignment is
broken gap: the island's valence edge lies far above the matrix conduction edge, so the matrix
continuum reaches right up through the hole well. Two things push the ladder that far below the
band edge — a large confinement energy (small island, deep well, heavy-hole mass), and
hybridization with that continuum, which the island's valence states are degenerate with rather
than separated from.

Seeded at `hole_sigma`, a folded-spectrum solve ran to its 8000-iteration limit, returned a
residual of $6\times10^{-4}$ eV, and reported four states **0.6% localized**. Converged-looking,
certified as genuine eigenpairs, and not hole states. Seeded at 0.44 eV instead, the *same solver
on the same matrix* converged in 652 iterations and 10 seconds to a residual of
$1.5\times10^{-7}$ eV and returned exactly the localized states the dense diagonalization found.

So neither the solver nor the conditioning was ever the problem. $\sigma$ was.

#### What `hole_ladder` does instead

It scans $\sigma$ down from the $k=0$ valence-band top toward the matrix valence edge, solves for
`PROBE` states at each, collapses each solve to physical levels, keeps the levels at least
`HOLE_LOC_MIN` localized in the island, then merges across $\sigma$ and takes the top $k$. Every
level reported was therefore found by a solve seeded near its own energy.

#### The check that matters is coverage, not convergence

Each solve returns only `PROBE` eigenvalues around its own $\sigma$, so it sees a finite energy
*window*. If two neighbouring windows do not overlap, everything between them is invisible to the
scan — and a level can be sitting there.

This is not hypothetical, and it is the reason `hole_ladder` computes coverage rather than
leaving it to the reader. At $r = 6$ nm with `N_SIGMA = 5` and `PROBE = 10`, **the two topmost
levels fell into the gap** between the $\sigma = 0.48$ and $\sigma = 0.32$ windows and were
simply absent from the ladder. Every other check still passed: Kramers degeneracy resolved to
$2\times10^{-11}$ meV, per-level residuals at $2\times10^{-7}$ eV, and `sigma_hits` nowhere near
the bottom of the scan. A clean-looking ladder, missing its top two rungs.

So the scan reports its window spans and warns on any uncovered gap. **Read that warning before
reading the levels.** The fix is a larger `PROBE` (wider windows) rather than more $\sigma$
points, and it is cheap — the extra eigenvalues come from the same factorisation-free iteration.

Two more consequences worth stating plainly. The scan costs `N_SIGMA` solves per island size
rather than one, and the $\sigma$ values that land in the continuum will not converge — expected,
not a fault, so residuals are tracked **per kept level** rather than as a maximum over the scan.
`sigma_hits` is still recorded and still worth a glance: if the top level came from the lowest
$\sigma$ in the scan, the scan did not bracket the ladder from below either.

**Kramers pairing remains a free correctness check.** Time reversal makes every eigenvalue exactly
twofold degenerate, so one physical level is a *pair*; the largest within-pair splitting has to
sit at the residual level, and if it does not, the level *count* is unreliable, not just the
splitting.

The sweep checkpoints to `insb_lens_levels.json` after every size, so an interrupted run resumes
rather than restarting. Delete that file, or pass `recompute=True`, to force a fresh run.""")

md(r"""#### The evidence, reproduced at the smallest island

The cheapest place to show the effect. The naive solve is *expected* to hit `maxiter` — that is
the symptom, not a misconfiguration — so it is capped low here to keep the cell affordable.
Compare its localization column against the scan that follows.""")

co("""R_DIAG = RADII[0]
env_d = ig.build(lens_of(R_DIAG), X_COMP, h=H_SWEEP, pad=PAD_SWEEP, use_piezo=USE_PIEZO,
                 matrix=MATRIX, verbose=False)
m_d, n_d = env_d['mask'], env_d['mask'].size
top_d = ig.valence_edge_top(env_d)

ops = kpc.GridOperators(m_d.shape, H_SWEEP, periodic=False)
fields = kp.material_fields(m_d, ms.kp_params(env_d['dot']), ms.kp_params(env_d['matrix']),
                            env_d['Ev'], env_d['Ec'], n_bands=8)
H_d = kp.confined_hamiltonian(ops, fields, n_bands=8, strain=env_d['strain'])

print(f"R = {R_DIAG} nm, {H_d.shape[0]:,} unknowns; k=0 valence top = {top_d:.4f} eV\\n")
print("NAIVE: folded spectrum seeded at the k=0 valence-band top, as the benchmark does --")
E_n, V_n, info_n = eig.solve_interior(H_d, k=4, sigma=top_d, tol=1e-7, maxiter=1500)
loc_n = np.array([float((np.abs(V_n[:, j].reshape(8, n_d))**2).sum(axis=0)
                        .reshape(m_d.shape)[m_d].sum() / (np.abs(V_n[:, j])**2).sum())
                  for j in range(V_n.shape[1])])
for j in np.argsort(-E_n):
    print(f"    E = {E_n[j]:.4f} eV, {loc_n[j]*100:5.1f}% inside the island")
print(f"  -> spread {np.ptp(E_n)*1e3:.1f} meV, max localization {loc_n.max()*100:.1f}%")
print("  A tight clump of barely-localized states is the signature: these are InAs box states.")
del H_d, V_n""")

co("""print("SCAN: the same matrix, sigma walked down through the well --")
lad_d = ig.hole_ladder(env_d, k=K_SWEEP, probe=PROBE, n_sigma=N_SIGMA)
print(f"\\n{'level':>6} {'E (eV)':>9} {'in island':>11} {'found at sigma':>15}")
for j, (e_, l_, s_) in enumerate(zip(lad_d['levels'], lad_d['level_loc'],
                                     lad_d['sigma_hits'])):
    print(f"{j:>6} {e_:>9.4f} {l_*100:>10.1f}% {s_:>15.3f}")
if len(lad_d['E']):
    print(f"\\nladder top is {(top_d - lad_d['E'][0])*1e3:.0f} meV below the k=0 valence edge")
    print(f"scan ran {lad_d['sigmas'].min():.3f} .. {lad_d['sigmas'].max():.3f} eV; "
          f"top level found at sigma = {lad_d['sigma_hits'][0]:.3f}")
    print("If that sigma is the LOWEST in the scan, the scan did not bracket the ladder.")
del env_d""")

md(r"""#### The sweep""")

co("""t0 = time.time()
results = ig.hole_size_sweep(lens_of, RADII, label='radius', x=X_COMP, cache=CACHE,
                             h=H_SWEEP, pad=PAD_SWEEP, k=K_SWEEP, use_piezo=USE_PIEZO,
                             probe=PROBE, n_sigma=N_SIGMA)
print(f"\\nsweep total {time.time()-t0:.0f}s")""")

co("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ig.plot_pocket_verdict(results, 'radius', ax=axes[0])
ig.plot_hole_ladder(results, 'radius', ax=axes[1])
axes[0].set_title('(a) Conduction band — no bound state to plot', fontsize=10)
axes[1].set_title('(b) Valence band — eight-band hole levels', fontsize=10)
fig.suptitle(f"Level energies vs island size — {dot['name']} lens in {MATRIX}, "
             f"$h/d$ = {ASPECT:.2f}, $h$ = {H_SWEEP} nm — cf. Pryor 1998 Fig. 4", fontsize=10)
fig.tight_layout()
plt.show()""")

co("""print(f"{'R':>5} {'V0R^2':>7} {'binds':>6} {'well':>7} {'gap':>7} {'n_lvl':>6} "
      f"{'E0':>8} {'E0-E1':>8} {'E1-E2':>8} {'loc0':>7} {'below edge':>11} "
      f"{'loc/probe':>11} {'kramers':>9} {'resid':>9} {'gaps':>6}")
nan = float('nan')
for r in results:
    vb = r['vb']
    lvl, lloc = np.array(vb['levels']), np.array(vb['level_loc'])
    keep = ig.bound_hole_levels([r])[1][0]
    sp = np.diff(lvl) * 1e3
    print(f"{r['radius']:>5.1f} {r['pocket_VR2']:>7.2f} {str(r['pocket_binds']):>6} "
          f"{r['hole_well']*1e3:>6.0f}m {r['broken_gap']*1e3:>6.0f}m {len(keep):>6} "
          f"{lvl[0] if len(lvl) else nan:>8.4f} "
          f"{sp[0] if len(sp) else nan:>7.1f}m {sp[1] if len(sp) > 1 else nan:>7.1f}m "
          f"{lloc[0]*100 if len(lloc) else nan:>6.1f}% "
          f"{(vb['valence_top']-lvl[0])*1e3 if len(lvl) else nan:>10.0f}m "
          f"{vb['n_localized']:>5}/{vb['n_probed']:<5} {vb['kramers_split']*1e3:>8.1e} "
          f"{vb['residual']:>9.1e} {len(vb['gaps']):>6}")
print(f"\\nbinding threshold V0R^2 = {ig.POCKET_CRITERION:.2f} eV nm^2 (conduction band)")
print("'well' = island E_v above unstrained InAs E_v; 'gap' = island E_v above InAs E_c.")
print("'below edge' = how far the ladder top sits under the k=0 valence-band maximum. A large")
print("  value is confinement energy plus the continuum hybridization discussed above -- it is")
print("  NOT a sign the solve went wrong, but it IS why sigma has to be scanned.")
print("'loc/probe' = island-localized states found / states probed across the scan. A low ratio")
print("  is normal here; a ZERO means the scan missed the ladder entirely.")
print("Kramers splitting must sit at the residual level; if it does not, the LEVEL COUNT is")
print("  unreliable, not just the splitting.")
print("'gaps' = uncovered energy gaps between neighbouring sigma windows. ANY nonzero value")
print("  means a level may be missing from that row's ladder with no other symptom. Raise")
print("  PROBE and re-run that size (delete its entry from the cache, or pass recompute=True).")""")


# ======================================================================================
# Closing
# ======================================================================================

md(r"""---
## What this notebook does and does not establish

**Does.** The strain field, the band-edge landscape and the eight-band hole ladder for an InSb
lens in InAs, on machinery that is validated independently of this material system: the
elasticity against the inhomogeneous misfitting-sphere result that no homogeneous solver can
reproduce (re-run above at *these* elastic constants, not just at Pryor's), the Bir–Pikus terms
against over-determined identities, the $k=0$ valence block against a Kramers degeneracy that
nothing in the code enforces, and the eight-band levels against their own Kramers pairing.

**Does — methodologically.** The $\sigma$-placement result in the Fig. 4(b) section is not a
detail of this notebook, it is a property of broken-gap systems: the exact $k=0$ valence-band
maximum, which is the correct target in the type-I benchmark, sits inside a dense matrix
continuum here and returns box states with a clean residual. It was established against a dense
diagonalization rather than by comparing two iterative solves, so it does not rest on either one
being right. Anyone extending this package to another broken-gap system should expect the same
and scan rather than seed.

**Does — negatively, and this is the substantive result.** The electron is not bound by any
single-particle well. Two independent arguments, neither of which relies on the other: the
apparent binding drifts with box size and never settles, and the box-independent criterion
$V_0R^2 > \pi^2\hbar^2/8m$ fails by a wide margin at every island size, on a *generous* reading
of the pocket geometry. Doubling the pocket depth would not fix it — a light electron cannot
localize in a thin shell however deep it is. So Pryor's Fig. 4(a) has no counterpart here, and
producing one from a finite-box solve would have been measuring the box.

**Does not.**

* **The parameter set is unbenchmarked and is the weakest link.** There is no published
  calculation for this system to check it against — this is the one thing the InAs/GaAs notebook
  had and this one does not. Four entries are tagged `UNVERIFIED` in `materials_sb.py`. The
  $[111]$ shear deformation potential $d$ is the worst of them for *this* notebook: it is not in
  the source database at all, and it enters the Bir–Pikus $S$ term, which is specifically a
  valence-mixing term, so it acts directly on the hole ladder. Vary it before quoting a level
  spacing.
* **The derived InSb electron mass is soft**, 0.010 against an accepted 0.0135 — at a 0.174 eV
  gap the Kane expression is very sensitive to $E_p$ and $F$. It enters the pocket criterion, so
  it enters the negative result above. The margin there is a factor of ~5, and a 35% mass error
  moves the threshold by 35%, so the conclusion survives; a marginal one would not have.
* **No exciton, and here that is the main gap, not a refinement.** The single-particle pocket
  does not bind an electron, so what binds it is the Coulomb attraction to the hole — a leading
  contribution. Pryor's Fig. 6 is exactly that calculation for the type-I case, and it is not
  ported here for the reason given at the top. Until it is, this notebook describes the
  *landscape* the exciton would live in, not the exciton.
* **No wetting layer.** Real islands of this kind grow on one, and it will bind states of its
  own — including, plausibly, the electron states this system otherwise lacks.
* **The hole ladder is the least settled thing here, and it is unbenchmarked twice over** — no
  published calculation for the material system, and no second method for the $\sigma$ scan. The
  scan is verified against a dense diagonalization at one small size only, because a dense solve
  is $O(N^3)$ and the sweep grids are far past where that is possible. What is checked at every
  size is weaker but not nothing: Kramers degeneracy resolved to the residual, per-level
  residuals from a solve seeded near that level, and `sigma_hits` showing the scan bracketed the
  ladder rather than running off its bottom end.
* **The levels sit far below the $k=0$ valence-band maximum** — several hundred meV. Part of that
  is ordinary confinement energy in a small, flat island with a heavy-hole mass; part is
  hybridization with the matrix continuum the levels are degenerate with. This notebook does not
  separate the two, and until it does, "confinement energy" should not be read off the gap
  between the ladder top and the band edge.
* **Weakly bound levels are not converged in box size.** The hole levels here are deep, so they
  are unaffected; the statement is kept because it is the reason the electron section had to
  use a box ladder rather than a single large box.
* **First-order piezoelectricity only**, and the second-order response is known to be comparable
  and often opposed in strained III–V dots (Bester, Zunger and co-workers, *Phys. Rev. B* **74**,
  081305(R) and *Phys. Rev. Lett.* **96**, 187602, both 2006). Pryor predates that work; first
  order is used here for comparability with the benchmark.
* **Linear elasticity at −6.5% misfit** is being pushed exactly as hard as it was in the
  benchmark, and **continuum** elasticity cannot see the zincblende lattice's own symmetry —
  only an atomistic relaxation recovers that (Pryor, Kim, Wang, Williamson & Zunger,
  *J. Appl. Phys.* **83**, 2548 (1998)).""")


nb['cells'] = c
nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
nb.metadata.language_info = dict(name='python', version='3.11')


def write(force=False):
    path = os.path.join(ROOT, FNAME)
    # Refuse to silently destroy executed results: the eight-band sweep in here is the long run,
    # and regenerating to fix a typo would throw it away. Same guard as make_ingasb_nb.py.
    if os.path.exists(path) and not force:
        existing = nbf.read(path, as_version=4)
        n_out = sum(1 for cell in existing.cells
                    if cell.cell_type == 'code' and cell.get('outputs'))
        if n_out:
            print(f"REFUSING to overwrite {FNAME}: it has {n_out} executed cells with output. "
                  f"Re-run with --force if you really mean to discard them.")
            return
    nbf.write(nb, path)
    print(f"wrote {FNAME} with {len(c)} cells "
          f"({sum(1 for x in c if x.cell_type == 'code')} code)")


if __name__ == '__main__':
    import sys
    write(force='--force' in sys.argv[1:])
