# qd3Dsolver

A 3D quantum-dot solver — continuum elasticity, multiband **k·p**, piezoelectricity and
Schrödinger–Poisson — built and benchmarked against

> C. Pryor, *"Eight-band calculations of strained InAs/GaAs quantum dots compared with one-,
> four-, and six-band approximations"*, **Phys. Rev. B 57, 7190 (1998)**.
> Preprint: [arXiv:cond-mat/9710304](https://arxiv.org/abs/cond-mat/9710304).

That paper is the reference implementation this package is measured against: its Table I supplies
the material parameters, and its Figs. 2, 4, 5, 6 and 7 are the targets. Where a result here
disagrees with it, the discrepancy is quantified rather than absorbed.

---

## Quick start

Everything is plain NumPy/SciPy plus `matplotlib`, `scikit-image` (isosurfaces) and `nbformat`.
There is no build step and no package install — modules import each other by bare name, so
**run from the repository root**.

```bash
python materials.py                        # the whole parameter table + 4 consistency checks (instant)
python scripts/elasticity_validation.py    # the elasticity solvers, against analytic ground truth
python scripts/strain_validation.py        # strain tensor + Bir–Pikus terms
python scripts/shape_validation.py         # dot/dash geometry vs closed-form volumes (seconds)
python scripts/insb_band_edges_vs_size.py  # InSb/InAs band edges vs size: scale invariance (~100 min)
python scripts/insb_pryor_pistol_check.py  # ...and why they miss Pryor & Pistol 2005 (seconds)
python scripts/insb_box_convergence.py     # periodic-box convergence of the dot gap (~25 min)
python scripts/pryor_fig2.py               # reproduce Pryor's Fig. 2 (homogeneous elasticity)
python pryor_fig4.py                       # bound-state energies vs island size (Fig. 4)
python pryor_fig6.py                       # ground-state exciton wave functions (Fig. 6)
```

`pryor_fig4.py` is the one long run here (~1 h: an eight-band solve for four Kramers pairs in
each band at each of seven island sizes). It checkpoints to `fig4_levels.json` after every size,
so an interrupted sweep resumes instead of restarting; pass sizes on the command line
(`python pryor_fig4.py 10 12`) to do a subset.

Notebooks are the primary deliverables and assume the working directory is the repository root:

| Notebook | What it shows |
|---|---|
| `pryor_inhomogeneous.ipynb` | **Start here.** Pryor's own FD + conjugate-gradient elasticity, applied to his Figs. 2, 4 and 6. |
| `ingasb_lens.ipynb`, `ingasb_dash.ipynb` | InAs / In*ₓ*Ga₁₋ₓSb / InAs — a **broken-gap, type-II** system. Composition, size and shape are parameters at the top of each notebook. |
| `insb_lens_inas.ipynb` | `pryor_inhomogeneous.ipynb` re-run for an **InSb lens in InAs**: Pryor's Figs. 2 and 4 with the material system and shape changed. Fig. 4(a) has no counterpart — see below. |
| `pryor_fig2.ipynb` | Fig. 2 with the faster Fourier (homogeneous) strain solver. |
| `pryor_benchmark.ipynb` | Bound-state energies vs Pryor's Fig. 7. *See "Trust" below.* |
| `interband_strain_terms.ipynb` | The strain-dependent interband terms *u*, *v*. |
| `multiband_bulk_validation.ipynb` | Bulk k·p dispersion checks. *See "Trust" below.* |
| `strain_corrections.ipynb` | Write-up of the strain-model corrections (deliberately unexecuted). |
| `lens_InAs_GaAs.ipynb`, `pyramid_InAs_GaAs.ipynb` | Single-band shape studies. |
| `schrodinger_poisson_exciton.ipynb` | Single-band self-consistent exciton. |

---

## Layout

```
.                     library modules + notebooks (flat: this is the import surface)
scripts/              validation suites and one-off analyses
tools/                notebook generators (write their .ipynb to the repo root)
logs/                 saved validation output
archive/              superseded or known-wrong artifacts — see archive/README.md
```

The library is deliberately **flat rather than a package**. Every module imports its siblings by
bare name (`import qdsolver_core as qd`), and the notebooks put the repo root on `sys.path`;
folding them into a package would mean rewriting every import in every executed notebook. Scripts
in `scripts/` and `tools/` add the parent directory to `sys.path` themselves, so they can be run
from anywhere.

---

## Modules

### Geometry and single-band machinery
| Module | Contents |
|---|---|
| `qdsolver_core.py` | Dot-shape masks, grid construction and **geometry guards**, the BenDaniel–Duke single-band Hamiltonian, `solve_states`. |
| `poisson_solver.py` | 3D Poisson on the same discretization. |
| `schrodinger_poisson.py` | Single-band self-consistent electron–hole pair (Hartree, no self-interaction). |

**Always build grids with `centered_axis`, and assert `mirror_asymmetry == 0` and
`mask_volume_error` before trusting a symmetry-sensitive result.** `np.arange` is not symmetric
about zero at the 1e-14 level, and the {101} facets of `pyramid_mask` fall exactly on grid points,
so an asymmetric axis flips voxels between +x and −x. That produced a spurious C4 symmetry
violation here once, and the symptom appeared far downstream of the cause.

Grid *parity* matters just as much and is not monotone in `h`: a mask that is exact to 0.16% at
one spacing is 12% oversized at the next. Measure it, never assume finer is better.

**Shapes.** `sphere_mask`, `ellipsoid_mask`, `lens_mask`, `pyramid_mask` ({101} facets, Pryor's
island), and `dash_mask` — an elongated **truncated rectangular pyramid** (rectangular base,
sloping side facets at a settable contact angle, flat top) after Tersoff & Tromp,
*Phys. Rev. Lett.* **70**, 2782 (1993). Each has a closed-form volume for `mask_volume_error`.

Build grids for the flat faceted shapes with **`island_grid`**, which samples at cell *centres* in
all three directions. This is not cosmetic: `dash_mask` includes its boundary, so a grid with
points lying *on* the bounding planes counts a 1 nm tall island as 1.5 nm thick at h = 0.5 —
**+53%**, and refining h only halves it. Cell-centred sampling gives −9% / −4% / −2% at
h = 0.5 / 0.25 / 0.125, i.e. genuine first-order facet staircasing that converges. Note this is the
*opposite* parity from what the {101} pyramid wants at some spacings — which is the point: mask
volume error is a joint property of shape and grid, and must be measured.

### Strain
| Module | Contents |
|---|---|
| `strain_fourier.py` | Full strain tensor by Fourier continuum elasticity (Andreev *et al.*, *J. Appl. Phys.* **86**, 297 (1999)). Exact operator, one FFT, **but requires homogeneous elastic constants**. |
| `elasticity_fd.py` | **Pryor's method**: strain energy discretized with trilinear (Q1) finite elements and minimized by conjugate gradient. Carries **position-dependent** `C_ijkl`. |

Both minimize the same functional; they differ only in that a Fourier solve cannot represent the
dot/matrix stiffness contrast. That is not a small effect. For a misfitting inclusion,

    Tr(ε) = −3 ε_T · 4μ_m / (3K_i + 4μ_m)

involves the **inclusion's** bulk modulus and the **matrix's** shear modulus, so no single choice
of constants reaches the true value — the homogeneous InAs/GaAs bracket is [−0.0841, −0.0927]
while the truth is −0.1068. Measured against Pryor's quoted conduction-well depths, using
`elasticity_fd` cuts the error from +69/+91 meV to +23/+16 meV.

Use `strain_fourier` when the constants really are uniform, as an independent check (the two agree
in that limit), and for speed.

#### A preconditioner cutoff that made the strain error grow with resolution

`elasticity_fd` preconditions CG with the exact inverse of the *homogeneous* symbol S(k), skipping
wavevectors where S is singular. Only k = 0 is — rigid translation costs no energy — and the test
for it used to be

```python
good = np.abs(det) > 1e-10 * np.abs(det).max()      # WRONG: a magnitude test, not a rank test
```

S vanishes as |k|² at small k, so det S ~ |k|⁶. On a grid whose longest side is N the smallest
nonzero wavevector 2π/N therefore has det/det_max falling off roughly as N⁻⁶, and it crosses the
fixed `1e-10` floor near N ≈ 150. Past that the test starts discarding a shell of long-wavelength
modes that are **perfectly well conditioned and merely small**, as the physics requires them to be
— and discards a wider shell the finer the grid. Measured, over the grids this repo actually ran:

| grid | sites | smallest legitimate det ratio | modes dropped |
|---|---|---|---|
| 70×70×40 | 0.20M | 8.2e-09 | 1 ✓ |
| 88×88×50 | 0.39M | 2.1e-09 | 1 ✓ |
| 116×116×67 | 0.90M | 3.9e-10 | 1 ✓ |
| 120×120×83 | 1.20M | 3.2e-10 | 1 ✓ |
| 140×140×103 | 2.02M | **1.3e-10** | 1 ✓ (by a factor of 1.3) |
| 158×158×91 | 2.27M | 5.7e-11 | **5** ✗ |
| 160×160×110 | 2.82M | 6.2e-11 | **5** ✗ |

Those are exactly the modes carrying an inclusion's long-range relaxation. Zeroing them makes the
preconditioner a **low-pass filter**, and CG cannot recover what the preconditioner never admits to
the Krylov space — so the residual still reports convergence, because it is measured in the
filtered space.

The failure signature is worth remembering, because none of it looks like a linear-algebra bug:

- one run returned a conduction edge at **−12.25 eV** and a negative gap;
- another returned a *plausible* gap while making ⟨Tr ε⟩ 4×10⁻³ **more** compressive as the grid
  refined at fixed box — impossible, since a better-resolved box relaxes more, not less;
- both were **slow**, which read as "big grid." In fact filtering left CG grinding unpreconditioned
  to its `maxiter=500` cap: 6125 s / 500 iters ≈ 12 s per iteration, against 109 s / 8 iters ≈ 13.6
  s per iteration on the 2.02M-site grid that *was* fine. Same cost per iteration, 60× the count.

So the iteration cap was a symptom, not the cause — and `build` reported the residual only under
`verbose=True`, so an entire box-convergence study ran on two silently broken grids.

Three changes, in decreasing order of importance:

1. **`_nonsingular`** tests `det` against `(tr/3)³`. Both scale as the cube of the block's own
   magnitude, so the ratio is O(1) for any well-conditioned block however small, making it a
   statement about *conditioning*. k = 0 is excluded separately on magnitude, which is the property
   it actually lacks.
2. **`require_converged=True`** on `solve_strain_fd`, which raises on a missed tolerance, a CG
   breakdown, or any mode dropped beyond k = 0. An unconverged strain field is not a slightly worse
   answer, it is a wrong one, and it propagates silently into every band edge downstream.
3. **`_pcg` distinguishes breakdown from slow convergence.** Non-positive `p·Ap` or `r·Mr` now
   reports a reason instead of breaking out of the loop and returning whatever the last step left
   in `U`.

`scripts/elasticity_preconditioner_check.py` reproduces the table above (`--diagnose`, seconds, no
solve), confirms the fix is invisible where the old test was safe, and re-runs both broken points.

### Materials

`materials.py` is the **single source of material data**. GaAs, InAs, GaSb and InSb, plus the
In*ₓ*Ga₁₋ₓSb and InAs₁₋ₓSb*ₓ* alloys, all transcribed from `bandparameters_vurgaftman2001.xlsx`
and `bowingparameters_vurgaftman_2001.xlsx` — the band-parameter and bowing tables of Vurgaftman,
Meyer and Ram-Mohan, *J. Appl. Phys.* **89**, 5815 (2001). Every value carries a `PROVENANCE`
tag; `python materials.py` prints the whole table plus four consistency checks.

Three things worth knowing before using it:

- **Sign convention.** Vurgaftman's, in which `a_v` is *negative* for all four binaries and the
  gap deformation potential is the **sum**, `a_gap = a_c + a_v`. This is not the commoner Van de
  Walle tabulation. `qdsolver_core.MATERIALS` is in that other convention and is deliberately
  **not** migrated, because the validated Pryor InAs/GaAs reproductions read it — converting is
  exactly `a_v → −a_v`, and mixing them unnoticed is a ~100 meV error that looks plausible.
- **Temperature.** `set_temperature(T)` moves the gaps (Varshni) *and* the lattice constants, and
  the lattice constants are the only thing setting the misfit. Default 0 K, which is what the
  rest of the set is — the Kane expression reproduces all four tabulated band-edge masses to the
  digit at 0 K and is 24% out on InSb at 300 K.
- **`e14` and `ε_r` are additions to the workbook**, not from the review's own tables, and
  `e14`'s sign is opposite to `pryor1998.PRYOR_TABLE_I`. A global flip reverses the piezoelectric
  potential everywhere; it changes nothing computed with `use_piezo=False`. The Pryor fixture
  keeps its published values, so the two deliberately disagree.

`materials_sb.py` is now a thin antimonide layer over it — the narrative for the InAs/In*ₓ*Ga₁₋ₓSb
system, the legacy switches, and `SB_MATERIALS` aliased to `materials.MATERIALS`. New code should
import `materials` directly.

### The antimonide system (InAs / In*ₓ*Ga₁₋ₓSb)
| Module | Contents |
|---|---|
| `materials_sb.py` | The InAs/In*ₓ*Ga₁₋ₓSb narrative layer over `materials.py`: `ingasb(x)`, the alignment table, the legacy switches, and the record of the three parameter errors. Holds no numbers of its own. |
| `ingasb_dot.py` | Shapes, environment build, band landscape, the strain-pocket binding criterion, single-band electron states, the eight-band opt-in, and the size sweep of eight-band hole levels (`hole_size_sweep`, checkpointed to JSON) with its two figures. |

This system is **broken gap**: the In*ₓ*Ga₁₋ₓSb valence edge lies above the InAs conduction edge at
every composition. Holes sit in the island, electrons are expelled from it, and the only
single-particle thing that could hold an electron is the pocket the island's compression digs in
the surrounding InAs. Measured, that pocket **does not bind** — over x = 0.35 → 1.0 its depth goes
83 → 190 meV and the computed ground state moves 3.6 meV, in the wrong direction, because it is box
zero-point energy. `ingasb_dot.pocket_metrics` makes this box-independent: binding needs
V₀R² > π²ħ²/8m ≈ 3.78 eV·nm², and the pocket reaches at most 2.45 even for the largest island at
x = 1. So the electron here is Coulomb-bound to the hole, and any "electron binding energy" from a
single-particle solve in a finite box is measuring the box.

### The antimonide parameter set had three errors, and the benchmark now passes

This system has a published calculation to check against — C. E. Pryor and M.-E. Pistol,
*Phys. Rev. B* **72**, 205311 (2005), whose Table I (wells) and Table III (dots) both cover
InSb on InAs. Reproduce with `python scripts/insb_pryor_pistol_check.py`; full parameter audit
sheet via `python scripts/dump_sb_parameters.py > SB_PARAMETERS.md`.

| | our gap | their gap | error |
|---|---|---|---|
| pseudomorphic well (analytic) | 0.394 | 0.394 | **0 meV** |
| dot, spherical cap h/d = ¼, **box-converged** | 0.644 | 0.673 | **−29 meV** |

Both absolute well edges land exactly too: CB 0.645, VB 0.251 on their scale. The well is the
load-bearing benchmark: closed-form biaxial strain, no elasticity solve, no shape, no grid, no
box. Nothing numerical can flatter it.

**The dot number is box-dependent, and an earlier version of this README quoted +2 meV for it.**
That was `h = 0.5, pad = 7.5` — one point on a curve that had not been converged. `elasticity_fd`
is periodic, so a tight box lets the island feel its own images, which resist relaxation, overstate
|Tr ε| and overstate the strain-opened gap. Running the padding out to 30 nm
(`scripts/insb_box_convergence.py`, now affordable — see the preconditioner section above):

| pad (nm) | lateral fill | vertical fill | ⟨Tr ε⟩ | gap | vs P&P |
|---|---|---|---|---|---|
| 5 | 68% | 32% | −0.08487 | 0.6994 | +26 meV |
| 7.5 | 57% | 24% | −0.08188 | 0.6684 | −5 meV |
| 10 | 50% | 19% | −0.08082 | 0.6574 | −16 meV |
| 14 | 42% | 14% | −0.08005 | 0.6503 | −23 meV |
| 18 | 36% | 12% | −0.07970 | 0.6473 | −26 meV |
| 24 | 29% | 9% | −0.07944 | 0.6453 | −28 meV |
| 30 | 25% | 7% | −0.07932 | 0.6444 | **−29 meV** |

Successive steps are −31.0, −11.0, −7.1, −3.0, −2.0, **−0.9 meV**: converged. Repeating at
`h = 0.3` gives a constant **−2.8 meV** resolution offset at both pad 14 and pad 18, so this is the
box and not staircasing of the curved cap.

**The box is converged; the resolution is not.** At fixed `pad = 3R` the gap is 0.6489 at
`h = 0.5`, 0.6444 at `h = 0.4`, and the `h = 0.3` runs sit −2.8 meV below their `h = 0.4`
counterparts. Refinement is still moving the answer downward by a few meV per step, so **−29 meV
is a lower bound on the disagreement, not a converged value** — the `h → 0` limit is further from
Pryor & Pistol, not closer. Quote the residual as −24 meV at `h = 0.5`, −29 meV at `h = 0.4`, and
say which.

Two things follow, and the second is the more useful one:

1. **The box was not the explanation.** Enlarging it moves *away* from their value. The residual
   disagreement is real and lives in the shape or the 3D strain field, not in the parameter set —
   which the analytic well already certifies at 0 meV.
2. **Our curve crosses their value at just under 60% fill.** Pryor's 1997 grid states 50%/17% fill
   (130×130×120 sites, island in 65×65×20), which is the pad = 10 row at −16 meV. So the former
   +2 meV was largely the bias of a box slightly tighter than his; matching his stated fill roughly
   halves the discrepancy without closing it.

Any dot band edge quoted at pad ≤ 7.5 nm is biased high by at least 24 meV. The well is not.

*(This table is `box_conv3.log`, regenerated after the lattice constant became temperature
dependent. Every row moved down by a uniform ~1.5 meV — a 0 K lattice gives a 0.4% smaller misfit
than the 300 K one used before. The convergence behaviour and both conclusions are unchanged.)*

**How the errors were found.** The well is closed-form biaxial strain — no elasticity solve, no
shape, no grid — so having it disagree while the dot disagreed *by exactly the same amount scaled
by the strain ratio* (92 × 0.08269/0.05908 = 129 meV, to under a meV) proved the fault was a
coefficient multiplying Tr(ε), not anything geometric. Back-solving the well for the required
hydrostatic gap deformation potential gave `a_gap(InSb) = −7.29 eV` against the database's −5.73.
Vurgaftman's own table gives **−7.30**. That 10 meV agreement is what pinned it.

**Three errors in `aestimo/database.py`, all corrected from the review's own tables**
(`bandparameters_vurgaftman2001.xlsx` in the repo root, now the sole source for `materials.py`;
nothing in the package reads the database any more):

1. **`a_v` sign, for InAs and InSb.** The review is explicit that its `a_v` is negative and that
   `a_gap = a_c + a_v` — *"This implies a negative value for a = a_c + a_v. Note that our sign
   convention for a_v is different from many other works found in the literature."* The database
   has InAs **+1.00** and InSb **+0.31** against the review's −1.00 and −0.36, which makes the
   valence band move the wrong way under compression. GaSb's −1.32 had the right sign and the
   wrong magnitude (−0.80). This **inverts the module's former diagnosis**: GaSb was the *correct*
   entry, and `GASB_AV_CONVENTION='signfixed'` made things worse. That switch is now deprecated.
2. **`a_c` for InSb and GaSb**: −6.04 → **−6.94** (a transposed digit, most likely) and
   −9.33 → **−7.50**.
3. **The gaps were ~300 K while everything else was 0 K.** The *derived* electron mass settles
   this independently: at 0 K it gives 0.0260 / 0.0390 / 0.0135, reproducing the review's own
   tabulated masses to the digit, where the 300 K gaps give 0.0102 for InSb — 24% out. `Ep` and
   `F` are 0 K quantities. The module now initializes to `set_gap_source('varshni', T=0)`.

Also corrected from the same source: InAs `delta_so`, the InAs Luttingers, GaSb `C44`, the InAs
`VBO` by 10 meV, and `d` for GaSb and InSb — which were tagged `UNVERIFIED` and now are not.

Resulting `a_gap = a_c + a_v` (the convention-independent number): InAs **−6.08**, GaSb **−8.30**,
InSb **−7.30**. Pryor's independently-transcribed 1998 Table I gives InAs −6.00.

**What moved.** For the InSb lens in InAs: hole well 888 → **816 meV**, island valence edge above
the matrix maximum 581 → **514 meV**, strained broken-gap overlap 488 → **399 meV**, unstrained
overlap 180 → **173 meV**. The electron pocket is unchanged at 184 meV and **still does not
bind** — V₀R² reaches 0.57 against a threshold of 3.61 eV·nm² — so that conclusion survives.


Band edges themselves are **size-independent** — Pryor & Pistol's scale-invariance argument,
verified to 0.0 meV over a 4× size range by `scripts/insb_band_edges_vs_size.py` when the grid
*and* box are scaled with the island. With them held fixed the same sweep shows a spurious
**50.2 meV** drift, driven by the island filling 40% → 73% of the periodic box. Size enters the
physics only through confinement energy, not through the band edges.

The scaled test is exact rather than approximate, which is worth knowing when reading it: with
`h = R/20` and `pad = 3R`, the grid is 8R/(R/20) = **160 × 160 × 130 at every R**, so each size is
literally the same discrete problem. Every column agrees to the printed digit — same 6,872 island
cells, same volume error, same ⟨Tr ε⟩ = −0.07976, same edges — and the 0.0 meV spread is bit
equality, not cancellation.

**x = 1 is InSb**, and `insb_lens_inas.ipynb` takes that end of the range on its own terms: an
InSb lens in InAs, put through Pryor's Figs. 2 and 4 rather than through the composition sweep.
Misfit −6.50% (against −6.69% for InAs/GaAs, so linear elasticity is pushed just as hard),
stiffness contrast K_dot/K_matrix = 0.824 (against 0.753), broken-gap overlap 180 meV unstrained
and several hundred once the compression is applied. Fig. 4(a) has no counterpart — there is no
bound electron to plot — so what is swept in its place is the V₀R² criterion. Fig. 4(b), the
eight-band hole ladder, ports directly. **Fig. 6 is deliberately not ported**: it is a Hartree
electron–hole pair, and here the Hartree loop would be doing the binding rather than correcting
it, on top of a σ-targeting problem that is genuinely unsolved in a broken-gap spectrum.

**The source data is no longer aestimo's `database.py`.** `materials.py` is transcribed from the
Vurgaftman workbooks directly, which removed the two defects that file had — the 10× unit
inconsistency in `C11`/`C12`, and the `a_v` signs (see the three errors above; the diagnosis the
old `GASB_AV_CONVENTION` switch was built on turned out to be **inverted**, and the switch is
deprecated). Nothing here imports `aestimo_database.py`.

Masses are still **derived** from the Kane expression rather than tabulated, because that database
listed InAs `m_e = 0.4`, an order of magnitude off. With the corrected 0 K parameter set the
derived values now reproduce the review's own tabulated masses to the digit for all four binaries
— which is a joint check on `Eg`, `Δ_so`, `E_P` and `F` at once, and the sharpest evidence the set
is self-consistent. `materials.audit()` prints it.

### k·p and fields
| Module | Contents |
|---|---|
| `kp_pryor.py` | **The authoritative k·p matrix.** 4-, 6- and 8-band Hamiltonians transcribed from Pryor's 8×8, Bir–Pikus strain terms, interband strain terms *u*/*v*, `local_band_edges`, `hole_sigma`. |
| `kp_confined.py` | Grid operators and the validated finite-difference discretization (`GridOperators`). |
| `piezoelectric.py` | Zincblende piezoelectric potential from the shear strain. |
| `pryor1998.py` | Pryor's Table I and his band-edge conventions. |
| `eigensolvers.py` | LOBPCG, folded-spectrum and shift-invert solvers, plus a residual-based verification layer. |

### Reproductions
| Module | Contents |
|---|---|
| `pryor_fig4.py` | Bound-state energies vs island size (Fig. 4). Environment build, localization scoring that separates bound states from finite-box states, Kramers pairing, the size sweep, both panels, and `check_claims` — which prints each quantitative statement in Pryor's Sec. V beside what this code gives. |
| `pryor_fig6.py` | The b = 14 nm environment, the eight-band Hartree exciton, and the isosurface rendering (Fig. 6). |

Both are imported by `pryor_inhomogeneous.ipynb`, hence their place at the root.

---

## Two conventions that will bite you

**1. Pryor's `a_v` sign is not Van de Walle's.** Pryor's Table I satisfies `a_g = a_v + a_c`, so
his valence edge shifts by **−a_v·Tr(ε)**. Vurgaftman et al. use the same convention and say so
explicitly, so **`materials.py` and `pryor1998.py` agree**: `a_v` negative, gap potential the sum.
The commoner Van de Walle convention has `a_g = a_c − a_v`, quotes `a_v` positive, and shifts by
**+a_v·Tr(ε)** — that is what `qdsolver_core.MATERIALS` carries, and it is kept that way because
the validated Pryor InAs/GaAs reproductions read it. Converting is exactly `a_v → −a_v`. Mixing
them moves the valence band the wrong way and inflates the strained gap by ~150 meV.

**2. Hydrostatic strain is easy to count twice.** `band_edge_fields` returns band edges that
*already* carry `a_c·Tr(ε)` and `−a_v·Tr(ε)`. Passing those to a Hamiltonian builder with
`include_hydrostatic=True` applies it again. The flag defaults to `False` for that reason; set it
`True` only when handing in genuinely unstrained edges.

---

## Verification approach

Numerical results are checked against closed-form answers wherever one exists, and against a
second independent method where one does not:

- **Exact analytic targets** — pseudomorphic and clamped slabs (2.8e-17), the Eshelby sphere,
  the *inhomogeneous* misfitting sphere (0.26%).
- **Over-determined identities** — the Bir–Pikus terms are fixed relative to the kinetic ones by
  two constants but must reproduce three complex expressions (6.9e-18).
- **Symmetries nothing in the code enforces** — Kramers degeneracy at k = 0 (5e-16 eV), the C4
  antisymmetry of the piezoelectric potential (6e-15), valence-band isotropy in the spherical
  approximation.
- **Two routes to the same number** — e.g. applying the hydrostatic shift before vs inside the
  Hamiltonian builder (exactly 0).
- **Residual bounds** — for Hermitian *A*, ‖Av − θv‖ = r bounds the distance to a true
  eigenvalue. `eigensolvers.certify` applies it.

A residual **does not** certify that the eigenvalue is the one you wanted. A folded-spectrum
solve returns eigenvalues nearest σ, and a badly placed σ gives converged, certified, wrong
answers — that happened here and invalidated a session's worth of eight-band hole numbers. Use
`kp_pryor.hole_sigma`, and *always* confirm by re-running with σ raised.

**And `hole_sigma` is not enough in a broken-gap system.** It returns the exact k = 0 top of the
local valence band, which is the right target when the hole states sit just below it in a gap —
i.e. in the Pryor benchmark. For InSb/InAs it is the *wrong* target, measured against a dense
diagonalization of the full eight-band matrix: the island-localized states lie **370–850 meV
below** it, and the spectrum at σ itself is a dense InAs electron continuum (~1 meV spacing,
~1% localized) that reaches up through the hole well because the alignment is broken gap. Seeded
there, LOBPCG ran to its iteration limit and returned four box states 0.6% localized; seeded at
the ladder, the same solver converged in 652 iterations to 1.5e-7 eV.
`ingasb_dot.hole_ladder` therefore **scans** σ and selects by localization rather than seeding
once. Three traps it also handles, all of which bit during development:

1. A σ in the continuum never converges, so residuals are tracked per kept level rather than as
   a maximum over the scan — otherwise every run reports a failure that happened to nothing it
   returned.
2. Deduplicating merged results by energy destroys one member of every Kramers pair, after which
   pairing the survivors reports a *level spacing* as a "Kramers splitting" (35 meV, in the case
   that caught it). Levels are formed **within** each solve, before anything is merged.
3. **Coverage, which is the one with no other symptom.** Each solve returns only `probe`
   eigenvalues around its σ, so it covers a finite window; if neighbouring windows do not
   overlap, a level in the gap is invisible. At r = 6 nm with `n_sigma=5, probe=10` the two
   topmost levels vanished this way while Kramers (2e-11 meV), residuals (2e-7 eV) and
   `sigma_hits` all looked clean. `hole_ladder` now computes the window spans and warns on any
   uncovered gap; raise `probe`, not `n_sigma`.

---

## Trust: what is current, and what is not

| Status | Files |
|---|---|
| **Current** | `materials`, `elasticity_fd`, `strain_fourier`, `kp_pryor`, `piezoelectric`, `pryor1998`, `eigensolvers`, `qdsolver_core`, `kp_confined` (discretization only), `pryor_fig4`, `pryor_fig6`, `pryor_inhomogeneous.ipynb`, `pryor_fig2.ipynb` |
| **Superseded, still correct** | `qdsolver_core.trace_strain_from_mask` — hydrostatic-only *and* returns constrained rather than elastic strain, overstating the band-edge shift by 1.165×. Kept only so older notebooks still run. |
| **Known wrong — do not use** | `kp_luttinger.py` and `kp_confined.build_confined_luttinger_kohn` — the split-off band is on the wrong side of the diagonal and the R/S off-diagonals are misplaced. Both carry docstrings saying so. |
| **Suspect results** | `pryor_benchmark.ipynb` and `multiband_bulk_validation.ipynb` import `kp_luttinger`; their **multiband** numbers predate the matrix fix. Their single-band content is unaffected. |
| **Current, and now benchmarked** | `materials`, `materials_sb`, `ingasb_dot`, `ingasb_lens.ipynb`, `ingasb_dash.ipynb`, `insb_lens_inas.ipynb`. Parameters reproduce Pryor & Pistol 2005's InSb/InAs pseudomorphic well to **0 meV** on both edges — closed form, so nothing numerical flatters it. The **dot** gap converges to **−29 meV** against their Table III once the periodic box is enlarged to 25% fill; a former "+2 meV" claim here was one unconverged point at 57% fill, near where our curve happens to cross theirs. Getting this far required correcting three errors in the source database (see "The antimonide parameter set had three errors" below; results from before that are wrong by ~130 meV) and one preconditioner bug in `elasticity_fd` (see "A preconditioner cutoff that made the strain error grow with resolution"). **The residual dot disagreement is unexplained and is not the parameter set.** |
| **Retained in the other sign convention** | `qdsolver_core.MATERIALS` — Van de Walle (`Av` positive), read by the validated Pryor InAs/GaAs reproductions. Correct as it stands; deliberately not migrated. Use `materials.py` for anything new, and never mix the two. |
| **Unaudited — do not use** | `aestimo_database.py` — has a 10× units inconsistency in C11/C12 (GaAs/InAs in 10¹¹ dyne/cm², InSb/GaSb in GPa). Nothing here imports it, and nothing should: `materials.py` supersedes it, transcribed from the review's own tables rather than a copy of them. |

`archive/` holds artifacts whose numbers are known to be wrong; see `archive/README.md`.

---

## Known limitations

- **Discretization.** `elasticity_fd` differs from the exact-operator Fourier solve by ~1.6% at
  h = 1/3 nm in the homogeneous limit, shrinking roughly linearly in h.
- **The pyramid apex is unresolvable.** It tapers to a geometric point, so the topmost cells are
  1–2 across and their strain is staircase noise. Read the taper where the cross-section is still
  ≳8 cells wide.
- **Weakly bound levels are not converged in box size.** An electron bound by ~6 meV has a decay
  length near 10 nm, comparable to the padding around the island, so levels within a few tens of
  meV of the barrier edge shift when the box grows. The Fig. 4 notebook measures this rather than
  assuming it away; deep levels are unaffected.
- **Linear elasticity** at ~7% mismatch is being pushed, and **continuum** elasticity gives the
  pyramid C4v symmetry rather than the true C2v of the zincblende lattice — only an atomistic
  relaxation recovers that (Pryor, Kim, Wang, Williamson & Zunger, *J. Appl. Phys.* **83**, 2548
  (1998)).
- **First-order piezoelectricity only.** Now known to be unreliable in these dots: the
  second-order response is comparable and often opposes it (Bester, Zunger and co-workers,
  *Phys. Rev. B* **74**, 081305(R) and *Phys. Rev. Lett.* **96**, 187602, both 2006). Pryor
  predates that work, so first order is used here for comparability.
- **Open puzzle.** We find 6- and 8-band close with 4-band the outlier; Pryor finds 4- and 8-band
  agreeing to 3 meV with 6-band ~40 meV away. Unresolved.

---

## Citation policy

References are given at the level of *named results* plus paper/chapter locations. **Equation
numbers from sources that could not be checked directly are deliberately not quoted** — a
plausible-looking but wrong equation number is worse than none. Where a source *was* fetched and
read (Pryor 1998, via the arXiv preprint), that is stated explicitly, and the transcription is
noted as having a verifiable provenance chain.
