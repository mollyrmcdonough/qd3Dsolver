# Working notes for Claude

Orientation for an agent working in this repo. `README.md` is the reference; this is the part
that is easy to get wrong.

## What this is

A 3D quantum-dot solver: continuum elasticity → strained band edges → multiband k·p → confined
states. Benchmarked against Pryor 1998 (InAs/GaAs) and cross-checked against Yeap et al. 2009
(InAsSb/InAs). The physics goal is band alignment and **which island sizes confine a carrier**.

## Running things

- **Run from the repository root.** The library is deliberately flat, not a package; modules
  import siblings by bare name. `scripts/` and `tools/` add the parent to `sys.path` themselves.
- Windows. Use `PYTHONIOENCODING=utf-8` — several scripts print characters that crash the default
  Windows console codec.
- **Never pipe a long-running job through `tail`/`head`/`tr`.** They buffer, and the output does
  not appear until the process exits. Redirect to a file and read the file. This mistake has been
  made repeatedly in this repo's history.
- Solves are slow (minutes to hours). Run them in the background, write to a log, and check the
  log. Prefer resumable scripts that write results per item.

## The traps, in rough order of how much time they have cost

### 1. A converged eigenvalue is not the state you wanted

This is the defining failure mode here and it has recurred in three different forms. A residual
certifies that a returned pair *is* an eigenpair, not that it is the one you asked for.

- **Eight-band holes in a broken-gap system.** Seeded at the exact k=0 valence top, a solve
  converged to 7.4e-08 eV and returned twelve perfectly Kramers-paired states that were **0.6%**
  inside the island (random baseline: 8.2%). Use `heterostructure.six_band_holes`.
- **Six-band at fine grids.** Interface artifacts that become *more* localised as h refines
  (42% → 70%), so localisation cannot filter them. See "resolution floor" below.
- **Localisation is a LARGE-DOT criterion. Do not apply the 75% threshold to a small dot.**
  On a 2.5 nm island, `scripts/gamma3_diagnostic.py` measured three *converged* solves at
  40.7%, 46.8% and 54.7% — indistinguishable from the artifact-contaminated solve at 42–54%.
  A small dot barely holds its hole, so genuine spill-out is large; the random-vector baseline
  there is 0.9%, so even 40% is forty times baseline. At 2.5 nm only the **h-trend**
  discriminates. The 80–97% figures quoted elsewhere were measured on 37 nm islands.
- **A non-elliptic parameter set.** The six-band Luttinger operator needs **γ₁ − 2γ₂ > 0 and
  γ₁ − 2γ₃ > 0**. Violate either and it admits modes whose energy is bounded only by the grid.
  Measured: forcing InSb's γ₃ = 16.5 into the InAs matrix (γ₁ = 20.0, so γ₁ − 2γ₃ = −13) returned
  sixteen states at **+2.33 eV, 0% inside the island** — 1.7 eV above the well top. It looks like
  a solver failure and it is a parameter-set failure.
- **Always read the localisation fraction, never the energy alone.** And when a result looks
  clean, ask what check would catch it if it were wrong.

**These materials sit close to the ellipticity boundary**, which is worth carrying around:
γ₁ − 2γ₃ is **+1.8 for InSb (5.2% of γ₁) and +1.6 for InAs (8.0%)**, against 16% for GaAs. Linear
averaging across the interface preserves it (+1.65 to +1.75 at every mixture), so ordinary
averaging is not what breaks it — but any scheme that extrapolates, over-weights one side, or
mixes γ₃ from one material with γ₁ from another can cross it. `scripts/gamma3_diagnostic.py`
carries `check_elliptic`, which asserts both inequalities pointwise before any solve; run it on a
new material pair before trusting a first result.

**But passing γ₁ − 2γ₂ > 0 and γ₁ − 2γ₃ > 0 buys much less than it looks like.** That is the
*Legendre–Hadamard* condition and it is necessary, not sufficient. The condition that governs
whether the spectrum is bounded at an interface is **strong** ellipticity, and it fails for every
material here — see trap 2, which is the single most important thing in this file.

### 2. The interface modes are the OPERATOR, not the grid — small dots are unreachable

**This entry was wrong three times. Read it before touching the solver.**

The symmetrized six-band operator `k_i γ(r) k_j` has **no maximum at an abrupt interface**. Not a
discretization defect — a property of the operator every correct discretization must reproduce.

Why: the 6-band kinetic tensor, written as one 18×18 matrix over (direction, band), is
**Legendre–Hadamard elliptic but not strongly elliptic**. LH — γ₁−2γ₂ > 0 and γ₁−2γ₃ > 0, i.e.
every bulk band curves downward — holds everywhere. Strong ellipticity, the same for *every*
18-vector rather than just the rank-one ones, holds nowhere:

| | γ₁−2γ₂ | γ₁−2γ₃ | **λ_max(A)** | worst rank-one |
|---|---|---|---|---|
| GaAs | +2.86 | +1.12 | **+0.226** | −0.043 |
| GaSb | +4.00 | +1.40 | **+0.533** | −0.063 |
| InAs | +3.00 | +1.60 | **+0.937** | −0.082 |
| InSb | +3.80 | +1.80 | **+1.741** | −0.102 |

Constant or *continuous* coefficients only ever expose the rank-one directions, where A is
negative — so the bulk is fine and a graded interface is fine. A step exposes the rest.

**γ₃ carries all of it**, which is also why `scripts/gamma3_diagnostic.py` isolated γ₃ — the right
parameter from a hypothesis about stencils that was wrong:

| λ_max(A) | full | γ₃ = 0 | γ₂ = 0 | both = 0 |
|---|---|---|---|---|
| InSb | +1.7412 | **−0.1448** | +0.5601 | −1.3259 |
| InAs | +0.9373 | **−0.1143** | +0.2896 | −0.7620 |

Zeroing γ₃ restores strong ellipticity; zeroing γ₂ does not. γ₃ is the only coefficient `ops.cross`
is ever called with, and the R and S blocks are exactly what Burt–Foreman reorders.

**The controlled measurement** (`scripts/planewave_validation.py --only 3`), on Yeap's 2.5 nm
island at fixed h = 0.20 nm so only the plane-wave basis grows:

| gmax | s = 1 (λ_max = +1.741) | s = 0.25 (λ_max = −0.559) |
|---|---|---|
| 3.00 | +0.4571 | +0.0260 |
| 5.00 | +1.0115 | +0.0262 |
| 7.85 | +1.9508 | +0.0263 |

`s` scales γ₂,γ₃ and changes nothing else. Island valence edge 0.677 eV. One column is flat to
0.1 meV; the other quadruples and is still climbing. Refining the grid at fixed basis fraction
gives a ceiling ∝ 1/h² (E·h² = 0.094, 0.094, 0.073, 0.078 eV·nm² at h = 0.40 → 0.20), which is
the FD solver's measured 0.046/h² artifact — same law, because in both methods h is setting how
sharp the interface may be, not how fine a stencil is.

**Three fixes were tried and none failed — all three were aimed at a cause that was not there:**
the compact-vs-central α-sampling mismatch, `cross_k2_operator_staggered` (a *wrong* operator, its
symbol is non-negative everywhere and cannot represent a mixed derivative), and
`cross_k2_operator_fe` (a *correct* Q1 element, symbol exact to 7e-15, no effect). Both rewrites
are preserved. **Do not attempt a fourth stencil.**

### The fix, and where it is

Burt–Foreman **ordering** — a different operator, not a different discretization. In the DKK
orbital basis the cross term is `H_XY = kx N⁺ ky + ky N⁻ kx` with `N⁺ + N⁻ = N = −6γ₃·ħ²/2m₀`;
symmetrized takes `N± = N/2`, Burt–Foreman takes **`N⁻ = M`**, the diagonal coefficient
(Burt, J. Phys.: Condens. Matter 4, 6651 (1992); Foreman, PRB 48, 4964 (1993)).

**Implemented in `kp_planewave`, and it works.** λ_max goes +1.741 → 5e-16 for InSb and to machine
zero for every material. Same island, same grid, real parameters, only the ordering different:

| gmax | symmetrized | burt-foreman |
|---|---|---|
| 3.00 | +0.4571 | +0.1727 |
| 5.00 | +1.0115 (+434 meV) | +0.1799 (+1.1 meV) |
| 7.85 | +1.9507 (+365 meV) | +0.1813 (+0.3 meV) |

Bound 0.7630 eV: Burt–Foreman respects it, symmetrized exceeds it by 1.19 eV. `ordering=` defaults
to `'burt-foreman'` on every solver entry point in that module.

Note it puts **γ₁ and γ₂ into the cross terms**, where symmetrized had only γ₃ — which is why
holding γ₃ uniform appeared to cure things.

**Quote E_trans, not confinement, on a small dot.** These are broken-gap type II, so
`E_trans = E_c(matrix, far) − E_hole` *is* the transition energy, and unlike a confinement energy
it never touches `v_top`. On a 2.5 nm mask almost every cell is a boundary cell, so even the mean
v_top drifts (0.6051 → 0.5891 eV over h = 0.40 → 0.20) while the hole level itself is flat to
~3 meV (0.1821 / 0.1788 / 0.1815 / 0.1813). Reading confinement there imports a drift that is not
in the solve.

**The box is the binding constraint once the ordering is right, and the reason is the LIGHT hole.**
A six-band state carries an lh admixture whose decay length is `sqrt(G0/(m·E))` = 3.7 nm at
m ≈ 0.015, against 0.73 nm for the heavy component — so a state that looks deeply bound still needs
several nm of matrix. Measured on the 2.5 nm island at h = 0.30, E_trans over crop pad 2 → 3 → 4 →
6 nm: **0.2279 → 0.2365 → 0.2383 → 0.2391 → 0.2395 eV** (+8.6, +1.8, +0.7, +0.4). The convenient
2 nm crop is 12 meV short. Use `kp_planewave.crop_env` to keep a generous strain box and a separately-chosen
k·p box, and **measure the pad series** — do not infer it from the confinement energy.

Converged, all three axes: **E_trans(2.5 nm, AR 1, 80 K) ≈ 0.240 eV**, against Yeap's 0.16 eV.
The +80 meV gap is now a physics/parameter question, not a numerics one.

**`kp_confined` is still symmetrized**, so the finite-difference path — `six_band_holes`, the
production sweeps — keeps its resolution floor. There, results are usable only where the runaway
is small against the well depth.

| case | h | ceiling | well | ratio | usable? |
|---|---|---|---|---|---|
| 2.5 nm dot (Yeap et al.) | 0.31 | 0.47 eV | 0.86 eV | 0.55 | **no** |
| 37 nm × 3 nm island | 0.62 | 0.12 eV | 0.86 eV | 0.14 | yes |

Keep h ≳ 0.5 nm AND require the dot to be well resolved at that spacing — islands of ~10 nm and
up. On the 2.5 nm dot the FD confinement drifts 430 → 326 → 299 → 214 → 174 meV over h = 0.50 →
0.25 with no convergence; **at the coarsest grid it happens to agree with the published value**,
and that agreement is coincidence.

GaAs's λ_max is eight times smaller than InSb's — a difference of degree, not of kind.
**InAs/GaAs runs away too**: a 12 nm InAs/GaAs ellipsoid at h = 0.40, symmetrized, gives
confinement 13.3 → −68.5 → −264.8 meV as the plane-wave basis grows. The Pryor benchmark passes
because it is a large island at a coarse spacing, where the 0.046/h² ceiling is small against the
well — not because GaAs is immune. Never read a passing benchmark as evidence the method is sound
at a smaller size or a finer grid.

### 3. Sign and composition conventions

- **`a_v` sign.** `materials.py`, `pryor1998.py` and Vurgaftman all use `a_gap = a_c + a_v` with
  `a_v` negative; the valence edge shifts by **−a_v·Tr(ε)**. `qdsolver_core.MATERIALS` is in the
  *other* (Van de Walle) convention and is deliberately not migrated. Mixing them is a ~150 meV
  error that looks entirely plausible.
- **Hydrostatic strain is easy to apply twice.** `band_edge_fields` output already carries it.
- **Alloy composition.** `materials.alloy('InAsSb', x)` takes x as the **antimony** fraction. Most
  of the literature quotes arsenic. `alloy('InGaSb', x)` takes x as the **indium** fraction.
- **Aspect ratio is AR = d/h**, diameter over height, so **larger AR = flatter island**. This is
  the literature convention (Yeap, Rybchenko, Pryor) and, since 2026-08-05, this repo's everywhere.
  It used to be h/d — the reciprocal — so **anything predating that change reads backwards**:
  sweeps written before it store `aspect` (h/d), sweeps since store `AR`. Both the sweep driver
  and the summariser refuse to run on a file carrying `aspect`; `scripts/migrate_aspect_to_AR.py`
  converts one in place, keeping a `.h_over_d.bak`. `scripts/convention_audit.py` pins the map
  (h/d 0.08 → AR 12.5, 0.10 → 10, 0.15 → 6.667) so a half-converted script fails loudly instead of
  producing a plausible curve running the wrong way.

### 4. Boxes and padding

A well's shallowest contour grows without bound with padding — the 0 meV "hole well" went
23,357 → 124,418 nm³ over pads 7.5 → 18 nm while the 100 meV contour held at ~1,440. Anything
touching the box wall is measuring the box. `well_metrics` flags it; `critical_size` discards it.

For a **bound** hole (hundreds of meV) a tight box is fine — it decays within a nm or two. For an
unbound electron it is not; those states just measure the box.

## Useful invariants

Cheap checks that have each caught a real error:

- **Scale invariance.** Continuum elasticity has no length scale, so at fixed *shape* the strained
  band edges are size-independent. If an "offset" column drifts across a size sweep, the strain
  solve or the box is not converged. This is free — it comes with every sweep.
- **Kramers degeneracy.** Every level is exactly twofold. Nothing imposes it, so it is a genuine
  check — but note it is necessary, not sufficient: the spurious states pass it too.
- **Random-vector localisation baseline** = island volume / box volume. A state below that is
  actively avoiding the island.
- **A bound hole lies below the valence edge that binds it.** `six_band_holes` filters on this.

## Physics worth knowing before touching the antimonides

- The antimonide dots here are **broken-gap type II**: the dot valence edge lies *above* the
  matrix conduction edge, so the electron is expelled (the InSb dot's conduction edge sits ~810
  meV above the InAs matrix) and the transition is spatially indirect,
  `E_c(matrix) − E_hole`.
- **A gap only opens when hole confinement exceeds the broken-gap offset.** The offset is
  size-independent; confinement is not — which turns the whole question into one about size.
- **Compressive strain makes it worse.** For InSb/InAs the unstrained overlap is 173 meV; coherent
  strain adds ~278 meV, almost all of it the biaxial shear raising the heavy hole.
- **The hole is doing the work, and it is heavy.** m_h/m_e ≈ 28 in InSb, so for the same
  confinement energy the hole needs an island ~5× smaller in linear size than an electron would.
  That is why these systems need very small dots, not because confinement is weak.
- **Confinement does not scale as 1/L².** In a finite well the state spills out as it is squeezed;
  measured behaviour over a 3× size range is closer to 1/L. Extrapolating from one size
  overestimates the critical radius substantially.

## Style

- Comments explain *why*, and record measured numbers. Docstrings carry the evidence for a claim
  and the failure that motivated a guard. When something is known-wrong, say so in its own
  docstring rather than only in the README.
- **Citation policy:** name results and give paper/chapter locations; do not quote equation
  numbers from sources that were not directly read. A plausible-looking wrong equation number is
  worse than none.
- Record rejected approaches with the measurement that rejected them. Several are preserved in
  this repo precisely so they are not retried.
