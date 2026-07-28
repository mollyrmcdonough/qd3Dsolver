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
python scripts/elasticity_validation.py    # the elasticity solvers, against analytic ground truth
python scripts/strain_validation.py        # strain tensor + Bir–Pikus terms
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
his valence edge shifts by **−a_v·Tr(ε)**. The commoner convention (and the rest of this
package's `MATERIALS` dict) has `a_g = a_c − a_v` and shifts by **+a_v·Tr(ε)**.
`pryor1998.band_edge_fields` implements *Pryor's*. Mixing them moves the valence band the wrong
way and inflates the strained gap by ~150 meV.

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

---

## Trust: what is current, and what is not

| Status | Files |
|---|---|
| **Current** | `elasticity_fd`, `strain_fourier`, `kp_pryor`, `piezoelectric`, `pryor1998`, `eigensolvers`, `qdsolver_core`, `kp_confined` (discretization only), `pryor_fig4`, `pryor_fig6`, `pryor_inhomogeneous.ipynb`, `pryor_fig2.ipynb` |
| **Superseded, still correct** | `qdsolver_core.trace_strain_from_mask` — hydrostatic-only *and* returns constrained rather than elastic strain, overstating the band-edge shift by 1.165×. Kept only so older notebooks still run. |
| **Known wrong — do not use** | `kp_luttinger.py` and `kp_confined.build_confined_luttinger_kohn` — the split-off band is on the wrong side of the diagonal and the R/S off-diagonals are misplaced. Both carry docstrings saying so. |
| **Suspect results** | `pryor_benchmark.ipynb` and `multiband_bulk_validation.ipynb` import `kp_luttinger`; their **multiband** numbers predate the matrix fix. Their single-band content is unaffected. |
| **Unaudited** | `aestimo_database.py` — has a 10× units inconsistency in C11/C12 (GaAs/InAs in 10¹¹ dyne/cm², InSb/GaSb in GPa). Nothing here imports it. `qdsolver_core.MATERIALS` is the correct source. |

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
