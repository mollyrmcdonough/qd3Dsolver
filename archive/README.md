# archive/

Nothing in here is current. These files are kept because they record how a result was obtained
and, in two cases, *why the number it reports is wrong* — deleting them would lose the audit
trail. **Do not cite any energy from this directory.** Nothing at the repository root imports
anything here.

Two defects account for most of it.

**The σ-placement bug.** A folded-spectrum or shift-invert solve returns the eigenvalues nearest
the target σ, and its residual certifies only that the answer *is* an eigenpair, not that it is
the one you wanted. Before the Bir–Pikus shear terms existed, the band-edge field `V_h` really
was the top of the valence band, so `V_h.max()` was a correct σ. Adding shear strain silently
invalidated that: the shear raises the local valence edge well above `V_h` — by ~145 meV in the
b = 14 nm pyramid — so eight-band hole solves seeded that way converged, certified, and landed
~55 meV below the true ground state. `kp_pryor.hole_sigma` now computes the exact k = 0 edge.
Every eight-band **hole** energy recorded in this directory predates that fix.

**Homogeneous elasticity.** Everything here used `strain_fourier`, which cannot carry the
dot/matrix stiffness contrast. That overstates the conduction well by ~45 meV at the base and
~75 meV at the tip; see the main README.

| File | Status |
|---|---|
| `benchmark_corrected_strain.log` | Eight-band hole values (`8-band hole 106.4 meV`, `8-band hole 154.8 meV`) are invalidated by the σ bug. The single-band and strain content is unaffected but superseded by `pryor_inhomogeneous.ipynb`. |
| `pryor_benchmark_piezo.ipynb` | Executed (6/6), and the only executed artifact here — which is exactly why it is dangerous. Carries the same σ-bug hole energies. Its piezoelectric potential is still correct and was re-validated independently. |
| `benchmark_rerun.py` | The driver that produced the log above. Kept for provenance only; it calls the pre-fix σ path. |
| `piezo_results.log` | Piezoelectric potential and level splittings from that same run. The potential itself stands (±44 meV, C4 antisymmetry 3e-15, reproduced since); the *state* energies beside it do not. |
| `band_alignment_and_wavefunctions.ipynb` | 0 of 11 cells executed. Superseded by the Fig. 2 sections of `pryor_fig2.ipynb` and `pryor_inhomogeneous.ipynb`, which do the same visualizations against validated strain. |
| `convergence_InAs_GaAs.ipynb` | 0 of 7 cells executed. A single-band grid-convergence sketch from before the multiband work. |
| `inSb_dot_in_inAs.ipynb` | 0 of 10 cells executed. The original InSb/InAs spherical-dot sketch that predates the whole Pryor benchmark; kept as the starting point of the project. |

The three unexecuted notebooks would still *run*, but they import the superseded
`qdsolver_core.trace_strain_from_mask` strain path (hydrostatic-only, and constrained rather than
elastic strain — it overstates the band-edge shift by 1.165×). Re-running them would produce
numbers that look plausible and are not.
