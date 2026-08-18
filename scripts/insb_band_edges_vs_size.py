"""Band edges of an InSb lens in InAs as a function of island size, Pryor-Pistol style.

Two questions, one cheap calculation each. Neither needs an eigensolve: band edges are the k = 0
eigenvalues of the strain Hamiltonian evaluated pointwise, which is seconds, not hours.

Q1: DO THE BAND EDGES DEPEND ON SIZE AT ALL?
--------------------------------------------
C. E. Pryor and M.-E. Pistol, "Band-edge diagrams for strained III-V semiconductor quantum wells,
wires, and dots", Phys. Rev. B 72, 205311 (2005), Sec. III:

    "Since the equations governing the strain are scale invariant, the strain and band energies
    do not vary with the size of the structure, and the band-edge results are applicable to any
    size structure. However, the size of the structure does affect the confinement energy."

So the answer should be NO -- exactly flat -- and any variation is numerical. But testing that
requires care, because a fixed grid does not respect the scale invariance it is testing:

  * with h FIXED, the small island is the most staircased (fewest cells across it);
  * with pad FIXED, the LARGE island is the most boxed-in -- `elasticity_fd` uses periodic
    boundaries, so an island filling more of its box interacts with its own periodic images.

Those two errors pull in opposite directions along the size axis and neither is the physics.
An earlier version of this script held both fixed, saw the conduction edge rise monotonically
past r = 6 nm, and attributed it to staircasing at the *small* end -- exactly backwards. The
mean hydrostatic strain was the tell: it got monotonically MORE compressive as the island grew,
which is the periodic-image signature, not a discretization one.

So this script runs both setups and prints them side by side:

  FIXED    h = 0.5 nm, pad = 6 nm for every island   -- what a naive sweep does
  SCALED   h and pad both proportional to the island -- every size is numerically the SAME
                                                        problem, so scale invariance can be
                                                        tested rather than confounded

Under SCALED the edges should agree to the level of the residual discretization error alone.

Q2: IS THE PARAMETER SET RIGHT?
-------------------------------
Their Table III (mean band energies for DOTS strained to binary substrates) gives, for substrate
InAs and dot material InSb:

    conduction  0.800 eV     valence  0.127 eV

on a scale where the UNSTRAINED InSb valence edge is zero. This package uses the unstrained InAs
valence edge as its zero, and `materials_sb` derives the offset from their Table I as
InAs(unstrained VB) = -0.590 eV on their scale, so  E(theirs) = E(ours) - 0.590.

This is the first published number this material system has ever been checked against, so three
things have to match for the comparison to mean anything:

1. **Shape.** Their Fig. 1(a) dot is a cap cut from a SPHERE at h/d = 1/4, not a half-ellipsoid.
   `spherical_cap_mask` is that shape; `lens_mask` is the ellipsoid and encloses 23% more volume
   at this aspect ratio, with the excess at the rim where the strain is least homogeneous. Both
   are run so the difference is measured rather than assumed.
2. **Statistic.** Their table is the mean over the dot volume, not the extremum. Means are what
   get compared; spreads are printed beside them because a dot band edge is a distribution --
   that is what their histograms show.
3. **Temperature.** They state they used Vurgaftman et al. at T = 0 K. `materials_sb` defaults to
   `GAP_SOURCE = 'database'`, which carries ~300 K gaps (InSb 0.174 against their 0.235 -- a
   61 meV difference before anything else happens), while its GaSb VBO was DERIVED from their
   0 K table. That internal inconsistency is documented in the module and is exactly what
   `set_gap_source('varshni', T=0)` exists to fix. Both are run below.

Run:  python scripts/insb_band_edges_vs_size.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import qdsolver_core as qd
import ingasb_dot as ig
import materials_sb as ms

#: Offset from this package's zero (unstrained InAs valence edge) to Pryor & Pistol's
#: (unstrained InSb valence edge).
PP_OFFSET = -0.590

#: Pryor & Pistol Table III, substrate InAs, dot material InSb, on THEIR scale.
PP_TABLE_III = dict(cb=0.800, vb=0.127)

RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
ASPECT = 0.25                    # AR = d/h = 4, as in their Fig. 1(a). ASPECT is the reciprocal:
                                 # it multiplies the diameter to give the height.

FIXED_H, FIXED_PAD = 0.5, 6.0    # the naive setup -- deliberately left as-is, it is the exhibit
SCALED_CELLS = 20                # cells across the island radius -> h = R / 20

# pad = 3 R, so the box scales with the island AND is converged. 0.75 R was the original value
# and is NOT converged: `insb_box_convergence.py` shows the R = 10 dot gap still moving 31 meV
# per padding step at 0.75 R (68% lateral fill) and only settling to under 1 meV per step by
# pad = 3 R (25% fill). Absolute edges computed at 0.75 R are biased high by ~25 meV.
#
# This does not affect the scale-invariance CONCLUSION -- a uniform box bias is identical at
# every size by construction, so it cancels out of the spread -- but it does bias every absolute
# edge value the SCALED columns report, which is what a reader will quote.
SCALED_PAD_FRAC = 3.0


def edges_for(shape, h, pad, use_piezo=False, x=1.0, matrix='InAs'):
    """Mean and extremal k=0 band edges inside the island, plus the mean hydrostatic strain."""
    env = ig.build(shape, x, h=h, pad=pad, use_piezo=use_piezo, matrix=matrix, verbose=False)
    e = ig.valence_edge(env)
    m = env['mask']
    g = env['strain'].at(m)
    out = dict(cb_mean=float(e['cb'][m].mean()), cb_min=float(e['cb'][m].min()),
               cb_max=float(e['cb'][m].max()),
               vb_mean=float(e['v1'][m].mean()), vb_min=float(e['v1'][m].min()),
               vb_max=float(e['v1'][m].max()),
               tr=float(g['exx'] + g['eyy'] + g['ezz']),
               cells=int(m.sum()), h=h, pad=pad, vol_err=float(env['vol_err']),
               box=float(len(env['cx']) * h),
               fill=float(2 * shape['params']['radius'] / (len(env['cx']) * h)))
    del env, e
    return out


def run(label, shape_of, scaled, use_piezo=False):
    print(f"\n{'='*104}\n{label}\n{'='*104}")
    print(f"{'R':>4}{'d':>5}{'h':>6}{'pad':>6}{'box':>6}{'fill':>7}{'cells':>8}"
          f"{'volerr':>8}{'<Tr eps>':>10}{'CB mean':>9}{'VB mean':>9}{'gap':>8}")
    rows = []
    for R in RADII:
        s = shape_of(R)
        h = (R / SCALED_CELLS) if scaled else FIXED_H
        pad = (SCALED_PAD_FRAC * R) if scaled else FIXED_PAD
        t0 = time.time()
        r = edges_for(s, h, pad, use_piezo=use_piezo)
        r['R'] = R
        rows.append(r)
        print(f"{R:>4.0f}{2*R:>5.0f}{h:>6.2f}{pad:>6.1f}{r['box']:>6.0f}{r['fill']*100:>6.0f}%"
              f"{r['cells']:>8,}{r['vol_err']:>+8.3f}{r['tr']:>+10.5f}"
              f"{r['cb_mean']:>9.4f}{r['vb_mean']:>9.4f}"
              f"{r['cb_mean']-r['vb_mean']:>8.4f}   {time.time()-t0:.0f}s")
    cb = np.array([r['cb_mean'] for r in rows])
    vb = np.array([r['vb_mean'] for r in rows])
    tr = np.array([r['tr'] for r in rows])
    print(f"  spread over the whole size range:  <Tr eps> {np.ptp(tr)*1e3:.2f} m-strain"
          f"   CB {np.ptp(cb)*1e3:.1f} meV   VB {np.ptp(vb)*1e3:.1f} meV"
          f"   gap {np.ptp(cb-vb)*1e3:.1f} meV")
    return rows


def compare(rows, label, pick=None):
    """Compare one row against Pryor & Pistol Table III. `pick` selects which island; default is
    the median size, which under SCALED is arbitrary (they all agree) and under FIXED is the
    least-contaminated compromise between staircasing and periodic images."""
    r = rows[len(rows) // 2] if pick is None else pick
    cb, vb = r['cb_mean'], r['vb_mean']
    print(f"\n{label}  (R = {r['R']:.0f} nm)   vs Pryor & Pistol 2005 Table III")
    print(f"  {'':<26}{'this work':>11}{'P&P 2005':>11}{'diff':>10}")
    for key, ours in (('conduction', cb), ('valence', vb)):
        theirs = PP_TABLE_III['cb' if key == 'conduction' else 'vb']
        print(f"  {key + ' (their scale)':<26}{ours+PP_OFFSET:>11.3f}{theirs:>11.3f}"
              f"{(ours+PP_OFFSET-theirs)*1e3:>+9.0f}m")
    gap_pp = PP_TABLE_III['cb'] - PP_TABLE_III['vb']
    print(f"  {'edge-to-edge gap':<26}{cb-vb:>11.3f}{gap_pp:>11.3f}"
          f"{(cb-vb-gap_pp)*1e3:>+9.0f}m")
    print("  The gap is the more meaningful comparison: it does not depend on the absolute")
    print("  scale alignment, which is the least certain part of the parameter set.")
    return cb - vb


if __name__ == '__main__':
    cap = lambda R: ig.spherical_lens(R, 2 * R * ASPECT)
    ell = lambda R: ig.lens(R, 2 * R * ASPECT)

    print(f"dot = InSb, matrix = InAs, AR = d/h = {1/ASPECT:g}")
    print(f"energy zero = unstrained InAs valence edge; P&P scale = ours {PP_OFFSET:+.3f} eV")
    print(f"gaps in force: {ms.GAP_SOURCE}  ->  " +
          ", ".join(f"{k} {v['Eg']:.3f}" for k, v in ms.SB_MATERIALS.items()))

    # ---- Q1: is it flat? ----
    fixed = run('SPHERICAL CAP -- FIXED grid and box (what a naive sweep does)', cap,
                scaled=False)
    scaled = run('SPHERICAL CAP -- SCALED grid and box (every size the same problem)', cap,
                 scaled=True)
    print("\nThe SCALED spread is the honest measure of scale invariance. The FIXED spread is")
    print("dominated by the island filling more of a fixed periodic box as it grows -- watch")
    print("the fill column and <Tr eps> move together.")

    # ---- Q2: does the parameter set match a published calculation? ----
    print(f"\n\n{'#'*104}\n#  Q2: against Pryor & Pistol Table III\n{'#'*104}")
    print("\n--- forcing the OLD ~300 K database gaps, to show what the temperature was worth")
    old = ms.set_gap_source('database')
    print("    " + ", ".join(f"{k} {v:.3f} eV" for k, v in old.items()))
    scaled_db = run('SPHERICAL CAP -- SCALED, 300 K gaps', cap, scaled=True)
    gap_db = compare(scaled_db, "spherical cap, gaps = 'database' (~300 K)")

    print(f"\n--- back to 0 K Varshni, the module default and what Pryor & Pistol state they used")
    now = ms.set_gap_source('varshni', T=0.0)
    print("    " + ", ".join(f"{k} {v:.3f} eV" for k, v in now.items()))
    scaled0 = run('SPHERICAL CAP -- SCALED, 0 K gaps', cap, scaled=True)
    gap_0k = compare(scaled0, "spherical cap, gaps = Varshni 0 K")

    ell0 = run('HALF-ELLIPSOID -- SCALED, 0 K gaps (the shape our own sweeps use)', ell,
               scaled=True)
    gap_ell = compare(ell0, "half-ellipsoid, gaps = Varshni 0 K")

    gap_pp = PP_TABLE_III['cb'] - PP_TABLE_III['vb']
    print(f"\n{'='*104}\nSUMMARY -- edge-to-edge gap of the strained InSb dot in InAs")
    print(f"  Pryor & Pistol 2005, Table III      {gap_pp:>8.3f} eV")
    print(f"  this work, 300 K gaps, cap          {gap_db:>8.3f} eV  ({(gap_db-gap_pp)*1e3:+.0f} meV)")
    print(f"  this work, 0 K gaps, cap            {gap_0k:>8.3f} eV  ({(gap_0k-gap_pp)*1e3:+.0f} meV)")
    print(f"  this work, 0 K gaps, half-ellipsoid {gap_ell:>8.3f} eV  ({(gap_ell-gap_pp)*1e3:+.0f} meV)")
    print("\n  The 0 K cap row is the apples-to-apples one -- their geometry, their temperature.")
    print(f"  It sits {abs(gap_0k-gap_pp)*1e3:.0f} meV BELOW them, and that residual is real: "
          "`insb_box_convergence.py`")
    print("  shows the periodic box converged to under 1 meV per step, and enlarging it moves")
    print("  AWAY from their value rather than toward it. An earlier version of this script")
    print("  padded by only 0.75 R and reported near-exact agreement; that was box contamination")
    print("  worth ~25 meV, not agreement. By contrast the pseudomorphic-well comparison in")
    print("  `insb_pryor_pistol_check.py` agrees to 0 meV on both edges, and IS the sharper")
    print("  test because it involves no elasticity solve and no shape approximation -- so the")
    print("  parameter set is exonerated and the residual lives in the shape or the 3D strain")
    print("  field.")
    print("")
    print("  RESOLUTION IS NOT CONVERGED. These sweeps run at h = R/20, which is 0.5 nm at")
    print("  R = 10. `insb_box_convergence.py` at the same pad but h = 0.4 gives 0.6444, and")
    print("  h = 0.3 is a further -2.8 meV. Refinement moves the gap DOWN a few meV per step,")
    print("  so this row is a lower bound on the disagreement, not its converged value.")
    print("")
    print("  The half-ellipsoid row is larger only because it is a different shape: same base and")
    print("  height, 23% more volume, and the excess sits at the rim where the strain is least")
    print("  homogeneous. Use the cap when comparing with them; the ellipsoid is what our own")
    print("  `ingasb_*` notebooks use.")
    print("\n  Reaching this required correcting three errors in aestimo's database -- two sign")
    print("  and value errors in the deformation potentials and a 300 K/0 K mismatch. See the")
    print("  README section 'The antimonide parameter set had three errors'.")
