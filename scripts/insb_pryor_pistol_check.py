"""Where does our InSb/InAs band-edge disagreement with Pryor & Pistol 2005 come from?

`insb_band_edges_vs_size.py` established that our strained InSb DOT in InAs disagrees with their
Table III at 0 K parameters and matched geometry, and that the disagreement is size-independent
(band edges are scale invariant, verified to 0.0 meV). This script asks which ingredient is
responsible.

HOW BIG THE DOT DISAGREEMENT ACTUALLY IS. It was 129 meV when this script was written, which was
the parameter set; correcting the three errors documented in `materials_sb` removed almost all of
it. What is left is -23 to -30 meV, depending on how far the grid is refined, and it only shows
up in a CONVERGED periodic box -- see `insb_box_convergence.py`. Section 2 below runs at
pad = 7.5 nm, which is 57% lateral fill, and at that fill our gap curve happens to CROSS their
value, so section 2 reports roughly zero error. That crossing is a coincidence of box size, not
agreement, and enlarging the box moves away from their number rather than toward it. Section 2
is kept at pad = 7.5 anyway, because this script's job is the WELL and the dot row is context;
paying for a converged box on every run would make it too slow to use as a quick check.

The decisive experiment is their Table I, not their Table III
-------------------------------------------------------------
Table I is for QUANTUM WELLS, and a pseudomorphic well is analytic:

    exx = eyy = (a_substrate - a_layer) / a_layer ,   ezz = -2 (C12/C11) exx ,   shear = 0

No finite-element elasticity, no dot shape, no grid, no periodic box. So if we disagree on the
WELL, the cause is in the deformation potentials or the band offsets -- the parameter set. If we
agree on the well but not the dot, the cause is in the 3D strain field or the geometry, and the
parameters are exonerated. That is a much sharper split than scanning parameters on the dot,
where every effect is entangled.

Their Table I, substrate InAs, well InSb:  conduction 0.645, valence 0.251 eV
Their Table III, substrate InAs, dot InSb: conduction 0.800, valence 0.127 eV
(both on a scale where the unstrained InSb valence edge is zero; ours is the unstrained InAs
valence edge, and E(theirs) = E(ours) - 0.590)

Then, and only if the well disagrees, the parameter sensitivity
---------------------------------------------------------------
The strain-induced gap opening is, in this convention,

    gap = Eg0 + (a_c + a_v) Tr(eps) - (shear raising of the valence edge)

so a_c and a_v enter only through their SUM, multiplied by a hydrostatic strain of about -0.06
in the well. A 1 eV error in (a_c + a_v) is therefore worth ~60 meV of gap. The scan below
reports d(gap)/d(parameter) for each of a_c, a_v, b and d, and solves for the value each would
have to take, on its own, to reconcile us with their table -- which is the useful form, because
it says immediately whether the required value is plausible or absurd.

CITATION NOTE. This script deliberately quotes no literature values for the deformation
potentials; it reports what our parameters ARE and what they WOULD HAVE TO BE to reconcile us
with their table. When it was written that comparison was left to a human with the review open,
because the review had not been fetched. It has since been done: `bandparameters_vurgaftman2001
.xlsx` carries the review's own tables, `materials.py` is transcribed from it, and the scan below
now solves for a_c + a_v = -7.30 eV -- which is exactly what the table gives. The scan is kept
because it is the thing that localised the error in the first place, and because it stays honest
if a parameter is changed again.

Run:  python scripts/insb_pryor_pistol_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import strain_fourier as sf
import kp_pryor as kp
import materials_sb as ms
import ingasb_dot as ig

PP_OFFSET = -0.590

#: Our box-converged dot gap, eV. Measured by `insb_box_convergence.py` at h = 0.4 nm, pad = 30
#: nm (200x200x163 sites, 25% lateral / 7% vertical fill), where successive padding steps have
#: fallen to -0.9 meV. Quoted here rather than recomputed because a converged box is a 7-minute
#: solve and this script is meant to be a quick check; section 2 runs a tight box instead and
#: says so.
CONVERGED_DOT_GAP = 0.6444

PP_WELL = dict(cb=0.645, vb=0.251)      # Table I,  InAs substrate, InSb well
PP_DOT = dict(cb=0.800, vb=0.127)       # Table III, InAs substrate, InSb dot


def pseudomorphic_strain(layer, substrate):
    """Analytic biaxial strain of a layer grown pseudomorphically on a (001) substrate."""
    eps_par = (substrate['a0'] - layer['a0']) / layer['a0']
    ezz = -2.0 * (layer['C12'] / layer['C11']) * eps_par
    return sf.StrainTensor(*[np.array([eps_par]), np.array([eps_par]), np.array([ezz]),
                             np.array([0.0]), np.array([0.0]), np.array([0.0])])


def well_edges(dot=None, matrix=None, **override):
    """k=0 band edges of a pseudomorphic InSb well on InAs, on our energy zero."""
    dot = dict(ms.SB_MATERIALS['InSb'] if dot is None else dot)
    matrix = ms.SB_MATERIALS['InAs'] if matrix is None else matrix
    dot.update(override)
    st = pseudomorphic_strain(dot, matrix)
    tr = float(st.exx[0] + st.eyy[0] + st.ezz[0])
    Ev0 = np.array([dot['VBO'] - matrix['VBO']])
    e = kp.local_band_edges(st, Ev=Ev0, Ec=Ev0 + dot['Eg'],
                            delta_so=np.array([dot['delta_so']]),
                            a_c=np.array([dot['a_c']]), a_v=np.array([dot['a_v']]),
                            b=np.array([dot['b']]), d=np.array([dot['d']]))
    return float(e['cb'][0]), float(e['v1'][0]), tr


def show(label, cb, vb, ref):
    """Print our (cb, vb) beside a Pryor-Pistol reference, on their scale."""
    print(f"  {label:<34}{cb+PP_OFFSET:>10.3f}{vb+PP_OFFSET:>10.3f}{cb-vb:>10.3f}"
          f"{(cb-vb)-(ref['cb']-ref['vb']):>+11.3f}")


if __name__ == '__main__':
    print(__doc__.split('Run:')[0])
    ms.set_gap_source('varshni', T=0.0)      # match their stated T = 0 K throughout
    print(f"gaps: 0 K Varshni -> InAs {ms.SB_MATERIALS['InAs']['Eg']:.3f}, "
          f"InSb {ms.SB_MATERIALS['InSb']['Eg']:.3f} eV")
    InSb, InAs = ms.SB_MATERIALS['InSb'], ms.SB_MATERIALS['InAs']

    # ---------------------------------------------------------------- 1. the well
    print(f"\n{'='*90}\n1. PSEUDOMORPHIC WELL -- analytic, no elasticity solve, no geometry"
          f"\n{'='*90}")
    cb_w, vb_w, tr_w = well_edges()
    print(f"  biaxial strain: exx = eyy = {(InAs['a0']-InSb['a0'])/InSb['a0']:+.5f}, "
          f"Tr(eps) = {tr_w:+.5f}")
    print(f"\n  {'':<34}{'CB(theirs)':>10}{'VB(theirs)':>10}{'gap':>10}{'gap err':>11}")
    show('this work', cb_w, vb_w, PP_WELL)
    print(f"  {'Pryor & Pistol Table I':<34}{PP_WELL['cb']:>10.3f}{PP_WELL['vb']:>10.3f}"
          f"{PP_WELL['cb']-PP_WELL['vb']:>10.3f}")

    # ---------------------------------------------------------------- 2. the dot
    print(f"\n{'='*90}\n2. THE DOT -- full FD elasticity, spherical cap at h/d = 1/4"
          f"\n{'='*90}")
    env = ig.build(ig.spherical_lens(10.0, 5.0), 1.0, h=0.5, pad=7.5, verbose=False)
    e = ig.valence_edge(env)
    m = env['mask']
    g = env['strain'].at(m)
    cb_d, vb_d = float(e['cb'][m].mean()), float(e['v1'][m].mean())
    print(f"  <Tr eps> in the dot = {g['exx']+g['eyy']+g['ezz']:+.5f}  "
          f"(well: {tr_w:+.5f} -- the dot relaxes more, as it must)")
    print(f"  NOT BOX CONVERGED: pad = 7.5 nm is {2*10.0/(m.shape[0]*env['h'])*100:.0f}% lateral "
          f"fill, and our gap curve crosses\n  their value near there. The converged gap is "
          f"{CONVERGED_DOT_GAP:.3f} eV ({(CONVERGED_DOT_GAP-(PP_DOT['cb']-PP_DOT['vb']))*1e3:+.0f}"
          f" meV), from `insb_box_convergence.py`.")
    print(f"\n  {'':<34}{'CB(theirs)':>10}{'VB(theirs)':>10}{'gap':>10}{'gap err':>11}")
    show('this work', cb_d, vb_d, PP_DOT)
    print(f"  {'Pryor & Pistol Table III':<34}{PP_DOT['cb']:>10.3f}{PP_DOT['vb']:>10.3f}"
          f"{PP_DOT['cb']-PP_DOT['vb']:>10.3f}")

    err_w = (cb_w - vb_w) - (PP_WELL['cb'] - PP_WELL['vb'])
    err_d = (cb_d - vb_d) - (PP_DOT['cb'] - PP_DOT['vb'])
    err_c = CONVERGED_DOT_GAP - (PP_DOT['cb'] - PP_DOT['vb'])
    print(f"\n  VERDICT")
    print(f"    gap error, well            : {err_w*1e3:+7.0f} meV")
    print(f"    gap error, dot at pad 7.5  : {err_d*1e3:+7.0f} meV  <- coincidence, see above")
    print(f"    gap error, dot converged   : {err_c*1e3:+7.0f} meV  <- the real number")
    if abs(err_w) > 0.04:
        print("    The WELL already disagrees, and the well involves no elasticity solve, no")
        print("    shape and no grid. So the cause is the PARAMETER SET, not the machinery.")
    else:
        print("    The well AGREES, to a precision no fitted parameter could fake -- it is a")
        print("    closed form, so nothing numerical is flattering it. The parameter set is")
        print("    therefore exonerated, and the residual dot disagreement lives in the 3D")
        print("    strain field or the dot geometry. It is still unexplained.")

    # ---------------------------------------------------------------- 3. sensitivity
    print(f"\n{'='*90}\n3. WHICH PARAMETER? one-at-a-time sensitivity, on the analytic well"
          f"\n{'='*90}")
    target = PP_WELL['cb'] - PP_WELL['vb']
    print(f"  target well gap = {target:.3f} eV;  ours = {cb_w-vb_w:.3f} eV "
          f"({err_w*1e3:+.0f} meV)\n")
    print(f"  {'param':>6}{'ours':>9}{'d(gap)/dp':>12}{'needed alone':>15}{'shift':>10}"
          f"   plausible?")
    for key in ('a_c', 'a_v', 'b', 'd'):
        p0 = InSb[key]
        step = 0.25
        cb_p, vb_p, _ = well_edges(**{key: p0 + step})
        cb_m, vb_m, _ = well_edges(**{key: p0 - step})
        slope = ((cb_p - vb_p) - (cb_m - vb_m)) / (2 * step)
        if abs(slope) < 1e-6:
            print(f"  {key:>6}{p0:>9.2f}{slope:>12.4f}{'--':>15}{'--':>10}"
                  f"   no effect on the gap")
            continue
        needed = p0 - err_w / slope
        print(f"  {key:>6}{p0:>9.2f}{slope:>12.4f}{needed:>15.2f}{needed-p0:>+10.2f}"
              f"   {'yes' if abs(needed-p0) < 2.0 else 'a big change'}")

    print(f"\n  a_c and a_v enter the gap only through their SUM in this convention, so the")
    print(f"  two rows above are the same lever seen twice. Ours: a_c + a_v = "
          f"{InSb['a_c']+InSb['a_v']:+.2f} eV.")
    slope_ac = (well_edges(a_c=InSb['a_c'] + 0.25)[0] - well_edges(a_c=InSb['a_c'] + 0.25)[1]
                - (well_edges(a_c=InSb['a_c'] - 0.25)[0]
                   - well_edges(a_c=InSb['a_c'] - 0.25)[1])) / 0.5
    if abs(slope_ac) > 1e-6:
        need_sum = (InSb['a_c'] + InSb['a_v']) - err_w / slope_ac
        print(f"  To match their well, a_c + a_v would have to be {need_sum:+.2f} eV "
              f"({need_sum - (InSb['a_c']+InSb['a_v']):+.2f} eV).")
    print("\n  See the citation note in the docstring: check the required value against")
    print("  Vurgaftman et al. (2001) directly rather than trusting a recalled number.")

    ms.set_gap_source('database')
