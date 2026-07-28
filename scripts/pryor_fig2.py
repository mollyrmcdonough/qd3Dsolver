"""Reproduce Pryor's Fig. 2: band structure from the LOCAL value of the strain.

Source: C. Pryor, Phys. Rev. B 57, 7190 (1998), preprint arXiv:cond-mat/9710304. Its Fig. 2
caption reads, verbatim:

    FIG. 2. Band structure based on the local value of the strain. (a) Bands along the 001
    direction, through the center of the island. (b) Bands Along the 100 direction, through
    the base of the island.

and its Sec. IV specifies the construction and the size:

    "Fig. 2 shows the band energies computed using the local value of the strain (i.e. the
     eigenvalues of Eq. 4 with k = 0). Since the coupling between conduction and valence bands
     is proportional to k, for k = 0 the model reduces to a six-band model with a decoupled
     conduction band. The bands are shown for an island with b = 10 nm."

NOTE b = 10 nm here, not the b = 14 nm used for the bound-state benchmarks elsewhere in this
package.

Two quantitative statements in that same section are used below as pass/fail checks rather
than as commentary, since they are the only numbers Pryor attaches to this figure:

    "The conduction band still has a potential well 0.4 eV deep at the base of the island,
     tapering to 0.27 eV at the tip."

Where this calculation differs from his
--------------------------------------
Pryor relaxes the strain with a conjugate-gradient minimization on the grid using each
material's own elastic constants; strain_fourier.solve_strain is a Fourier solve that requires
HOMOGENEOUS constants (GaAs is used here). InAs is ~30% softer, so the dot in his calculation
relaxes more than it does here and the interior strain magnitudes should come out somewhat
larger in ours. He also includes the piezoelectric potential in the full Hamiltonian; Fig. 2 is
defined as the eigenvalues of H_s, so no piezo enters it and none is added here.

Run: python pryor_fig2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qdsolver_core as qd
import strain_fourier as sf
import kp_pryor as kp
import pryor1998 as pr

BASE = 10.0                       # nm, Pryor's Sec. IV value for this figure
H = 0.5                           # nm grid spacing
N = 129                           # odd, so z = 0 (the island base) is exactly a grid point

dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])


def build():
    """Grid, mask and strain, with the geometry guards asserted before anything uses them."""
    c = qd.centered_axis(N, H)
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    pyr = qd.pyramid_mask(X, Y, Z, BASE)

    assert qd.mirror_asymmetry(pyr, 0) == 0, "mask not symmetric in x"
    assert qd.mirror_asymmetry(pyr, 1) == 0, "mask not symmetric in y"
    vol_err = qd.mask_volume_error(pyr, H, BASE ** 2 * (BASE / 2) / 3)

    strain = sf.solve_strain(pyr, eps_star, matrix['C11'], matrix['C12'], matrix['C44'], H)
    return c, X, Y, Z, pyr, strain, vol_err


def bands(strain, inside, index):
    """Local k=0 band energies along a 1D cut, on Pryor's energy zero (unstrained GaAs E_v)."""
    sub = sf.StrainTensor(*[getattr(strain, k)[index] for k in sf.StrainTensor.COMPONENTS])
    m = inside[index]

    def pick(key):
        return np.where(m, dot[key], matrix[key])

    Ev = np.where(m, dot['E_vbo'], matrix['E_vbo'])
    return kp.local_band_edges(sub, Ev=Ev, Ec=Ev + pick('Eg'), delta_so=pick('delta_so'),
                               a_c=pick('a_c'), a_v=pick('a_v'), b=pick('b'), d=pick('d'))


def main():
    c, X, Y, Z, pyr, strain, vol_err = build()
    i0 = N // 2                                    # index of the coordinate 0.0
    assert abs(c[i0]) < 1e-12
    print(f"grid {N}^3 at h={H} nm (span {N*H:.0f} nm), b={BASE} nm island")
    print(f"  mask volume error {vol_err:+.4f}, mirror asymmetry 0/0")
    print(f"  clamping residual (padding_report) {sf.padding_report(strain, pyr):.4f}")
    d_ = strain.at(pyr)
    print(f"  <Tr eps> in dot {d_['exx']+d_['eyy']+d_['ezz']:+.4f}")

    # ---- consistency check: the two ways of applying the hydrostatic shift must agree ----
    cut = (i0, i0, slice(None))
    b_a = bands(strain, pyr, cut)
    Ve_field, Vh_field, _, _ = pr.band_edge_fields(pyr, strain.trace)
    sub = sf.StrainTensor(*[getattr(strain, k)[cut] for k in sf.StrainTensor.COMPONENTS])
    b_pre = kp.local_band_edges(
        sub, Ev=Vh_field[cut], Ec=Ve_field[cut],
        delta_so=np.where(pyr[cut], dot['delta_so'], matrix['delta_so']),
        a_c=0.0, a_v=0.0, b=np.where(pyr[cut], dot['b'], matrix['b']),
        d=np.where(pyr[cut], dot['d'], matrix['d']), hydrostatic_applied=True)
    agree = max(np.abs(b_a[k] - b_pre[k]).max() for k in ('cb', 'v1', 'v2', 'v3'))
    print(f"  unstrained-edges vs band_edge_fields route agree to {agree:.2e} eV")
    print(f"  Kramers degeneracy residual {b_a['kramers']:.2e} eV")

    # ---- Pryor's two quoted numbers for this figure ----
    GaAs_cb = matrix['E_vbo'] + matrix['Eg']
    z = c
    in_axis = pyr[i0, i0, :]
    z_in = z[in_axis]
    depth = GaAs_cb - b_a['cb']
    print(f"\n  CB well depth along [001]: base(z=0) {depth[np.argmin(np.abs(z))]:.3f} eV "
          f"(Pryor 0.4), tip(z={z_in.max():.1f}) {depth[np.where(in_axis)[0][-1]]:.3f} eV "
          f"(Pryor 0.27)")

    v1 = b_a['v1']
    inside_idx = np.where(in_axis)[0]
    vin = v1[inside_idx]
    print(f"  valence edge along [001] inside the dot: base {vin[0]:.3f}, "
          f"min {vin.min():.3f} at z={z[inside_idx][np.argmin(vin)]:.1f}, "
          f"tip {vin[-1]:.3f} eV  -> peaks at base and tip with a dip between: "
          f"{'yes' if vin.min() < min(vin[0], vin[-1]) - 1e-3 else 'NO'}")

    ch = kp.local_band_character(
        sf.StrainTensor(*[getattr(strain, k)[cut] for k in sf.StrainTensor.COMPONENTS]),
        Ev=np.where(in_axis, dot['E_vbo'], matrix['E_vbo']),
        delta_so=np.where(in_axis, dot['delta_so'], matrix['delta_so']),
        a_v=np.where(in_axis, dot['a_v'], matrix['a_v']),
        b=np.where(in_axis, dot['b'], matrix['b']),
        d=np.where(in_axis, dot['d'], matrix['d']))
    hh_frac = ch[inside_idx, 0, 0]
    print(f"  v1 heavy-hole weight inside the dot runs {hh_frac.min():.2f} -> {hh_frac.max():.2f}"
          f"  (a crossing shows up as this sweeping across 0.5)")

    # ---- Pryor's qualitative claim about the barrier, checked over the whole grid ----
    # "From Fig. 2b we see that InAs islands have an elevation of the valence band above the
    #  island, but inside the island the valence-band edge is even higher. Hence we do not
    #  expect holes to be trapped in the barrier material."
    _, Vh_all, _, _ = pr.band_edge_fields(pyr, strain.trace)
    full = kp.local_band_edges(
        strain, Ev=np.where(pyr, dot['E_vbo'], matrix['E_vbo']),
        Ec=np.where(pyr, dot['E_vbo'] + dot['Eg'], matrix['E_vbo'] + matrix['Eg']),
        delta_so=np.where(pyr, dot['delta_so'], matrix['delta_so']),
        a_c=np.where(pyr, dot['a_c'], matrix['a_c']),
        a_v=np.where(pyr, dot['a_v'], matrix['a_v']),
        b=np.where(pyr, dot['b'], matrix['b']), d=np.where(pyr, dot['d'], matrix['d']))
    v1f = full['v1']
    print(f"\n  valence edge max: inside dot {v1f[pyr].max():.4f} eV, "
          f"in barrier {v1f[~pyr].max():.4f} eV -> holes trapped in barrier? "
          f"{'YES' if v1f[~pyr].max() > v1f[pyr].max() else 'no (Pryor agrees)'}")

    # ---- how accurate is kp.hole_sigma, which ignores R_eps and S_eps? ----
    # hole_sigma approximates the local valence edge as E_v - Q_eps, i.e. the HH diagonal
    # element alone. But unlike their kinetic counterparts R and S, the strain terms R_eps and
    # S_eps do NOT vanish at k = 0, so the true edge is the largest eigenvalue of the whole
    # block -- exactly what this figure computes. Quantify the gap, since sigma sitting BELOW
    # the target states is the failure mode that silently returns converged wrong eigenvalues.
    Qe, _, _ = kp.bir_pikus_terms(strain, np.where(pyr, dot['b'], matrix['b']),
                                  np.where(pyr, dot['d'], matrix['d']))
    old = float((Vh_all - np.real(Qe))[pyr].max())
    new = kp.hole_sigma(Vh_all, strain, np.where(pyr, dot['b'], matrix['b']),
                        np.where(pyr, dot['d'], matrix['d']),
                        np.where(pyr, dot['delta_so'], matrix['delta_so']), inside_mask=pyr)
    exact = float(v1f[pyr].max())
    print(f"  sigma for a hole solve: E_v - Q_eps alone {old:.4f} eV (the old approximation),"
          f" hole_sigma {new:.4f} eV, exact k=0 edge {exact:.4f} eV")
    print(f"    -> dropping R_eps/S_eps would place sigma {1e3*(exact-old):.1f} meV too LOW")

    # ---- how much of the gap against Pryor is the homogeneous-elasticity approximation? ----
    # He relaxes with each material's own constants; we must pick one set. GaAs (the matrix) is
    # the stiffer, so it under-relaxes the dot and should overstate the well depth; InAs brackets
    # it from the other side. If Pryor's numbers fall inside the bracket, the discrepancy is
    # this approximation rather than an error.
    print("\n  elastic-constant sensitivity (homogeneous solve must pick one set):")
    for label in ('GaAs', 'InAs'):
        m = pr.PRYOR_TABLE_I[label]
        s2 = sf.solve_strain(pyr, eps_star, m['C11'], m['C12'], m['C44'], H)
        bb = bands(s2, pyr, cut)
        dep = GaAs_cb - bb['cb']
        print(f"    C from {label}: <Tr eps>={s2.at(pyr)['exx']+s2.at(pyr)['eyy']+s2.at(pyr)['ezz']:+.4f}"
              f"  well depth base {dep[np.argmin(np.abs(z))]:.3f}  tip "
              f"{dep[np.where(in_axis)[0][-1]]:.3f} eV")
    print("    Pryor (inhomogeneous, his own relaxation):        base 0.400  tip 0.270 eV")

    # ---- the figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cuts = [((i0, i0, slice(None)), z, 'distance along 001  (nm)', (-15, 22)),
            ((slice(None), i0, i0), c, 'distance along 100  (nm)', (-25, 25))]
    titles = ['(a) [001] through the centre of the island',
              '(b) [100] through the base of the island']

    for ax, (cut_, axis_, xlabel, xlim), title in zip(axes, cuts, titles):
        bb = bands(strain, pyr, cut_)
        ax.plot(axis_, bb['cb'], 'k-', lw=1.3)
        ax.plot(axis_, bb['v1'], 'k-', lw=1.3)
        ax.plot(axis_, bb['v2'], 'k-', lw=0.9, alpha=0.75)
        ax.plot(axis_, bb['v3'], 'k-', lw=0.9, alpha=0.75)

        edges = np.flatnonzero(np.diff(pyr[cut_].astype(int)))
        for e in edges:
            ax.axvline((axis_[e] + axis_[e + 1]) / 2, color='0.75', lw=0.7, zorder=0)
        ax.axhline(0.0, color='0.85', lw=0.7, ls=':', zorder=0)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.48, 1.78)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('E  (eV)')
        ax.set_title(title, fontsize=9)

        lo, hi = (axis_[edges[0]], axis_[edges[-1]]) if len(edges) >= 2 else (0, axis_.max())
        mid = (lo + hi) / 2 if len(edges) >= 2 else hi / 2
        ax.text(mid, 1.62, 'InAs', ha='center', fontsize=8)
        ax.text(xlim[0] + 0.18 * (xlim[1] - xlim[0]), 1.62, 'GaAs', ha='center', fontsize=8)
        ax.text(xlim[1] - 0.18 * (xlim[1] - xlim[0]), 1.62, 'GaAs', ha='center', fontsize=8)

    fig.suptitle(f'Band structure from the local value of the strain, b = {BASE:.0f} nm '
                 f'(k = 0 eigenvalues of $H_s$) — cf. Pryor 1998 Fig. 2', fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pryor_fig2.png')
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
