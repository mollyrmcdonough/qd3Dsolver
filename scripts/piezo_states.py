"""Effect of the piezoelectric potential on confined states, on verified-clean grids.

The p-doublet splitting measured here is the quantity pryor_benchmark.ipynb predicted but could
not compute: a square-based pyramid is C4v, so p_x and p_y are degenerate until something breaks
that symmetry. Pryor, Phys. Rev. B 57, 7190 (1998) reports 2-6 meV for this dot.

References for the machinery: piezoelectric potential -- M. Grundmann, O. Stier and D. Bimberg,
Phys. Rev. B 52, 11969 (1995), for this exact geometry; strain -- A. D. Andreev, J. R. Downes,
D. A. Faux and E. P. O'Reilly, J. Appl. Phys. 86, 297 (1999); Bir-Pikus coupling -- G. L. Bir
and G. E. Pikus (Wiley, 1974); eigensolver -- A. V. Knyazev, SIAM J. Sci. Comput. 23, 517
(2001).

The grid is built with qdsolver_core.centered_axis and its mirror symmetry ASSERTED before any
solve, because an asymmetric grid manufactures exactly the splitting being measured.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import qdsolver_core as qd
import strain_fourier as sf
import piezoelectric as pz
import kp_confined as kpc
import kp_pryor as kp
import pryor1998 as pr
import eigensolvers as eig

np.set_printoptions(suppress=True, precision=6)
dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
C = (matrix['C11'], matrix['C12'], matrix['C44'])
EPS_R, BASE, GAAS_CB = matrix['eps_R'], 14.0, matrix['Eg']
EXACT_VOL = BASE ** 2 * (BASE / 2) / 3


def clean_grid(h, pad):
    nx = int(round(2 * (BASE / 2 + pad) / h)) // 2 * 2 + 1
    kz0 = int(round(pad / h))
    nz = kz0 + int(round((BASE / 2 + pad) / h)) + 1
    cx = qd.centered_axis(nx, h)
    cz = (np.arange(nz) - kz0) * h
    X, Y, Z = np.meshgrid(cx, cx, cz, indexing='ij')
    pyr = qd.pyramid_mask(X, Y, Z, BASE)
    assert qd.mirror_asymmetry(pyr, 0) == 0 and qd.mirror_asymmetry(pyr, 1) == 0
    return X, Y, Z, pyr


print("=" * 78)
print("ONE-BAND ELECTRON with piezoelectric potential  (h = 0.5, verified-clean grid)")
print("=" * 78)
print("The p-state splitting is the signature: a C4v pyramid has degenerate p_x/p_y, and the")
print("piezoelectric potential is the term that reduces the symmetry to C2v and splits them.")
for h, pad in ((0.5, 10.0),):
    X, Y, Z, pyr = clean_grid(h, pad)
    st = sf.solve_strain(pyr, eps_star, *C, h)
    e14f = np.where(pyr, dot['e14'], matrix['e14'])
    phi = pz.potential(st, e14f, EPS_R, h)
    print(f"\n  h={h} pad={pad}  grid {X.shape} = {X.size:,} pts, "
          f"volume err {qd.mask_volume_error(pyr,h,EXACT_VOL):+.3f}")
    print(f"    piezo potential: {phi.min()*1e3:+.2f} .. {phi.max()*1e3:+.2f} meV, "
          f"C4 antisymmetry {pz.c4_antisymmetry_error(phi):.1e}")
    for case in ('unstrained', 'strain_averaged'):
        V_e, V_h, m_e, m_h = pr.band_edge_fields(pyr, st.trace, mass_case=case)
        m = pr.PRYOR_ELECTRON_MASSES[case]['dot']
        for label, V in (("no piezo", V_e), ("piezo", V_e - phi)):
            # electron charge is -e, so its potential energy is -e*phi = -phi in eV
            E, _, _ = eig.solve_lowest(qd.build_hamiltonian(m_e, V, h), k=6,
                                       tol=1e-9, maxiter=8000)
            nb = int((E < GAAS_CB).sum())
            sp = (E[2] - E[1]) * 1e3 if nb > 2 else float('nan')
            print(f"    m={m:.3f} {label:>9s}  E0={E[0]:.5f} eV  binding "
                  f"{(GAAS_CB-E[0])*1e3:6.1f} meV  bound={nb}  "
                  f"p-splitting {sp:8.3f} meV")

print("\n" + "=" * 78)
print("MULTIBAND HOLES with piezoelectric potential  (h = 1.0, verified-clean grid)")
print("=" * 78)
h, pad = 1.0, 8.0
X, Y, Z, pyr = clean_grid(h, pad)
st = sf.solve_strain(pyr, eps_star, *C, h)
e14f = np.where(pyr, dot['e14'], matrix['e14'])
phi = pz.potential(st, e14f, EPS_R, h)
ops = kpc.GridOperators(X.shape, h, periodic=False)
print(f"  grid {X.shape} = {X.size:,} pts, volume err "
      f"{qd.mask_volume_error(pyr,h,EXACT_VOL):+.3f}")
print(f"  piezo potential {phi.min()*1e3:+.2f} .. {phi.max()*1e3:+.2f} meV, "
      f"C4 antisymmetry {pz.c4_antisymmetry_error(phi):.1e}")
V_e0, V_h0, _, _ = pr.band_edge_fields(pyr, st.trace)
for label, dphi in (("no piezo", 0.0), ("piezo", phi)):
    # hole charge is +e: potential energy +e*phi = +phi; electron gets -phi
    V_e, V_h = V_e0 - dphi, V_h0 + dphi
    print(f"  --- {label}")
    for nb in (4, 6):
        f = kp.material_fields(pyr, dot, matrix, V_h, V_e, n_bands=nb)
        H = kp.confined_hamiltonian(ops, f, n_bands=nb, hole_convention=True, strain=st)
        t0 = time.time()
        E, _, _ = eig.solve_lowest(H, k=6, tol=1e-8, maxiter=8000)
        lev = -E * 1e3
        print(f"      {nb}-band hole levels (meV): {np.round(lev[:4],2)}   "
              f"1-2 splitting {abs(lev[1]-lev[2]):.2f} meV  ({time.time()-t0:.0f}s)")
