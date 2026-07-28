"""Pryor b=14 nm benchmark, re-run with the corrected elastic trace + Bir-Pikus shear.

Benchmark target: C. Pryor, "Eight-band calculations of strained InAs/GaAs quantum dots
compared with one-, four-, and six-band approximations", Phys. Rev. B 57, 7190 (1998),
preprint arXiv:cond-mat/9710304. The reference values compared against are collected in
pryor1998.BENCHMARK_B14, where each carries the section or figure it was read from.

Physics used here, all cited at their point of definition:
  - strain_fourier.solve_strain    -- Andreev, Downes, Faux & O'Reilly, J. Appl. Phys. 86,
                                      297 (1999); Eshelby, Proc. R. Soc. A 241, 376 (1957)
  - kp_pryor.confined_hamiltonian  -- Pryor 1998's eight-band matrix; Luttinger & Kohn,
                                      Phys. Rev. 97, 869 (1955)
  - kp_pryor.bir_pikus_terms       -- Bir & Pikus, "Symmetry and Strain-Induced Effects in
                                      Semiconductors" (Wiley, 1974)
  - eigensolvers                   -- LOBPCG, Knyazev, SIAM J. Sci. Comput. 23, 517 (2001);
                                      folded spectrum, Wang & Zunger, J. Chem. Phys. 100,
                                      2394 (1994)

NOTE: the one-band section of this script uses h = 0.7, which was later found to produce an
asymmetric and 14%-undersized pyramid (see qdsolver_core.centered_axis). Its one-band numbers
are superseded by piezo_states.py at h = 0.5; the multiband grids (h = 1.4 pad 8.4, h = 1.0
pad 8.0) audit clean and stand.
"""
import sys, time
sys.path.insert(0, r'C:\Users\molly\code\qd3Dsolver')
import numpy as np
import qdsolver_core as qd
import strain_fourier as sf
import kp_confined as kpc
import kp_pryor as kp
import pryor1998 as pr
import eigensolvers as eig

np.set_printoptions(suppress=True)
dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
GAAS_CB, BASE = matrix['Eg'], 14.0
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
nu = qd.voigt_poisson_ratio(matrix['C11'], matrix['C12'], matrix['C44'])
C = (matrix['C11'], matrix['C12'], matrix['C44'])


def grid(h, pad):
    cx = np.arange(-(BASE / 2 + pad), BASE / 2 + pad + 1e-9, h)
    cz = np.arange(-pad, BASE / 2 + pad + 1e-9, h)
    X, Y, Z = np.meshgrid(cx, cx, cz, indexing='ij')
    return X, Y, Z, qd.pyramid_mask(X, Y, Z, BASE)


print("=" * 78)
print("ONE-BAND ELECTRON  (Pryor: E0 ~ 1.41 eV at m=0.023 with 1 bound state;")
print("                    E0 ~ 1.38 eV, E1-E0 ~ 110 meV, extra binding ~30 meV at m=0.040)")
print("=" * 78)
for h, pad in ((0.7, 10.5), (0.7, 14.0)):
    X, Y, Z, pyr = grid(h, pad)
    tr_new = sf.solve_strain(pyr, eps_star, *C, h).trace
    tr_old, _ = qd.trace_strain_from_mask(pyr, eps_star, nu)
    print(f"\n  h={h} pad={pad}  grid {X.shape} = {X.size:,} pts")
    print(f"    trace in dot: corrected {tr_new[pyr].mean():+.5f}   "
          f"old hydrostatic-only {tr_old[pyr].mean():+.5f}")
    for label, tr in (("corrected", tr_new), ("old", tr_old)):
        for case in ('unstrained', 'strain_averaged'):
            V_e, V_h, m_e, m_h = pr.band_edge_fields(pyr, tr, mass_case=case)
            depth = (GAAS_CB - V_e[pyr].mean()) * 1e3
            H = qd.build_hamiltonian(m_e, V_e, h)
            E, _, _ = eig.solve_lowest(H, k=6, tol=1e-9, maxiter=6000)
            nb = int((E < GAAS_CB).sum())
            m = pr.PRYOR_ELECTRON_MASSES[case]['dot']
            extra = ""
            if nb > 1:
                extra = f"  E1-E0={((E[1]-E[0])*1e3):7.1f} meV"
            print(f"    {label:>9s} m={m:.3f}  well {depth:6.1f} meV  E0={E[0]:.4f} eV  "
                  f"binding {(GAAS_CB-E[0])*1e3:6.1f} meV  bound={nb}{extra}")

print("\n" + "=" * 78)
print("MULTIBAND  (Pryor: 4-band hole ~195, 6-band ~235, 8-band hole ~195 meV;")
print("            8-band electron ~1.35 eV)")
print("=" * 78)
for h, pad in ((1.4, 8.4), (1.0, 8.0)):
    X, Y, Z, pyr = grid(h, pad)
    st = sf.solve_strain(pyr, eps_star, *C, h)
    tr_old, _ = qd.trace_strain_from_mask(pyr, eps_star, nu)
    ops = kpc.GridOperators(X.shape, h, periodic=False)
    d, dr = st.at(pyr), st.at_rms(pyr)
    print(f"\n  h={h} pad={pad}  grid {X.shape} = {X.size:,} pts, {8*X.size:,} unknowns (8-band)")
    print(f"    strain in dot: trace {d['exx']+d['eyy']+d['ezz']:+.5f}  "
          f"biaxial {d['exx']+d['eyy']-2*d['ezz']:+.5f}  rms ezx {dr['ezx']:.5f}")

    for label, tr, strain in (("old (hydro only)", tr_old, None),
                              ("corrected+shear", st.trace, st)):
        V_e, V_h, _, _ = pr.band_edge_fields(pyr, tr)
        print(f"    --- {label}")
        for nb in (4, 6):
            f = kp.material_fields(pyr, dot, matrix, V_h, V_e, n_bands=nb)
            H = kp.confined_hamiltonian(ops, f, n_bands=nb, hole_convention=True, strain=strain)
            t0 = time.time()
            E, _, _ = eig.solve_lowest(H, k=4, tol=1e-8, maxiter=6000)
            print(f"        {nb}-band hole ground state {(-E[0])*1e3:7.1f} meV "
                  f"above GaAs VB edge   ({time.time()-t0:.0f}s)")
        f8 = kp.material_fields(pyr, dot, matrix, V_h, V_e, n_bands=8)
        H8 = kp.confined_hamiltonian(ops, f8, n_bands=8, strain=strain)
        t0 = time.time()
        Ee, _, _ = eig.solve_interior(H8, k=3, sigma=float(V_e[pyr].min()), tol=1e-7, maxiter=8000)
        Eh, _, _ = eig.solve_interior(H8, k=3, sigma=float(V_h[pyr].max()), tol=1e-7, maxiter=8000)
        print(f"        8-band electron {Ee[0]:.4f} eV   8-band hole {Eh[0]*1e3:7.1f} meV "
              f"({time.time()-t0:.0f}s)")
