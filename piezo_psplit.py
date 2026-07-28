"""Multiband p-doublet splitting from the piezoelectric potential.

The hole levels come in exact Kramers pairs, so the states are E[0]=E[1] (s), E[2]=E[3] (p),
E[4]=E[5] (p). The C4v -> C2v splitting to measure is therefore E[2] vs E[4], NOT E[1] vs E[2]
(which is the s-p gap). Kramers degeneracy is a consequence of time-reversal symmetry for a
half-integer-spin system in zero magnetic field -- H. A. Kramers (1930); see any solid-state
text, e.g. the time-reversal discussion in C. Kittel, "Quantum Theory of Solids". It is used
here as a solver check: it is reproduced to ~1e-13 meV, which certifies convergence.

Unlike the one-band case, the multiband p doublet is split BEFORE any piezoelectric field,
because the k.p valence structure plus the Bir-Pikus shear terms already make [110] and [1-10]
inequivalent in zincblende. References as in piezo_states.py.
"""
import sys, time
sys.path.insert(0, r'C:\Users\molly\code\qd3Dsolver')
import numpy as np
import qdsolver_core as qd
import strain_fourier as sf
import piezoelectric as pz
import kp_confined as kpc
import kp_pryor as kp
import pryor1998 as pr
import eigensolvers as eig

np.set_printoptions(suppress=True, precision=3)
dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
C = (matrix['C11'], matrix['C12'], matrix['C44'])
EPS_R, BASE = matrix['eps_R'], 14.0

h, pad = 1.0, 8.0
nx = int(round(2 * (BASE / 2 + pad) / h)) // 2 * 2 + 1
kz0 = int(round(pad / h))
nz = kz0 + int(round((BASE / 2 + pad) / h)) + 1
cx = qd.centered_axis(nx, h)
cz = (np.arange(nz) - kz0) * h
X, Y, Z = np.meshgrid(cx, cx, cz, indexing='ij')
pyr = qd.pyramid_mask(X, Y, Z, BASE)
assert qd.mirror_asymmetry(pyr, 0) == 0 and qd.mirror_asymmetry(pyr, 1) == 0

st = sf.solve_strain(pyr, eps_star, *C, h)
phi = pz.potential(st, np.where(pyr, dot['e14'], matrix['e14']), EPS_R, h)
ops = kpc.GridOperators(X.shape, h, periodic=False)
V_e0, V_h0, _, _ = pr.band_edge_fields(pyr, st.trace)
print(f"grid {X.shape}, piezo phi {phi.min()*1e3:+.2f} .. {phi.max()*1e3:+.2f} meV")
print("hole levels are Kramers pairs; the C4v->C2v signature is E[2] vs E[4]\n")

for nb in (4, 6):
    print(f"  {nb}-band")
    for label, dphi in (("no piezo", 0.0), ("piezo", phi)):
        f = kp.material_fields(pyr, dot, matrix, V_h0 + dphi, V_e0 - dphi, n_bands=nb)
        H = kp.confined_hamiltonian(ops, f, n_bands=nb, hole_convention=True, strain=st)
        t0 = time.time()
        E, _, _ = eig.solve_lowest(H, k=8, tol=1e-9, maxiter=8000)
        lev = -E * 1e3
        kram = max(abs(lev[0] - lev[1]), abs(lev[2] - lev[3]), abs(lev[4] - lev[5]))
        print(f"    {label:>9s}  levels {np.round(lev[:6],3)}")
        print(f"    {'':>9s}  s-p gap {lev[0]-lev[2]:7.3f} meV   "
              f"p-doublet splitting {abs(lev[2]-lev[4]):7.3f} meV   "
              f"Kramers residual {kram:.2e} meV   ({time.time()-t0:.0f}s)")
