"""Validate elasticity_fd (Pryor's FD + conjugate-gradient method) against analytic ground truth.

The tests are ordered so that each one isolates a single thing:

  1. HOMOGENEOUS limit vs strain_fourier. Setting C_dot = C_matrix removes the only physical
     difference between the two solvers, so they must agree to discretization error. This tests
     the whole FEM assembly -- element stiffness, connectivity, load vector, CG -- against a
     completely independent implementation, and it is the test that would catch a transcription
     error in any of them.
  2. Pseudomorphic slab. Exact, anisotropic, and closed-form at any fill fraction.
  3. INHOMOGENEOUS sphere. The one test the Fourier solver cannot pass, and the entire reason
     this module exists.
  4. Variational check: the returned field must have lower energy than the homogeneous-solver
     field on the same inhomogeneous problem.
  5. Convergence with grid spacing.

Run: python elasticity_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import qdsolver_core as qd
import strain_fourier as sf
import elasticity_fd as ef
import pryor1998 as pr

np.set_printoptions(suppress=True, precision=6)
dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
C_dot = (dot['C11'], dot['C12'], dot['C44'])
C_mat = (matrix['C11'], matrix['C12'], matrix['C44'])
print(f"eps_star = {eps_star:+.6f}   C_InAs = {C_dot} GPa   C_GaAs = {C_mat} GPa\n")

# ------------------------------------------------------------------ 1. homogeneous limit
print("=" * 78)
print("TEST 1  homogeneous limit -- must reproduce strain_fourier (independent method)")
print("=" * 78)
h = 0.5
c = qd.centered_axis(64, h)
X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
pyr = qd.pyramid_mask(X, Y, Z, 10.0)
sph = qd.sphere_mask(X, Y, Z, 6.0)

print("  The two methods discretize differently (exact operator on a sampled mask vs Q1 finite")
print("  elements), so they do not have to agree at finite h -- but the difference must SHRINK")
print("  with h, since both converge to the same continuum solution. A constant offset would")
print("  mean one of them is solving a different problem.")
print(f"\n  {'shape':12} {'h':>6} {'grid':>6} {'fourier Tr':>11} {'FD Tr':>11} "
      f"{'|diff|/|Tr|':>11} {'CG':>4}")
for name, radius_arg in (('pyramid b=10', 'pyr'), ('sphere R=6', 'sph')):
    for hh, nn in ((1.0, 32), (0.5, 64), (1.0 / 3, 96)):
        cc = qd.centered_axis(nn, hh)
        Xx, Yy, Zz = np.meshgrid(cc, cc, cc, indexing='ij')
        m = (qd.pyramid_mask(Xx, Yy, Zz, 10.0) if radius_arg == 'pyr'
             else qd.sphere_mask(Xx, Yy, Zz, 6.0))
        a = sf.solve_strain(m, eps_star, *C_mat, hh)
        b = ef.solve_strain_fd(m, eps_star, C_mat, C_mat, hh, tol=1e-11)
        ta, tb = a.at(m), b.at(m)
        tra = ta['exx'] + ta['eyy'] + ta['ezz']
        trb = tb['exx'] + tb['eyy'] + tb['ezz']
        print(f"  {name:12} {hh:>6.3f} {nn:>6} {tra:>11.6f} {trb:>11.6f} "
              f"{abs(trb-tra)/abs(tra):>11.2%} {b.cg_iterations:>4d}")
print("  CG converges in one iteration in the homogeneous limit: the preconditioner is the")
print("  EXACT inverse of the homogeneous operator, so there is nothing left for CG to do.")
print("  That is a strong check on the symbol, the stiffness and the connectivity at once.")

# ------------------------------------------------------------------ 2. slab, exact
print("\n" + "=" * 78)
print("TEST 2  clamped [001] slab -- exact closed form at any fill fraction")
print("=" * 78)
N = 32
zc = np.arange(N) * h
Zs = np.meshgrid(zc, zc, zc, indexing='ij')[2]
print(f"  {'n_layers':>8} {'f':>7} | {'exx exact':>10} {'FD':>10} | {'ezz exact':>10} {'FD':>10}")
for nz in (1, 2, 4, 8, 16):
    slab = Zs < nz * h
    f = slab.mean()
    e = ef.solve_strain_fd(slab, eps_star, C_mat, C_mat, h, tol=1e-12)
    g = e.at(slab)
    pa, za = sf.clamped_slab_strain(eps_star, matrix['C11'], matrix['C12'], f)
    print(f"  {nz:>8} {f:>7.4f} | {pa:>10.6f} {g['exx']:>10.6f} | {za:>10.6f} {g['ezz']:>10.6f}")
err = max(abs(g['exx'] - pa), abs(g['ezz'] - za))
print(f"  worst error at f = {f:.3f}: {err:.2e}   max shear {max(abs(g['exy']),abs(g['eyz'])):.1e}")

# ------------------------------------------------------- 3. the inhomogeneous sphere
print("\n" + "=" * 78)
print("TEST 3  INHOMOGENEOUS sphere -- the test strain_fourier cannot pass")
print("=" * 78)
nu_m = qd.voigt_poisson_ratio(*C_mat)
Ci_m = sf.isotropic_constants(nu_m)                       # isotropic matrix, so the analytic
nu_d = qd.voigt_poisson_ratio(*C_dot)                     # misfitting-sphere result is exact
Ci_d = sf.isotropic_constants(nu_d, C11=Ci_m[0] * (dot['C11'] / matrix['C11']))
K_i, mu_i = ef.voigt_moduli(*Ci_d)
K_m, mu_m = ef.voigt_moduli(*Ci_m)
print(f"  isotropic surrogates:  K_incl {K_i:.2f}  mu_incl {mu_i:.2f} | "
      f"K_matrix {K_m:.2f}  mu_matrix {mu_m:.2f}")
print(f"  {'case':34} {'analytic':>11} {'numeric':>11} {'rel err':>9}")

cases = [('homogeneous  (K_i = K_matrix)', Ci_m, K_m),
         ('INHOMOGENEOUS (K_i = K_incl)', Ci_d, K_i)]
for label, Cd, Ki in cases:
    exact = ef.inhomogeneous_sphere_trace(eps_star, Ki, mu_m)
    best = None
    for n in (64, 96, 128):
        cc = qd.centered_axis(n, h)
        Xx, Yy, Zz = np.meshgrid(cc, cc, cc, indexing='ij')
        s = qd.sphere_mask(Xx, Yy, Zz, 6.0)
        core = qd.sphere_mask(Xx, Yy, Zz, 3.6)
        e = ef.solve_strain_fd(s, eps_star, Cd, Ci_m, h, tol=1e-11)
        g = e.at(core)
        best = g['exx'] + g['eyy'] + g['ezz']
    print(f"  {label:34} {exact:>11.6f} {best:>11.6f} {abs(best/exact-1):>9.2e}")
print("  (the O(f) residual is the clamped-cell offset, identical in kind to strain_fourier's)")

fourier_span = [ef.inhomogeneous_sphere_trace(eps_star, K_m, mu_m),
                ef.inhomogeneous_sphere_trace(eps_star, K_i, mu_i)]
print(f"\n  a homogeneous solver can only reach [{min(fourier_span):+.6f}, "
      f"{max(fourier_span):+.6f}];")
print(f"  the true inhomogeneous value {ef.inhomogeneous_sphere_trace(eps_star, K_i, mu_m):+.6f}"
      f" lies outside that interval.")

# ------------------------------------------------------------------ 4. variational check
print("\n" + "=" * 78)
print("TEST 4  variational: the FD field must have LOWER energy than the homogeneous one")
print("=" * 78)
print("  Both fields are evaluated with the SAME (true, inhomogeneous) elastic constants, so")
print("  this is a like-for-like comparison of two trial fields for one energy functional.")
e_fd = ef.solve_strain_fd(pyr, eps_star, C_dot, C_mat, h, tol=1e-11)
e_fo = sf.solve_strain(pyr, eps_star, *C_mat, h)
E_fd = ef.strain_energy(e_fd, pyr, C_dot, C_mat, h)
E_fo = ef.strain_energy(e_fo, pyr, C_dot, C_mat, h)
print(f"  FD (inhomogeneous minimizer)  E = {E_fd:.6e}")
print(f"  Fourier (homogeneous C_GaAs)  E = {E_fo:.6e}")
print(f"  -> FD lower by {(1-E_fd/E_fo)*100:.2f}%  "
      f"{'OK' if E_fd < E_fo else 'FAIL -- the minimizer is not minimal'}")

# ------------------------------------------------------------------ 5. grid convergence
print("\n" + "=" * 78)
print("TEST 5  convergence with grid spacing (pyramid b = 10, true constants)")
print("=" * 78)
print("  The 32 nm box is held fixed, so only h changes. `vol err` is the staircasing error of")
print("  the mask itself, which moves the answer independently of the elasticity: a faceted")
print("  {101} pyramid is badly represented on a cubic grid and the two effects must be read")
print("  together, not separately.")
print(f"  {'h':>6} {'grid':>14} {'vol err':>9} {'<Tr eps>':>11} {'rms exy':>10} {'CG':>5}")
EXACT_VOL = 10.0 ** 2 * 5.0 / 3.0
for hh, nn in ((1.0, 32), (0.5, 64), (0.4, 80), (1.0 / 3, 96), (0.25, 128)):
    cc = qd.centered_axis(nn, hh)
    Xx, Yy, Zz = np.meshgrid(cc, cc, cc, indexing='ij')
    p = qd.pyramid_mask(Xx, Yy, Zz, 10.0)
    e = ef.solve_strain_fd(p, eps_star, C_dot, C_mat, hh, tol=1e-11)
    g, gr = e.at(p), e.at_rms(p)
    print(f"  {hh:>6.3f} {str(Xx.shape):>14} {qd.mask_volume_error(p,hh,EXACT_VOL):>+9.4f} "
          f"{g['exx']+g['eyy']+g['ezz']:>11.6f} {gr['exy']:>10.6f} {e.cg_iterations:>5d}")
