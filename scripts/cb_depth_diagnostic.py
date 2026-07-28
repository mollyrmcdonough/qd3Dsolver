"""Why is our conduction well deeper than Pryor's, and why did the elastic-constant bracket fail?

Pryor's Sec. IV quotes, for the b = 10 nm island, a conduction well "0.4 eV deep at the base of
the island, tapering to 0.27 eV at the tip". pryor_fig2.py reproduces every qualitative feature
of his Fig. 2 but gives 0.468 / 0.332 eV. Swapping the homogeneous elastic constants between
GaAs and InAs moved the answer the WRONG way, which ruled out the obvious explanation.

This script tests the actual hypothesis: the error is controlled by the dot/matrix stiffness
CONTRAST, which a homogeneous solve cannot represent at all, and which swapping the (single)
constant set does not probe.

The clean statement comes from the classic misfitting-sphere result. For a spherical inclusion
with a dilatational eigenstrain eps_T in an infinite matrix, the elastic dilatation inside is

    Tr(eps) = -3 eps_T * 4 mu_m / (3 K_i + 4 mu_m)

-- the INCLUSION's bulk modulus K_i and the MATRIX's shear modulus mu_m, and nothing else. A
homogeneous solve is forced to take both from the same material, so it cannot land on the true
value no matter which material it picks. (References: J. D. Eshelby, Proc. R. Soc. London A 241,
376 (1957) for the inclusion/inhomogeneity problem; the misfitting-sphere form above is the
standard textbook special case.)

Run: python cb_depth_diagnostic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import qdsolver_core as qd
import strain_fourier as sf
import pryor1998 as pr

dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
eps_T = -eps_star
BASE, H, N = 10.0, 0.5, 129


def moduli(p):
    """Voigt-averaged bulk and shear moduli (GPa)."""
    K = (p['C11'] + 2 * p['C12']) / 3.0
    mu = (p['C11'] - p['C12'] + 3 * p['C44']) / 5.0
    return K, mu


def sphere_trace(K_i, mu_m):
    return -3.0 * eps_T * 4.0 * mu_m / (3.0 * K_i + 4.0 * mu_m)


K_in, mu_in = moduli(dot)
K_ga, mu_ga = moduli(matrix)
print(f"Voigt moduli (GPa):  InAs K={K_in:.1f} mu={mu_in:.1f}   GaAs K={K_ga:.1f} mu={mu_ga:.1f}")
print(f"stiffness contrast K_InAs/K_GaAs = {K_in/K_ga:.3f}\n")

print("=" * 78)
print("1. The misfitting-sphere formula, validated against solve_strain")
print("=" * 78)
print("  solve_strain is homogeneous, so it can only reproduce the two K_i = K_m rows.")
c = qd.centered_axis(96, 0.5)
X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
sph = qd.sphere_mask(X, Y, Z, 6.0)
core = qd.sphere_mask(X, Y, Z, 3.6)
for label, p in (('GaAs', matrix), ('InAs', dot)):
    K, mu = moduli(p)
    nu = qd.voigt_poisson_ratio(p['C11'], p['C12'], p['C44'])
    es = sf.solve_strain(sph, eps_star, *sf.isotropic_constants(nu), 0.5)
    g = es.at(core)
    num = g['exx'] + g['eyy'] + g['ezz']
    print(f"  homogeneous {label}: analytic {sphere_trace(K, mu):+.5f}   "
          f"solve_strain {num:+.5f}   (diff {abs(num-sphere_trace(K,mu)):.1e}, O(f) clamping)")
print(f"  INHOMOGENEOUS (K_i = InAs, mu_m = GaAs): {sphere_trace(K_in, mu_ga):+.5f}   "
      f"<- what Pryor solves")

f_ga = sphere_trace(K_ga, mu_ga)
f_in = sphere_trace(K_in, mu_in)
f_true = sphere_trace(K_in, mu_ga)
print(f"\n  homogeneous bracket spans [{f_in:+.5f}, {f_ga:+.5f}]")
print(f"  the true value {f_true:+.5f} lies OUTSIDE it, more compressive by "
      f"{f_true/f_ga:.3f}x than our GaAs choice.")
print("  -> the bracket test was uninformative: it varied overall stiffness (which barely")
print("     matters, since only ratios enter a homogeneous eigenstrain problem) instead of")
print("     the dot/matrix contrast (which is the entire effect).")

print("\n" + "=" * 78)
print("2. What trace do Pryor's two quoted well depths imply?")
print("=" * 78)
# depth = E_c(GaAs, unstrained) - [E_vbo + Eg + a_c Tr] = const - a_c Tr
#      -> Tr = (const - depth) / a_c
GaAs_CB = matrix['E_vbo'] + matrix['Eg']
const = GaAs_CB - (dot['E_vbo'] + dot['Eg'])
implied = lambda depth: (const - depth) / dot['a_c']
depth_of = lambda tr: const - dot['a_c'] * tr

cx = qd.centered_axis(N, H)
X, Y, Z = np.meshgrid(cx, cx, cx, indexing='ij')
pyr = qd.pyramid_mask(X, Y, Z, BASE)
strain = sf.solve_strain(pyr, eps_star, matrix['C11'], matrix['C12'], matrix['C44'], H)
i0 = N // 2
tr_axis = strain.trace[i0, i0, :]
inside = pyr[i0, i0, :]
i_base = int(np.argmin(np.abs(cx)))
i_tip = int(np.where(inside)[0][-1])

print(f"  {'':10} {'Pryor depth':>12} {'implied Tr':>12} | {'our Tr':>10} {'our depth':>10} "
      f"{'ratio':>7}")
for name, i, dep_p in (('base', i_base, 0.400), ('tip', i_tip, 0.270)):
    tr_o = tr_axis[i]
    print(f"  {name:10} {dep_p:>12.3f} {implied(dep_p):>12.5f} | {tr_o:>10.5f} "
          f"{depth_of(tr_o):>10.3f} {implied(dep_p)/tr_o:>7.3f}")
print("  Pryor's strain is 9-12% more compressive than ours at both points.")

print("\n" + "=" * 78)
print("3. Does the inhomogeneity correction close the gap?")
print("=" * 78)
scale = f_true / f_ga
print(f"  scaling our trace by the sphere-model inhomogeneity factor {scale:.3f}:")
print(f"  {'':10} {'ours':>10} {'x factor':>10} {'Pryor':>10}")
for name, i, dep_p in (('base', i_base, 0.400), ('tip', i_tip, 0.270)):
    print(f"  {name:10} {depth_of(tr_axis[i]):>10.3f} "
          f"{depth_of(tr_axis[i]*scale):>10.3f} {dep_p:>10.3f}")
print("  Pryor's numbers now sit BETWEEN the uncorrected and fully-corrected values, so the")
print("  contrast accounts for the discrepancy with room to spare. The overshoot is expected:")
print("  a sphere factor applied uniformly to a pyramid is a scaling argument, not a solve.")

print("\n" + "=" * 78)
print("4. Ruled out")
print("=" * 78)
print(f"  periodic clamping   : offset -3 eps_T f = {-3*eps_T*pyr.mean():+.2e} vs a "
      f"{implied(0.400)-tr_axis[i_base]:+.4f} discrepancy -- too small by ~100x")
print(f"  piezoelectric field : identically zero on the [001] axis (phi is C4-antisymmetric,")
print(f"                        the axis is C4-invariant), and both quoted numbers are on it")
alt = (matrix['a0'] - dot['a0']) / matrix['a0']
print(f"  eigenstrain convention: ours (a_m-a_d)/a_d = {eps_star:+.5f}; the other common form")
print(f"                        (a_m-a_d)/a_m = {alt:+.5f} is {abs(alt/eps_star):.3f}x larger,")
print(f"                        so it would close part of the gap but not the sign of the")
print(f"                        bracket failure. Cannot be pinned down -- Pryor states neither.")
