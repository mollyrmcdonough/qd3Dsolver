"""Assert the conventions that would otherwise masquerade as physics. Seconds to run.

Every check here corresponds to a mistake that is easy to make, produces a plausible-looking
number, and has either already been made in this project or is one sign flip away from being
made. None of them is caught by any other test, because each one moves a result without making it
look wrong.

Run:  python scripts/convention_audit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elasticity_fd as ef
import heterostructure as hs
import materials as mt

FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


# ======================================================================================
print("\n1. DEFORMATION-POTENTIAL SIGN CONVENTION")
# Vurgaftman/Pryor: a_gap = a_c + a_v with a_v NEGATIVE; the valence edge shifts by -a_v*Tr(e).
# Van de Walle: a_gap = a_c - a_v with a_v positive. Mixing them is ~150 meV and looks plausible.
# Rybchenko Eq. 8 writes E_V = E_V0 + a_V*Tr(e), so a_V(theirs) = -a_v(repo).
# ======================================================================================
mt.set_temperature(0.0)
for name in mt.BINARIES:
    m = mt.material(name)
    check(f"a_gap({name}) < 0  (compression opens the gap)", mt.a_gap(m) < 0,
          f"a_c={m['a_c']:+.2f} a_v={m['a_v']:+.2f} a_gap={mt.a_gap(m):+.2f} eV")
    check(f"a_v({name}) < 0  (Vurgaftman, not Van de Walle)", m['a_v'] < 0)

# The conversion to Rybchenko's sign, stated explicitly so a reader cannot miss it.
insb = mt.material('InSb')
print(f"       -> Rybchenko a_V = {-insb['a_v']:+.2f} eV for InSb "
      f"(repo a_v = {insb['a_v']:+.2f})")

# ======================================================================================
print("\n2. ENERGY ZERO")
# Both papers put zero at the UNSTRAINED InSb valence edge. The repo default puts it at the
# unstrained MATRIX valence edge. For an InAs matrix those differ by exactly VBO(InAs).
# ======================================================================================
dot, matrix = mt.material('InSb'), mt.material('InAs')
mask = np.zeros((4, 4, 4), dtype=bool)
mask[1:3, 1:3, 1:3] = True
Ec_m, Ev_m = mt.band_edge_fields(mask, 0.0, dot, matrix, zero='matrix_vb')
Ec_a, Ev_a = mt.band_edge_fields(mask, 0.0, dot, matrix, zero='absolute')
offset = float((Ev_m - Ev_a).ravel()[0])
check("matrix_vb - absolute == -VBO(matrix)", abs(offset + matrix['VBO']) < 1e-12,
      f"{offset:+.4f} eV vs -VBO(InAs) = {-matrix['VBO']:+.4f}")
check("papers' zero: unstrained InSb VB == 0 on the absolute scale",
      abs(float(Ev_a[mask][0])) < 1e-12, f"{float(Ev_a[mask][0]):+.4e} eV")

# ======================================================================================
print("\n3. ASPECT RATIO -- AR = d/h everywhere, matching the literature")
# Yeap/Rybchenko: AR = base / height, so a LARGER AR is a FLATTER island.
# This repo used h/d until 2026-08-05 and now agrees with them. The checks below assert the
# convention AND pin the map from the old h/d numbers, so a stale record or a half-converted
# script shows up as a failure rather than as a trend running the wrong way.
# ======================================================================================
for ar in (1.0, 2.0, 4.0):
    shape = hs.ellipsoid(2.5, 2.5 / ar)
    got = shape['params']['aspect_ratio']
    hd = shape['params']['height'] / shape['params']['base']
    check(f"ellipsoid(2.5, 2.5/{ar:g}) has AR = {ar:g}", abs(got - ar) < 1e-12,
          f"AR = d/h = {got:.3f}, old h/d = {hd:.3f}, product = {got*hd:.3f}")

check("larger AR is a FLATTER island",
      hs.ellipsoid(20.0, 20.0 / 12.5)['params']['height']
      < hs.ellipsoid(20.0, 20.0 / 6.25)['params']['height'],
      "AR 12.5 -> h = 1.6 nm, AR 6.25 -> h = 3.2 nm")

for hd_old, ar_new in ((0.08, 12.5), (0.10, 10.0), (0.15, 6.6667)):
    check(f"legacy h/d = {hd_old:g} is AR = {ar_new:g}", abs(1.0 / hd_old - ar_new) < 1e-3,
          f"1/{hd_old:g} = {1/hd_old:.4f}")

check("spherical_lens also reports AR = d/h",
      abs(hs.spherical_lens(10.0, 5.0)['params']['aspect_ratio'] - 4.0) < 1e-12,
      "r = 10, h = 5 -> d = 20, AR = 4 (Pryor & Pistol's 'h/d = 1/4')")

# ======================================================================================
print("\n4. ALLOY COMPOSITION -- their x is ARSENIC, ours is ANTIMONY")
# materials.alloy('InAsSb', x) takes x = Sb fraction. Yeap's x is the As fraction.
# So x_ours = 1 - x_theirs. Getting this backwards gives a plausible curve running the wrong way.
# ======================================================================================
check("alloy('InAsSb', 1.0) is pure InSb",
      abs(mt.alloy('InAsSb', 1.0)['Eg'] - mt.material('InSb')['Eg']) < 1e-12)
check("alloy('InAsSb', 0.0) is pure InAs",
      abs(mt.alloy('InAsSb', 0.0)['Eg'] - mt.material('InAs')['Eg']) < 1e-12)
for x_as in (0.0, 0.2, 0.4):
    a = mt.alloy('InAsSb', 1.0 - x_as)
    print(f"       x_As={x_as:.1f} -> alloy('InAsSb', {1-x_as:.1f}) = {a['name']}, "
          f"Eg = {a['Eg']:.4f} eV")
check("alloy('InGaSb', 1.0) is pure InSb",
      abs(mt.alloy('InGaSb', 1.0)['Eg'] - mt.material('InSb')['Eg']) < 1e-12)

# ======================================================================================
print("\n4b. LUTTINGER ELLIPTICITY -- gamma1 > 2*gamma2 and gamma1 > 2*gamma3")
# Violate either and the six-band operator admits modes bounded only by the grid: they converge,
# they are Kramers-paired, and they are not states. Measured, forcing InSb's gamma3 = 16.5 into
# the InAs matrix (gamma1 = 20.0): sixteen eigenvalues at +2.33 eV, 0% inside the island, 1.7 eV
# above the well top. Every binary here passes -- but the antimonide margins are thin, so the
# ALLOYS are checked too, since bowing is not linear and an alloy can fail where both parents
# pass.
# ======================================================================================
for name in ('InSb', 'InAs', 'GaAs', 'GaSb'):
    p = mt.kp_params(mt.material(name))
    g1, g2, g3 = p['gamma1L'], p['gamma2L'], p['gamma3L']
    check(f"{name} is elliptic", min(g1 - 2 * g2, g1 - 2 * g3) > 0,
          f"g1-2g2 = {g1-2*g2:+.2f}, g1-2g3 = {g1-2*g3:+.2f} "
          f"(margin on g3 {100*(g1-2*g3)/g1:.1f}% of g1)")
for system, xs in (('InAsSb', (0.25, 0.5, 0.75)), ('InGaSb', (0.25, 0.5, 0.75))):
    worst, at = np.inf, None
    for x in xs:
        p = mt.kp_params(mt.alloy(system, x))
        g1, g2, g3 = p['gamma1L'], p['gamma2L'], p['gamma3L']
        if min(g1 - 2 * g2, g1 - 2 * g3) < worst:
            worst, at = min(g1 - 2 * g2, g1 - 2 * g3), x
    check(f"{system} is elliptic across composition", worst > 0,
          f"worst margin {worst:+.2f} at x = {at:g}")

# ======================================================================================
print("\n4c. STRONG ELLIPTICITY -- the condition symmetrized ordering FAILS, and the fix")
# 4b checks Legendre-Hadamard ellipticity: <phi|A|phi> <= 0 for RANK-ONE phi = k (x) v, which is
# the statement that every bulk band curves downward. It passes everywhere.
#
# Strong ellipticity asks the same of EVERY 18-vector, not just the rank-one ones. It FAILS for
# every material here, and that is not a curiosity: with constant or continuous coefficients only
# rank-one directions are accessible, so Legendre-Hadamard suffices and the spectrum is bounded by
# the band edges -- but at an ABRUPT interface the positive directions of A open up and the
# symmetrized six-band operator has no maximum at all. The interface modes this project has been
# chasing are that unboundedness, not a discretization defect.
#
# So this section ASSERTS NOTHING. It reports a known and unfixable-by-discretization property of
# the symmetrized Hamiltonian, and it is here so that the number is on the record next to the
# Legendre-Hadamard margins that look reassuring. lambda_max scales with how bad the problem is:
# GaAs is eight times better than InSb, which is why the InAs/GaAs benchmark behaves.
# ======================================================================================
import kp_planewave as pw                                          # noqa: E402
print(f"    {'material':<10} {'g1-2g2':>8} {'g1-2g3':>8} | {'lmax sym':>10} | {'lmax B-F':>11}")
for name in ('GaAs', 'GaSb', 'InAs', 'InSb'):
    p = mt.kp_params(mt.material(name))
    g1, g2, g3 = p['gamma1L'], p['gamma2L'], p['gamma3L']
    a = pw.strong_ellipticity(g1, g2, g3, 'symmetrized')
    b = pw.strong_ellipticity(g1, g2, g3, 'burt-foreman')
    check(f"{name}: Burt-Foreman ordering is strongly elliptic", b['strongly_elliptic'],
          f"g1-2g2 {g1-2*g2:+.2f}, g1-2g3 {g1-2*g3:+.2f} | lambda_max "
          f"{a['lambda_max']:+.4f} symmetrized -> {b['lambda_max']:+.2e} Burt-Foreman")
print("    Legendre-Hadamard holds everywhere (4b), and buys less than it looks like.")
print("    SYMMETRIZED ordering is strongly elliptic NOWHERE, so an abrupt interface leaves the")
print("    operator unbounded above -- which is what the interface modes are. BURT-FOREMAN")
print("    ordering lands lambda_max on zero everywhere and is the fix. See kp_planewave.")

# ======================================================================================
print("\n5. ELASTIC MODULI -- Rybchenko Eq. 3 is already `voigt_moduli`")
# Their mu_av = (C11 - C12)/5 + 3*C44/5, the Voigt shear average used for the matrix in the
# isotropic approximation. If this were not already what the repo computes, the isotropic
# cross-checks in Stage 2 would be comparing two different definitions.
# ======================================================================================
for name in mt.BINARIES:
    C11, C12, C44 = mt.elastic(mt.material(name))
    theirs = (C11 - C12) / 5.0 + 3.0 * C44 / 5.0
    ours = ef.voigt_moduli(C11, C12, C44)[1]
    check(f"voigt_moduli({name})[1] == Rybchenko Eq. 3", abs(ours - theirs) < 1e-12,
          f"{ours:.6f} vs {theirs:.6f} GPa")

# ======================================================================================
print("\n6. TEMPERATURE -- Varshni gaps at the replication temperature")
# Yeap computed at 80 K and 300 K. `set_temperature` mutates MATERIALS in place and `build`
# stores COPIES, so it must be called BEFORE build; a stale call returns identical numbers with
# no error, which looks like a null result rather than a bug.
# ======================================================================================
for T, targets in ((80.0, dict(InAs=0.4068, InSb=0.2268)), (300.0, {})):
    mt.set_temperature(T)
    for name, want in targets.items():
        got = mt.material(name)['Eg']
        check(f"Eg({name}, {T:g} K) = {want:.4f}", abs(got - want) < 5e-4, f"got {got:.4f} eV")
    if not targets:
        print(f"       {T:g} K: " + ", ".join(
            f"{n} {mt.material(n)['Eg']:.4f}" for n in ('InAs', 'InSb')) + " eV")

mt.set_temperature(80.0)
d80 = mt.material('InAs')['Eg']
mt.set_temperature(77.0)
d77 = mt.material('InAs')['Eg']
print(f"       77 K vs 80 K: InAs Eg differs by {abs(d80-d77)*1e3:.2f} meV "
      f"(the replication needs 80 K; production wants 77 K)")

# ======================================================================================
print("\n7. HYDROSTATIC STRAIN IS NOT COUNTED TWICE")
# band_edge_fields returns edges that ALREADY carry a_c*Tr and -a_v*Tr. Handing those to a
# Hamiltonian builder with include_hydrostatic=True applies it again.
# ======================================================================================
mt.set_temperature(0.0)
tr = -0.05
Ec0, Ev0 = mt.band_edge_fields(mask, 0.0, dot, matrix, zero='absolute')
Ec1, Ev1 = mt.band_edge_fields(mask, tr, dot, matrix, zero='absolute')
d_ec = float((Ec1 - Ec0)[mask][0])
d_ev = float((Ev1 - Ev0)[mask][0])
check("band_edge_fields applies +a_c*Tr to Ec", abs(d_ec - dot['a_c'] * tr) < 1e-12,
      f"{d_ec:+.5f} vs {dot['a_c']*tr:+.5f} eV")
check("band_edge_fields applies -a_v*Tr to Ev", abs(d_ev + dot['a_v'] * tr) < 1e-12,
      f"{d_ev:+.5f} vs {-dot['a_v']*tr:+.5f} eV")

# ======================================================================================
print("\n" + "=" * 70)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("All convention checks passed.")
