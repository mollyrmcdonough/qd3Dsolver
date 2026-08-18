"""Closed-form oblate-spheroid strain vs the finite-element solver, across aspect ratio.

Two purposes:

1. **Fix the convention.** `eshelby.oblate_inclusion_strain` transcribes Rybchenko's appendix
   section 2, whose e^T is NOT the elastic strain the deformation potentials act on. Measured on
   a sphere: e^T = 1.11407 eps* while the elastic strain is 0.46700 eps*, and the ratio 0.41919
   is exactly |1 - alpha| with alpha = (1+sigma)/(3(1-sigma)) the dilatational Eshelby factor.
   A fixed convention offset shows up as a CONSTANT ratio across aspect ratio; a transcription
   error does not. This distinguishes them.

2. **Validate the FE solver where nothing else does.** The existing elasticity validation covers
   the sphere and the clamped slab. Nothing covered intermediate aspect ratios -- the flat-lens
   regime self-assembled dots actually occupy, and where staircasing and padding are worst.

The FE solver is the reference here, not the closed form: it is validated to 2.8e-17 on the exact
anisotropic slab and 2.6e-03 on the inhomogeneous sphere, and it is what production uses. The
closed form is isotropic, so a few per cent of disagreement is expected and is Rybchenko's own
"IE model" difference, not an error.

Run:  python scripts/ellipsoid_strain_validation.py [--fine]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elasticity_fd as ef
import eshelby as es
import heterostructure as hs
import materials as mt

BASE = 20.0                     # nm; scale-free, so this is just a convenient size
CELLS = 24 if '--fine' in sys.argv else 12   # cells across the HEIGHT
PAD_FRAC = 1.0
ARS = (1.0, 1.5, 2.0, 4.0, 8.0)


def fe_interior(dot, matrix, AR, cells=CELLS, pad_frac=PAD_FRAC):
    height = BASE / AR
    h = max(height / cells, 0.15)
    env = hs.build(hs.ellipsoid(BASE, height), dot, matrix=matrix, h=h,
                   pad=pad_frac * BASE, z_pad=pad_frac * BASE,
                   use_piezo=False, vol_tol=0.35, verbose=False)
    from scipy.ndimage import binary_erosion
    sel = binary_erosion(env['mask'], iterations=1)
    if not sel.any():
        sel = env['mask']
    g = env['strain']
    return dict(e11=float(g.exx[sel].mean()), e33=float(g.ezz[sel].mean()),
                e11_std=float(g.exx[sel].std()), e33_std=float(g.ezz[sel].std()),
                h=h, n=int(sel.sum()), vol_err=env['vol_err'])


if __name__ == '__main__':
    mt.set_temperature(80.0)
    dot, matrix = mt.material('InSb'), mt.material('InAs')
    Cd, Cm = mt.elastic(dot), mt.elastic(matrix)
    eps = mt.misfit(dot, matrix)
    C11, C12, _ = Cd
    qw_par, qw_perp = eps, -2.0 * C12 / C11 * eps
    # The closed form is ISOTROPIC, so it converges to the isotropic QW limit, not the cubic one.
    # Comparing it against the cubic limit would look like a 30% error that is really the
    # approximation doing exactly what it says on the tin.
    k_i, mu_i = ef.voigt_moduli(*Cd)
    nu_dot = (3.0 * k_i - 2.0 * mu_i) / (2.0 * (3.0 * k_i + mu_i))
    qw_perp_iso = -2.0 * nu_dot / (1.0 - nu_dot) * eps

    print(f"InSb in InAs at 80 K,  eps* = {eps:+.6f},  base {BASE:g} nm, "
          f"{CELLS} cells across the height")
    print(f"QW limits:  e_par {qw_par:+.6f} | e_perp cubic {qw_perp:+.6f} "
          f"(FE target)  isotropic {qw_perp_iso:+.6f} (closed-form target, nu_dot={nu_dot:.4f})\n")

    print("A. FE INTERIOR STRAIN vs CLOSED FORM")
    print(f"{'AR':>5} {'h':>6} {'e11 FE':>10} {'e11 CF':>10} {'ratio':>8} | "
          f"{'e33 FE':>10} {'e33 CF':>10} {'ratio':>8}")
    print("-" * 82)
    rows = []
    for AR in ARS:
        fe = fe_interior(dot, matrix, AR)
        cf = es.oblate_inclusion_strain(eps, Cd, Cm, BASE / 2.0, BASE / (2.0 * AR))
        r11 = fe['e11'] / cf['e11'] if cf['e11'] else float('nan')
        r33 = fe['e33'] / cf['e33'] if cf['e33'] else float('nan')
        rows.append((AR, fe, cf, r11, r33))
        print(f"{AR:5.1f} {fe['h']:6.3f} {fe['e11']:+10.6f} {cf['e11']:+10.6f} {r11:8.4f} | "
              f"{fe['e33']:+10.6f} {cf['e33']:+10.6f} {r33:8.4f}", flush=True)

    r11s = np.array([r[3] for r in rows])
    r33s = np.array([r[4] for r in rows])
    print(f"\n   e11 FE/CF ratio: {r11s.min():.4f} .. {r11s.max():.4f}")
    print(f"   e33 FE/CF ratio: {r33s.min():.4f} .. {r33s.max():.4f}")
    print("   Ratios near 1 mean the isotropic closed form tracks the cubic solve. Expect it to\n"
          "   hold for e11 and to FAIL for e33 as the dot flattens -- that is the isotropic\n"
          "   approximation, not an error, and it is why the FE solver stays the production path.")

    print("\nB. INTERIOR HOMOGENEITY (Eshelby's theorem: std must be ~0)")
    print(f"{'AR':>5} {'n_int':>7} {'e11 std':>11} {'e33 std':>11} {'/|eps*|':>9} {'vol_err':>9}")
    print("-" * 60)
    for AR, fe, cf, _, _ in rows:
        worst = max(fe['e11_std'], fe['e33_std'])
        print(f"{AR:5.1f} {fe['n']:7d} {fe['e11_std']:11.3e} {fe['e33_std']:11.3e} "
              f"{worst/abs(eps):9.3f} {fe['vol_err']:+9.3f}")

    print("\nC. LIMITS")
    fe1 = rows[0][1]
    K_i = ef.voigt_moduli(*Cd)[0]
    mu_m = ef.voigt_moduli(*Cm)[1]
    ref = ef.inhomogeneous_sphere_trace(eps, K_i, mu_m)
    tr = fe1['e11'] * 2 + fe1['e33']
    print(f"   sphere (AR=1): FE trace {tr:+.6f} vs inhomogeneous_sphere_trace {ref:+.6f}  "
          f"rel {abs(tr-ref)/abs(ref):.2e}")
    feN = rows[-1][1]
    e11N, e33N = feN['e11'], feN['e33']
    print(f"   QW limit (AR={ARS[-1]:g}): FE e11 {e11N:+.6f} vs cubic {qw_par:+.6f} "
          f"({abs(e11N - qw_par) / abs(qw_par) * 100:.1f}%),  "
          f"e33 {e33N:+.6f} vs cubic {qw_perp:+.6f} "
          f"({abs(e33N - qw_perp) / abs(qw_perp) * 100:.1f}%)")
    cfN = rows[-1][2]
    print(f"   closed form at the same AR: e11 {cfN['e11']:+.6f}, e33 {cfN['e33']:+.6f} "
          f"-- its target is the ISOTROPIC e_perp {qw_perp_iso:+.6f}, not the cubic one")
