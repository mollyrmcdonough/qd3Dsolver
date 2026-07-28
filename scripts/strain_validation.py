"""Validate strain_fourier, and the Bir-Pikus terms it feeds, against analytic ground truths.

Run directly: python strain_validation.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import qdsolver_core as qd
import strain_fourier as sf
import kp_pryor as kp
import pryor1998 as pr
from qdsolver_core import HBAR2_OVER_2M0 as G

np.set_printoptions(suppress=True, precision=6)
dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
C11, C12, C44 = matrix['C11'], matrix['C12'], matrix['C44']
print(f"eps_star = {eps_star:.6f}  (InAs dot in GaAs);  GaAs C11/C12/C44 = "
      f"{C11}/{C12}/{C44} GPa\n")

# ------------------------------------------------- 1. clamped slab, exact at every fill fraction
print("=" * 78)
print("TEST 1  [001] slab in a clamped cell -- exact closed form, anisotropic")
print("=" * 78)
N, h = 32, 0.5
zc = np.arange(N) * h
print(f"  {'n_layers':>8} {'f':>7} | {'exx analytic':>13} {'numeric':>11} | "
      f"{'ezz analytic':>13} {'numeric':>11}")
for nz in (1, 2, 4, 11, 16):
    Z = np.meshgrid(zc, zc, zc, indexing='ij')[2]
    slab = Z < nz * h
    f = slab.mean()
    e = sf.solve_strain(slab, eps_star, C11, C12, C44, h)
    g = e.at(slab)
    pa, za = sf.clamped_slab_strain(eps_star, C11, C12, f)
    print(f"  {nz:>8} {f:>7.4f} | {pa:>13.6f} {g['exx']:>11.6f} | "
          f"{za:>13.6f} {g['ezz']:>11.6f}")
    if nz == 1:
        err = max(abs(g['exx'] - pa), abs(g['ezz'] - za))
        shear = max(abs(g['exy']), abs(g['eyz']), abs(g['ezx']))
pl, zl = sf.pseudomorphic_layer_strain(eps_star, C11, C12)
print(f"  {'f -> 0':>16} | {pl:>13.6f} {'':>11} | {zl:>13.6f}   <- pseudomorphic limit")
print(f"  thinnest-slab max error {err:.2e}, max shear {shear:.2e}")

# ------------------------------------------------------------- 2. isotropic Eshelby sphere
print("\n" + "=" * 78)
print("TEST 2  isotropic Eshelby sphere -- uniform interior, Davies' zero trace outside")
print("=" * 78)
nu = qd.voigt_poisson_ratio(C11, C12, C44)
Ci = sf.isotropic_constants(nu)
tr_a = sf.isotropic_sphere_trace(eps_star, nu)
print(f"  Voigt nu = {nu:.5f};  analytic elastic trace inside = {tr_a:+.6f}")
print(f"  {'box(nm)':>8} {'f':>8} {'trace(core)':>13} {'rel err':>10} {'shear':>10} "
      f"{'|tr| r>2R':>11}")
h, R = 0.5, 6.0
for N in (48, 64, 96, 128, 160):
    c = np.arange(N) * h - (N - 1) * h / 2
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    sph = qd.sphere_mask(X, Y, Z, R)
    es = sf.solve_strain(sph, eps_star, *Ci, h)
    core = qd.sphere_mask(X, Y, Z, R * 0.6)
    g, gr = es.at(core), es.at_rms(core)
    tr = g['exx'] + g['eyy'] + g['ezz']
    far = (X**2 + Y**2 + Z**2) > (2.0 * R) ** 2
    print(f"  {N*h:>8.0f} {sph.mean():>8.5f} {tr:>13.6f} {abs(tr/tr_a-1):>10.2e} "
          f"{max(gr['exy'],gr['eyz'],gr['ezx']):>10.2e} "
          f"{np.abs(es.trace[far]).mean():>11.2e}")
print(f"  interior isotropy at the largest box: exx={g['exx']:+.6f} eyy={g['eyy']:+.6f} "
      f"ezz={g['ezz']:+.6f}")
print("  residual / f is constant at ~1.16 -- the O(f) clamping offset and nothing else.")

# --------------------------------------------- 3. the factor vs the old hydrostatic-only model
print("\n" + "=" * 78)
print("TEST 3  comparison with qdsolver_core.trace_strain_from_mask")
print("=" * 78)
_, old_tr = qd.trace_strain_from_mask(np.array([True]), eps_star, nu)
print(f"  trace_strain_from_mask  {old_tr:+.6f}   <- CONSTRAINED (total) strain, alpha*eps_T")
print(f"  correct elastic trace   {tr_a:+.6f}   <- (alpha-1)*eps_T, what a_c/a_v act on")
print(f"  ratio {old_tr/tr_a:.4f}  =  (1+nu)/(2(1-2nu)) = {(1+nu)/(2*(1-2*nu)):.4f}")
print("  -> the existing hydrostatic-only model overstates the band-edge shift by this factor.")

# ------------------------------------------------------- 4. Pryor pyramid, full anisotropic
print("\n" + "=" * 78)
print("TEST 4  Pryor b=14 nm pyramid, full anisotropic cubic elasticity")
print("=" * 78)
base = 14.0
for h, pad in ((0.7, 10.5), (0.7, 17.5), (0.5, 17.5)):
    cx = np.arange(-(base / 2 + pad), base / 2 + pad + 1e-9, h)
    cz = np.arange(-pad, base / 2 + pad + 1e-9, h)
    X, Y, Z = np.meshgrid(cx, cx, cz, indexing='ij')
    pyr = qd.pyramid_mask(X, Y, Z, base)
    ep = sf.solve_strain(pyr, eps_star, C11, C12, C44, h)
    d, dr = ep.at(pyr), ep.at_rms(pyr)
    tr = d['exx'] + d['eyy'] + d['ezz']
    print(f"  h={h} pad={pad}  grid {X.shape} = {X.size:,} pts, f={pyr.mean():.4f}, "
          f"clamping residual {sf.padding_report(ep, pyr):.4f}")
    print(f"     <exx>={d['exx']:+.6f}  <eyy>={d['eyy']:+.6f}  <ezz>={d['ezz']:+.6f}   "
          f"trace={tr:+.6f}  biaxial={d['exx']+d['eyy']-2*d['ezz']:+.6f}")
    print(f"     rms shear  exy={dr['exy']:.6f}  eyz={dr['eyz']:.6f}  ezx={dr['ezx']:.6f}   "
          f"(means vanish by symmetry: {abs(d['ezx']):.1e})")
print(f"  hydrostatic-only model would give a uniform trace of {old_tr:+.6f} with zero shear")

# ------------------------------------------- 5. Bir-Pikus <-> kinetic correspondence (exact)
print("\n" + "=" * 78)
print("TEST 5  Bir-Pikus terms vs the kinetic terms -- over-determined, must be exact")
print("=" * 78)
print("  Setting eps_ij = k_i k_j with b = -2 G g2, d = -2 sqrt(3) G g3, a_v = G g1, a_c = G")
print("  must reproduce the kinetic Hamiltonian exactly: 3 complex expressions (Q, R, S)")
print("  pinned by 2 constants, so a misplaced or mis-phased element cannot survive.")
kx, ky, kz = 0.21, -0.13, 0.31
st_k = sf.StrainTensor(kx * kx, ky * ky, kz * kz, kx * ky, ky * kz, kz * kx)
for nb in (4, 6, 8):
    p = dict(dot)
    if nb == 8:
        p['Ep'] = 0.0     # U and V are linear in k and have no strain analogue
        g1, g2, g3 = kp.modified_luttinger(p['gamma1L'], p['gamma2L'], p['gamma3L'],
                                           p['Eg'], p['delta_so'], p['Ep'])
    else:
        g1, g2, g3 = p['gamma1L'], p['gamma2L'], p['gamma3L']
    q = dict(p, b=-2 * G * g2, d=-2 * np.sqrt(3) * G * g3, a_v=G * g1, a_c=G)
    H_kin = kp.bulk_hamiltonian(kx, ky, kz, p, n_bands=nb)
    H_str = kp.bulk_hamiltonian(0, 0, 0, q, n_bands=nb, strain=st_k, include_hydrostatic=True)
    H_0 = kp.bulk_hamiltonian(0, 0, 0, p, n_bands=nb)
    print(f"  {nb}-band  max|H_BirPikus - H_kinetic| = {np.abs(H_str-H_kin).max():.2e}   "
          f"(scale {np.abs(H_kin-H_0).max():.4f} eV)")

# ------------------------------------------------- 6. physical sign: HH on top under compression
print("\n" + "=" * 78)
print("TEST 6  Bir-Pikus sign -- compressive biaxial strain must put HH on top")
print("=" * 78)
print("  (independent of TEST 5, which only fixes the terms relative to one another)")
par, zz = sf.pseudomorphic_layer_strain(eps_star, C11, C12)
st_b = sf.StrainTensor(par, par, zz, 0.0, 0.0, 0.0)
Qe, Re, Se = kp.bir_pikus_terms(st_b, dot['b'], dot['d'])
print(f"  InAs on GaAs: exx=eyy={par:+.5f} ezz={zz:+.5f};  "
      f"Q_eps={Qe.real:+.5f} eV (b={dot['b']}, d={dot['d']})")
for nb in (4, 6, 8):
    H = kp.bulk_hamiltonian(0, 0, 0, dot, n_bands=nb, strain=st_b)
    E, V = np.linalg.eigh(H)
    top = np.argsort(E)[-1] if nb != 8 else np.argsort(E)[-3]
    w = np.abs(V[:, top]) ** 2
    print(f"  {nb}-band  hermiticity {np.abs(H-H.conj().T).max():.1e}   "
          f"top valence state is {kp.BAND_LABELS[nb][int(np.argmax(w))]} "
          f"(weight {w.max():.4f})")
E4 = np.sort(np.linalg.eigvalsh(kp.bulk_hamiltonian(0, 0, 0, dot, n_bands=4, strain=st_b)))[::-1]
print(f"  4-band HH-LH splitting {(E4[0]-E4[2])*1e3:.1f} meV = 2|Q_eps| "
      f"{2*abs(Qe.real)*1e3:.1f} meV")
# LH sits on the -P+Q diagonal (so +Q_eps) and SO on -P-delta; they are coupled by the
# -sqrt(2) Q element, giving |coupling|^2 = 2 Q_eps^2.
lh, so = Qe.real, -dot['delta_so']
disc = np.sqrt(((lh - so) / 2) ** 2 + 2 * Qe.real ** 2)
print(f"  6-band LH-SO repulsion through Q_eps: analytic 2x2 gives "
      f"{(lh+so)/2 + disc:+.4f} / {(lh+so)/2 - disc:+.4f} eV")
E6 = np.sort(np.linalg.eigvalsh(kp.bulk_hamiltonian(0, 0, 0, dot, n_bands=6, strain=st_b)))[::-1]
print(f"  6-band numeric                              {E6[2]:+.4f} / {E6[4]:+.4f} eV")

# ------------------------------------- 7. Pryor's u and v: hydrostatic limit is exact algebra
print("\n" + "=" * 78)
print("TEST 7  interband strain terms u, v -- pure hydrostatic strain must renormalize P0")
print("=" * 78)
print("  For eps_ij = e delta_ij the contractions collapse: u = e U and v = e V exactly, so")
print("  U -> U - u = (1-e) U and V -> (1-e) V. Hydrostatic strain therefore does nothing to")
print("  these terms except scale P0 by (1-e) -- a closed form to check the contraction against.")
e_h = 0.037
st_h = sf.StrainTensor(e_h, e_h, e_h, 0.0, 0.0, 0.0)
P0 = kp.kane_P0(dot['Ep'])
u_n, v_n = kp.interband_strain_scalars(st_h, P0, kx, ky, kz)
u_a, v_a = e_h * P0 * kz / np.sqrt(3.0), e_h * P0 * (kx - 1j * ky) / np.sqrt(6.0)
print(f"  |u_numeric - e*U| = {abs(u_n-u_a):.2e},  |v_numeric - e*V| = {abs(v_n-v_a):.2e}")

# And the same statement at the level of the assembled matrix: the 8-band Hamiltonian with
# hydrostatic strain and interband terms on must equal the unstrained one built with a scaled Ep
# (Ep goes as P0^2, hence the square).
p_scaled = dict(dot, Ep=dot['Ep'] * (1 - e_h) ** 2)
H_uv = kp.bulk_hamiltonian(kx, ky, kz, dot, n_bands=8, strain=st_h, interband_strain=True)
H_no = kp.bulk_hamiltonian(kx, ky, kz, dot, n_bands=8, strain=st_h, interband_strain=False)
# Ep also feeds the modified Luttinger parameters, so compare only the conduction-row couplings.
rows = np.zeros((8, 8), bool)
rows[:2, 2:] = True
rows[2:, :2] = True
H_ref = kp.bulk_hamiltonian(kx, ky, kz, p_scaled, n_bands=8, strain=st_h, interband_strain=False)
print(f"  conduction-valence block: |H(u,v on) - H(scaled Ep)| = "
      f"{np.abs((H_uv-H_ref)[rows]).max():.2e}   "
      f"(effect being tested: {np.abs((H_uv-H_no)[rows]).max():.4f} eV)")
print(f"  u, v vanish at k = 0 as they must: "
      f"{max(abs(x) for x in kp.interband_strain_scalars(st_k, P0, 0.0, 0.0, 0.0)):.2e}")
