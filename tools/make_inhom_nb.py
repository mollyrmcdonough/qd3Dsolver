"""Generate pryor_inhomogeneous.ipynb (Pryor's elasticity method -> his Fig. 2 and Fig. 6)."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))

md(r"""# Pryor's elasticity method: Fig. 2 and Fig. 6

C. Pryor, *Phys. Rev. B* **57**, 7190 (1998); preprint
[arXiv:cond-mat/9710304](https://arxiv.org/abs/cond-mat/9710304).

Everything earlier in this package used `strain_fourier.solve_strain`, which diagonalizes the
Christoffel tensor per wavevector. That is exact in the elasticity operator and costs one FFT,
but the Fourier diagonalization **requires spatially constant elastic constants**. This notebook
switches to Pryor's own method, from his Sec. II:

> "First, the strain is calculated using linear continuum elastic theory. The strain energy for
> the system is computed using a finite differencing approximation, and then minimized using the
> conjugate gradient algorithm."

(the energy functional there cites Landau & Lifshitz, *Theory of Elasticity*), on "a cubic grid
with periodic boundary conditions".

## Why the method matters

Both approaches minimize the same functional and find the same stationary point. They differ in
one consequence: a real-space minimization carries **position-dependent** $C_{ijkl}$ for free,
and a Fourier solve cannot carry it at all. For a misfitting sphere,

$$\mathrm{Tr}\,\varepsilon = -3\,\varepsilon_T\,\frac{4\mu_m}{3K_i + 4\mu_m}$$

involves the **inclusion's** bulk modulus and the **matrix's** shear modulus. A homogeneous
solver must take both from one material, so no choice of constants reaches the true value.
InAs is ~25% softer than GaAs in bulk modulus, and that is worth ~15% in the strain — the
entire unexplained residual against Pryor's quoted conduction-well depths.

`elasticity_fd.solve_strain_fd` implements this with trilinear (Q1) hexahedral elements and full
$2\times2\times2$ quadrature, matrix-free, preconditioned by the exact inverse of the
corresponding *homogeneous* operator (so CG only has to resolve the material contrast).

Full validation is in `elasticity_validation.py`; the decisive parts are reproduced below.""")

co("""import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

import qdsolver_core as qd
import strain_fourier as sf
import elasticity_fd as ef
import kp_pryor as kp
import pryor1998 as pr

dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
C_dot = (dot['C11'], dot['C12'], dot['C44'])
C_mat = (matrix['C11'], matrix['C12'], matrix['C44'])
GaAs_CB = matrix['E_vbo'] + matrix['Eg']
K_i, mu_i = ef.voigt_moduli(*C_dot)
K_m, mu_m = ef.voigt_moduli(*C_mat)
print(f"Voigt moduli (GPa):  InAs K={K_i:.1f} mu={mu_i:.1f} | GaAs K={K_m:.1f} mu={mu_m:.1f}")
print(f"stiffness contrast K_InAs/K_GaAs = {K_i/K_m:.3f}")""")

md("""## Validation: the test a homogeneous solver cannot pass

An isotropic surrogate is used so the misfitting-sphere formula is exact. The homogeneous case
must reproduce what `strain_fourier` already validates against; the inhomogeneous case is the
new capability, and the point is that its answer lies **outside** the interval a homogeneous
solver can reach for any choice of constants.""")

co("""h = 0.5
nu_m = qd.voigt_poisson_ratio(*C_mat)
nu_d = qd.voigt_poisson_ratio(*C_dot)
Ci_m = sf.isotropic_constants(nu_m)
Ci_d = sf.isotropic_constants(nu_d, C11=Ci_m[0] * (dot['C11'] / matrix['C11']))
Ki, _ = ef.voigt_moduli(*Ci_d)
Km, Mm = ef.voigt_moduli(*Ci_m)

cc = qd.centered_axis(96, h)
Xs, Ys, Zs = np.meshgrid(cc, cc, cc, indexing='ij')
sph = qd.sphere_mask(Xs, Ys, Zs, 6.0)
core = qd.sphere_mask(Xs, Ys, Zs, 3.6)

print(f"{'case':32} {'analytic':>11} {'numeric':>11} {'rel err':>9}")
for label, Cd, Kx in (('homogeneous   (K_i = K_GaAs)', Ci_m, Km),
                      ('INHOMOGENEOUS (K_i = K_InAs)', Ci_d, Ki)):
    e = ef.solve_strain_fd(sph, eps_star, Cd, Ci_m, h, tol=1e-11)
    g = e.at(core)
    num = g['exx'] + g['eyy'] + g['ezz']
    exact = ef.inhomogeneous_sphere_trace(eps_star, Kx, Mm)
    print(f"{label:32} {exact:>11.6f} {num:>11.6f} {abs(num/exact-1):>9.2e}")

lo = ef.inhomogeneous_sphere_trace(eps_star, Ki, ef.voigt_moduli(*Ci_d)[1])
hi = ef.inhomogeneous_sphere_trace(eps_star, Km, Mm)
true = ef.inhomogeneous_sphere_trace(eps_star, Ki, Mm)
print(f"\\nany homogeneous solve is confined to [{min(lo,hi):+.6f}, {max(lo,hi):+.6f}]")
print(f"the true inhomogeneous value {true:+.6f} is outside it")""")

md("""---
# Fig. 2 — band structure from the local value of the strain

> **FIG. 2.** Band structure based on the local value of the strain. (a) Bands along the 001
> direction, through the center of the island. (b) Bands Along the 100 direction, through the
> base of the island.

Sec. IV: *"the eigenvalues of Eq. 4 with $\\vec k = 0$ ... The bands are shown for an island with
$b = 10$ nm."* At $k=0$ the interband terms $u, v$ vanish (they carry a derivative), so the
eight-band problem factorizes into a closed-form conduction band plus a $6\\times6$ valence
block — which is what `kp_pryor.local_band_edges` diagonalizes.

### Choosing the grid

The grid is not a free choice here, and getting it wrong is not subtle. A $\\{101\\}$ pyramid
has facets at 45°, and whether the sample points fall just inside or just outside them swings
the mask volume by over 12% with **no smooth trend in $h$**. At $h = 0.4$ the *half-offset*
grid (even $N$, so the points straddle $z=0$) samples the facets almost exactly, while the grid
with a point *on* $z=0$ is the worst case:

| $h$ | $N$ even (half-offset) | $N$ odd (point at $z=0$) |
|---|---|---|
| 0.5 | −14.5% | **−0.25%** |
| 0.4 | **−0.16%** | +12.3% |
| 0.25 | −7.4% | **−0.06%** |

So the parity that is right at one spacing is wrong at the next. `mask_volume_error` is checked
here rather than assumed. The half-offset grid puts the island's base layer of cells at
$z = 0.2$ nm rather than exactly 0, which is if anything the better "through the base" cut,
since $z=0$ is the boundary itself.""")

co("""BASE, H, N = 10.0, 0.4, 112       # N EVEN: half-offset grid, which samples {101} facets well
cx = qd.centered_axis(N, H)
X, Y, Z = np.meshgrid(cx, cx, cx, indexing='ij')
pyr = qd.pyramid_mask(X, Y, Z, BASE)
assert qd.mirror_asymmetry(pyr, 0) == 0 and qd.mirror_asymmetry(pyr, 1) == 0
i0 = N // 2
assert abs(cx[i0] - H / 2) < 1e-12       # base layer of cells, just above z = 0
vol_err = qd.mask_volume_error(pyr, H, BASE**2 * (BASE/2) / 3)
assert abs(vol_err) < 0.01, f"mask volume error {vol_err:+.4f} -- wrong grid parity"

t0 = time.time()
strain = ef.solve_strain_fd(pyr, eps_star, C_dot, C_mat, H, tol=1e-10)
strain_hom = sf.solve_strain(pyr, eps_star, *C_mat, H)      # the old homogeneous field
g, gh = strain.at(pyr), strain_hom.at(pyr)
print(f"grid {X.shape} = {X.size:,} pts, h={H}, box {N*H:.1f} nm, "
      f"vol err {qd.mask_volume_error(pyr, H, BASE**2*(BASE/2)/3):+.4f}")
print(f"  FD: CG {strain.cg_iterations} iters, rel {strain.cg_residual:.1e}, {time.time()-t0:.0f}s")
print(f"  <Tr eps>  inhomogeneous (FD) {g['exx']+g['eyy']+g['ezz']:+.5f}"
      f"   homogeneous (Fourier) {gh['exx']+gh['eyy']+gh['ezz']:+.5f}")
print(f"  rms shear exy  FD {strain.at_rms(pyr)['exy']:.5f}"
      f"   Fourier {strain_hom.at_rms(pyr)['exy']:.5f}")""")

co("""def bands_on(st, index):
    sub = sf.StrainTensor(*[getattr(st, k)[index] for k in sf.StrainTensor.COMPONENTS])
    m = pyr[index]
    pick = lambda key: np.where(m, dot[key], matrix[key])
    Ev = np.where(m, dot['E_vbo'], matrix['E_vbo'])
    return kp.local_band_edges(sub, Ev=Ev, Ec=Ev + pick('Eg'), delta_so=pick('delta_so'),
                               a_c=pick('a_c'), a_v=pick('a_v'), b=pick('b'), d=pick('d'))

in001 = pyr[i0, i0, :]
idx = np.where(in001)[0]
i_base = int(idx[0])                       # lowest layer of island cells, z = +0.2 nm

cut001 = (i0, i0, slice(None))             # [001] through the centre column
cut100 = (slice(None), i0, i_base)         # [100] through the base layer
b001 = bands_on(strain, cut001)
print(f"Kramers degeneracy residual {b001['kramers']:.2e} eV")
print(f"island spans z = {cx[idx[0]]:.1f} .. {cx[idx[-1]]:.1f} nm on this grid")""")

co("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
panels = [(cut001, 'distance along 001  (nm)', (-15, 22),
           '(a) [001], through the centre of the island'),
          (cut100, 'distance along 100  (nm)', (-22, 22),
           f'(b) [100], through the base of the island (z = {cx[i_base]:.1f} nm)')]

for ax, (cut, xlabel, xlim, title) in zip(axes, panels):
    bb = bands_on(strain, cut)
    ax.plot(cx, bb['cb'], 'k-', lw=1.4)
    ax.plot(cx, bb['v1'], 'k-', lw=1.4)
    ax.plot(cx, bb['v2'], '-', color='0.45', lw=1.0)
    ax.plot(cx, bb['v3'], '-', color='0.45', lw=1.0)
    for e in np.flatnonzero(np.diff(pyr[cut].astype(int))):
        ax.axvline((cx[e] + cx[e+1]) / 2, color='0.8', lw=0.7, zorder=0)
    ax.axhline(0.0, color='0.9', lw=0.7, ls=':', zorder=0)
    ax.set(xlim=xlim, ylim=(-0.48, 1.78), xlabel=xlabel, ylabel='E  (eV)')
    ax.set_title(title, fontsize=9)
    span = xlim[1] - xlim[0]
    ax.text(xlim[0] + 0.14*span, 1.63, 'GaAs', ha='center', fontsize=8)
    ax.text(xlim[1] - 0.14*span, 1.63, 'GaAs', ha='center', fontsize=8)
axes[0].text(2.4, 1.63, 'InAs', ha='center', fontsize=8)
axes[1].text(0.0, 1.63, 'InAs', ha='center', fontsize=8)
fig.suptitle('Band structure from the local value of the strain, $b=10$ nm, '
             'INHOMOGENEOUS elasticity — cf. Pryor 1998 Fig. 2', fontsize=10)
fig.tight_layout()
plt.show()""")

md("""## Did the method change close the gap?

Pryor attaches exactly two numbers to this figure (Sec. IV): the conduction well is *"0.4 eV
deep at the base of the island, tapering to 0.27 eV at the tip"*. Those are the targets.

**The apex cannot be read off the last grid cell.** A pyramid tapers to a geometric point, so
the topmost cells are 1–2 wide and their strain is staircase noise, not physics — the taper
below is smooth and monotone, then the last few points scatter by ±0.08 eV. The profile is
printed with the cross-section width beside it so the cut-off is visible rather than chosen
quietly; "tip" below means the highest cell where the island is still at least 8 cells across.""")

co("""depth_fd = GaAs_CB - bands_on(strain, cut001)['cb']
depth_ho = GaAs_CB - bands_on(strain_hom, cut001)['cb']
width = np.array([int(pyr[:, i0, k].sum()) for k in idx])

print(f"{'z (nm)':>7} {'cells across':>13} {'depth FD':>10} {'depth homog':>12}")
for k, w in zip(idx, width):
    flag = '' if w >= 8 else '   <- apex, unresolved'
    print(f"{cx[k]:>7.1f} {w:>13} {depth_fd[k]:>10.3f} {depth_ho[k]:>12.3f}{flag}")

i_tip = int(idx[width >= 8][-1])
print(f"\\n{'':34} {'base':>8} {'tip':>8}")
print(f"{'homogeneous (Fourier, C_GaAs)':34} {depth_ho[i_base]:>8.3f} {depth_ho[i_tip]:>8.3f}")
print(f"{'INHOMOGENEOUS (FD, Pryor method)':34} {depth_fd[i_base]:>8.3f} {depth_fd[i_tip]:>8.3f}")
print(f"{'Pryor 1998, Sec. IV':34} {0.400:>8.3f} {0.270:>8.3f}")
print(f"\\nerror vs Pryor:  homogeneous {depth_ho[i_base]-0.400:+.3f} / {depth_ho[i_tip]-0.270:+.3f}"
      f"   ->   inhomogeneous {depth_fd[i_base]-0.400:+.3f} / {depth_fd[i_tip]-0.270:+.3f} eV")""")

co("""# The four statements Pryor makes about this figure, checked rather than eyeballed.
v = bands_on(strain, cut001)['v1'][idx]
print(f"1. valence peaked near tip, second peak near base, crossing between:")
print(f"   base {v[0]:.3f}, dip {v.min():.3f} at z={cx[idx][np.argmin(v)]:.1f} nm, tip {v[-1]:.3f} eV"
      f"  -> {'yes' if v.min() < min(v[0], v[-1]) else 'NO'}")

full = kp.local_band_edges(strain,
    Ev=np.where(pyr, dot['E_vbo'], matrix['E_vbo']),
    Ec=np.where(pyr, dot['E_vbo']+dot['Eg'], matrix['E_vbo']+matrix['Eg']),
    delta_so=np.where(pyr, dot['delta_so'], matrix['delta_so']),
    a_c=np.where(pyr, dot['a_c'], matrix['a_c']),
    a_v=np.where(pyr, dot['a_v'], matrix['a_v']),
    b=np.where(pyr, dot['b'], matrix['b']), d=np.where(pyr, dot['d'], matrix['d']))
v1 = full['v1']
print(f"2. no hole trapping in the barrier: max inside {v1[pyr].max():.4f} eV vs "
      f"barrier {v1[~pyr].max():.4f} eV -> {'holes stay in the dot' if v1[pyr].max()>v1[~pyr].max() else 'BARRIER TRAP'}")

vb = bands_on(strain, cut100)['v1']
inb = np.where(pyr[cut100])[0]
print(f"3. slight valence peak at the edge of the base: edge {vb[inb[0]]:.4f} vs "
      f"centre {vb[i0]:.4f} eV -> {'yes' if vb[inb[0]] > vb[i0] else 'NO'}")""")

md(r"""---
# Fig. 6 — ground-state exciton wave functions

> **FIG. 6.** Electron and hole wave functions for the ground state exciton in the Hartree
> approximation, with $b = 14$ nm. Surfaces are $\sum_{i=1}^{8}|\psi_i(\vec r)|^2$ equal to 0.1
> of the peak value.

Three things are specified by that caption and are followed exactly: the **eight-band**
Hamiltonian (the sum runs over all eight spinor components), the **Hartree** electron–hole
self-consistency, and an isosurface at **0.1 of the peak** density.

Sec. V on the self-consistency: *"$\psi^e$ and $\psi^h$ were found by self-consistent iteration,
with convergence to within 0.1 meV usually taking only two iterations."* Two Poisson solves per
iteration keep each particle out of its own field.

The strain is the inhomogeneous FD field; the piezoelectric potential is included, as Pryor
states it is. Grid spacing is $h=1$ nm — the eight-band Hamiltonian is 8 unknowns per grid
point and the bound states are interior eigenvalues, so this is the resolution at which the
folded-spectrum solves stay affordable.""")

co("""import pryor_fig6 as f6

env = f6.build(h=1.0, pad=8.0)
t0 = time.time()
res = f6.hartree(env, n_iter=3, k=2, verbose=True)
print(f"\\ntotal {time.time()-t0:.0f}s")""")

co("""h_ = [r for r in res['history']]
print(f"{'iter':>5} {'E_e (eV)':>10} {'E_h (eV)':>10} {'E_int (meV)':>12} {'exciton (eV)':>13}")
for r in h_:
    print(f"{r['it']:>5} {r['E_e']:>10.5f} {r['E_h']:>10.5f} {r['E_int']*1e3:>12.2f} {r['gap']:>13.5f}")
d_last = abs(h_[-1]['gap'] - h_[-2]['gap']) * 1e3
print(f"\\nchange in the exciton energy over the last iteration: {d_last:.2f} meV")
print(f"Coulomb binding (E_int): {-h_[-1]['E_int']*1e3:.1f} meV   "
      f"[Pryor's Fig. 5 reads ~21.5 meV at b = 14 nm]")""")

md("""### The figure""")

co("""fig = f6.plot_wavefunctions(env, res, level=0.1)
plt.show()""")

co("""# Pryor's claim about this figure (Sec. V): "the electron and hole wave functions appear
# ordinary. The wave functions are spread out over most of the island, with no signs of
# localization around smaller regions."
pyr6, h6 = env['pyr'], env['h']
for label, dens in (('electron', res['n']), ('hole', res['p'])):
    inside = dens[pyr6].sum()
    # "spread out over most of the island": what fraction of the island's volume is enclosed by
    # the 0.1-of-peak isosurface?
    occupied = (dens >= 0.1 * dens.max()) & pyr6
    print(f"{label:9}: {inside*100:5.1f}% of the density inside the island; "
          f"the 0.1-peak surface encloses {occupied.sum()/pyr6.sum()*100:5.1f}% of its volume; "
          f"peak at z = {env['cz'][np.unravel_index(dens.argmax(), dens.shape)[2]]:.1f} nm")""")

md("""---
## What is and is not settled

**Settled.** The elasticity is now Pryor's — real-space energy minimization with each material's
own elastic constants. It is validated against the inhomogeneous misfitting-sphere result, which
no homogeneous solver can reproduce, and it reduces to `strain_fourier` in the homogeneous limit
(where the preconditioner is exact and CG converges in a single iteration). The conduction-well
depths move toward Pryor's quoted values by roughly the amount the contrast argument predicted.

**Remaining, in order of size.**

* *Discretization.* The FD strain still differs from the exact-operator Fourier solve by ~2% at
  $h=0.4$ in the homogeneous limit, shrinking roughly linearly in $h$. The faceted $\\{101\\}$
  pyramid is badly represented on a cubic grid: `mask_volume_error` swings by tens of percent
  between spacings with no smooth trend, so grid refinement is not monotone.
* *Fig. 6 is at $h = 1$ nm*, where that discretization error is ~7%. The wave-function *shapes*
  are robust to it; the energies are not, and should not be quoted from this notebook.
* *Linear elasticity* at ~7% mismatch is being pushed, and *continuum* elasticity gives the
  pyramid $C_{4v}$ symmetry rather than the true $C_{2v}$ of the zincblende lattice — only an
  atomistic relaxation recovers that (Pryor, Kim, Wang, Williamson & Zunger, *J. Appl. Phys.*
  **83**, 2548 (1998)).
* *First-order piezoelectricity only.* Now known to be unreliable in these dots — the
  second-order response is comparable and often opposes it (Bester, Zunger and co-workers,
  *Phys. Rev. B* **74**, 081305(R) and *Phys. Rev. Lett.* **96**, 187602, both 2006). Pryor
  predates that work, so first order is what is used here for comparability.
* The **band-ordering puzzle** from earlier work is untouched by this change: we get 6-band and
  8-band close with 4-band the outlier, while Pryor gets 4- and 8-band agreeing to 3 meV with
  6-band ~40 meV away.""")

nb['cells'] = c
nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
nbf.write(nb, os.path.join(ROOT, 'pryor_inhomogeneous.ipynb'))
print(f"wrote pryor_inhomogeneous.ipynb with {len(c)} cells")
